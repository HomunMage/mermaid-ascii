"""Layout module — Sugiyama-style graph layout pipeline.

Phases:
  1. Cycle removal  (this file — greedy-FAS approach)
  2. Layer assignment (rank each node)
  3. Crossing minimization (barycenter heuristic)
  4. Coordinate assignment (x/y positions)

1:1 port of layout.rs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

from mermaid_ascii.graph import EdgeData, NodeData

if TYPE_CHECKING:
    from mermaid_ascii.graph import GraphIR

# ─── Cycle Removal (Greedy-FAS) ───────────────────────────────────────────────


@dataclass
class CycleRemovalResult:
    """Result of cycle removal: a set of (src, tgt) edge tuples that were
    reversed to make the graph a DAG.

    These are "back-edges" in the original graph. The caller can use this set
    to flip arrow directions in the rendering phase (so the displayed arrow
    still points the "right" way visually).

    In Python/networkx we use (src_id, tgt_id) string tuples as edge identifiers
    since networkx doesn't have stable numeric edge indices like petgraph.
    """

    reversed_edges: set[tuple[str, str]] = field(default_factory=set)


def greedy_fas_ordering(graph: nx.DiGraph) -> list[str]:
    """Compute a node ordering using the greedy-FAS heuristic.

    Returns a list of node ids in an ordering that minimizes back-edges.
    Nodes earlier in the ordering should have outgoing edges going forward.

    Algorithm (Eades, Lin, Smyth 1993):
    - Maintain dynamic in/out degree counters updated as nodes are removed.
    - Repeatedly:
        1. Move all sinks (out_deg == 0) to s2.
        2. Move all sources (in_deg == 0) to s1.
        3. Of remaining nodes in cycles, pick max (out - in) and add to s1.
    - Final ordering: s1 + reversed(s2).
    """
    active: set[str] = set(graph.nodes)

    # Dynamic degree counters (count edges among active nodes only).
    out_deg: dict[str, int] = {}
    in_deg: dict[str, int] = {}
    for node in graph.nodes:
        out_deg[node] = graph.out_degree(node)
        in_deg[node] = graph.in_degree(node)

    # s1: nodes placed at the "left" (sources, high out-degree surplus)
    # s2: nodes placed at the "right" (sinks)
    s1: list[str] = []
    s2: list[str] = []

    while active:
        # Step 1: Pull all sinks (out_deg == 0) into s2.
        changed = True
        while changed:
            changed = False
            sinks = [n for n in active if out_deg[n] == 0]
            if sinks:
                changed = True
                for sink in sinks:
                    active.remove(sink)
                    s2.append(sink)
                    for pred in graph.predecessors(sink):
                        if pred in active:
                            out_deg[pred] -= 1

        # Step 2: Pull all sources (in_deg == 0) into s1.
        changed = True
        while changed:
            changed = False
            sources = [n for n in active if in_deg[n] == 0]
            if sources:
                changed = True
                for source in sources:
                    active.remove(source)
                    s1.append(source)
                    for succ in graph.successors(source):
                        if succ in active:
                            in_deg[succ] -= 1

        # Step 3: If nodes remain (in cycles), pick max (out - in) node.
        if active:
            best = max(active, key=lambda n: out_deg[n] - in_deg[n])
            active.remove(best)
            s1.append(best)
            for succ in graph.successors(best):
                if succ in active:
                    in_deg[succ] -= 1
            for pred in graph.predecessors(best):
                if pred in active:
                    out_deg[pred] -= 1

    # Final ordering: s1 + reversed(s2)
    s2.reverse()
    s1.extend(s2)
    return s1


def remove_cycles(graph: nx.DiGraph) -> tuple[nx.DiGraph, set[tuple[str, str]]]:
    """Remove cycles from a copy of the DiGraph using the greedy-FAS heuristic.

    Returns a tuple of:
    - new_graph: copy of graph with back-edges reversed (self-loops removed)
    - reversed_edges: set of (src_id, tgt_id) tuples that were reversed
      (identified relative to the ORIGINAL graph's edge directions)

    Back-edges are edges where source comes AFTER target in the greedy-FAS
    ordering, or self-loops (which are removed entirely from the DAG).
    """
    if graph.number_of_nodes() == 0:
        return graph.copy(), set()

    # Build node ordering via greedy-FAS.
    ordering = greedy_fas_ordering(graph)

    # Build position map: node_id → position in the ordering.
    position: dict[str, int] = {node: pos for pos, node in enumerate(ordering)}

    # Identify back-edges: edges where source comes AFTER target in ordering,
    # or self-loops (source == target).
    reversed_edges: set[tuple[str, str]] = set()
    for src, tgt in graph.edges():
        is_self_loop = src == tgt
        src_pos = position[src]
        tgt_pos = position[tgt]
        if is_self_loop or src_pos > tgt_pos:
            reversed_edges.add((src, tgt))

    # Build the modified graph with back-edges reversed.
    new_graph: nx.DiGraph = nx.DiGraph()

    # Add all nodes preserving node data.
    for node_id in graph.nodes:
        new_graph.add_node(node_id, **graph.nodes[node_id])

    # Add edges, reversing back-edges. Skip self-loops entirely.
    for src, tgt, edge_attrs in graph.edges(data=True):
        if src == tgt:
            # Self-loop: omit from the DAG entirely.
            continue
        if (src, tgt) in reversed_edges:
            new_graph.add_edge(tgt, src, **edge_attrs)
        else:
            new_graph.add_edge(src, tgt, **edge_attrs)

    return new_graph, reversed_edges


# ─── Layer Assignment ─────────────────────────────────────────────────────────


class LayerAssignment:
    """Result of layer assignment: each node is assigned a layer (rank).

    Layer 0 is the "first" layer (top for TD, left for LR).
    This is computed on a cycle-free copy of the graph produced by
    ``remove_cycles``.

    Attributes:
        layers: Maps node id → layer index.
        layer_count: Total number of layers.
        reversed_edges: Edges reversed during cycle removal (as (src, tgt) pairs).
    """

    def __init__(
        self,
        layers: dict[str, int],
        layer_count: int,
        reversed_edges: set[tuple[str, str]],
    ) -> None:
        self.layers = layers
        self.layer_count = layer_count
        self.reversed_edges = reversed_edges

    @classmethod
    def assign(cls, gir: GraphIR) -> LayerAssignment:
        """Assign layers to all nodes using fixed-point iteration.

        Algorithm: for each edge u→v in the DAG, rank[v] = max(rank[v], rank[u]+1).
        Repeat until stable. Runs in O(V * E) worst case, fast in practice.
        """
        dag, reversed_edges = remove_cycles(gir.digraph)

        # Initialize all layers to 0.
        layers: dict[str, int] = {node_id: 0 for node_id in gir.digraph.nodes}

        # Fixed-point iteration: propagate ranks along DAG edges.
        changed = True
        while changed:
            changed = False
            for src, tgt in dag.edges():
                src_rank = layers[src]
                tgt_rank = layers[tgt]
                if tgt_rank < src_rank + 1:
                    layers[tgt] = src_rank + 1
                    changed = True

        layer_count = (max(layers.values()) + 1) if layers else 1

        return cls(layers=layers, layer_count=layer_count, reversed_edges=reversed_edges)


# ─── Dummy Node Insertion ──────────────────────────────────────────────────────

DUMMY_PREFIX = "__dummy_"


@dataclass
class DummyEdge:
    """The result of dummy node insertion for a single long edge.

    When an edge spans more than one layer (layer[tgt] - layer[src] > 1),
    it is replaced by a chain of dummy nodes — one per intermediate layer.
    Each dummy node gets a synthetic id starting with DUMMY_PREFIX.
    """

    original_src: str
    original_tgt: str
    dummy_ids: list[str]
    edge_data: EdgeData


@dataclass
class AugmentedGraph:
    """A graph augmented with dummy nodes for edges that span multiple layers.

    After dummy node insertion, every edge in the augmented graph connects
    nodes in adjacent layers (layer difference == 1). This is a pre-condition
    for crossing minimisation and coordinate assignment.
    """

    graph: nx.DiGraph
    layers: dict[str, int]
    layer_count: int
    dummy_edges: list[DummyEdge]


def insert_dummy_nodes(dag: nx.DiGraph, la: LayerAssignment) -> AugmentedGraph:
    """Insert dummy nodes into the cycle-free, layer-assigned graph.

    For each edge (u → v) where layer[v] - layer[u] > 1, the edge is removed
    and replaced by the chain:
        u → d₁ → d₂ → … → dₖ → v
    where each dᵢ lives in layer ``layer[u] + i``.

    Args:
        dag: The cycle-free DiGraph produced by ``remove_cycles``.
        la:  The layer assignment produced by ``LayerAssignment.assign``.

    Returns:
        An ``AugmentedGraph`` where every edge connects adjacent-layer nodes.
    """
    from mermaid_ascii import ast as mast

    # Build a new graph; copy all original nodes preserving node attributes.
    g: nx.DiGraph = nx.DiGraph()
    for node_id in dag.nodes:
        g.add_node(node_id, **dag.nodes[node_id])

    # Layer map: extended with dummy nodes as they are inserted.
    layers: dict[str, int] = copy.copy(la.layers)

    dummy_edges: list[DummyEdge] = []
    edge_counter = 0

    # Collect all edges up-front so iteration is over a stable snapshot.
    all_edges: list[tuple[str, str, EdgeData]] = [
        (src, tgt, attrs.get("data")) for src, tgt, attrs in dag.edges(data=True)
    ]

    for src_id, tgt_id, edge_data in all_edges:
        src_layer = layers[src_id]
        tgt_layer = layers[tgt_id]

        # Edges going "upward" (reversed back-edge in display) — treat as span 1.
        layer_diff = tgt_layer - src_layer if tgt_layer > src_layer else 1

        if layer_diff <= 1:
            # Adjacent-layer edge — copy as-is.
            g.add_edge(src_id, tgt_id, data=edge_data)
            continue

        # Long edge: replace with a chain of dummy nodes.
        steps = layer_diff - 1  # number of intermediate layers
        this_edge = edge_counter
        edge_counter += 1

        dummy_ids: list[str] = []
        chain_prev = src_id

        for i in range(steps):
            dummy_layer = src_layer + i + 1
            dummy_id = f"{DUMMY_PREFIX}{this_edge}_{i}"

            dummy_data = NodeData(
                id=dummy_id,
                label="",
                shape=mast.NodeShape.Rectangle,
                attrs=[],
                subgraph=None,
            )
            g.add_node(dummy_id, data=dummy_data)
            layers[dummy_id] = dummy_layer
            dummy_ids.append(dummy_id)

            # Segment edge from previous node to this dummy (no label).
            segment_edge = EdgeData(
                edge_type=edge_data.edge_type if edge_data else mast.EdgeType.Arrow,
                label=None,
                attrs=[],
            )
            g.add_edge(chain_prev, dummy_id, data=segment_edge)
            chain_prev = dummy_id

        # Final segment: last dummy → original target, carry the label.
        last_segment = EdgeData(
            edge_type=edge_data.edge_type if edge_data else mast.EdgeType.Arrow,
            label=edge_data.label if edge_data else None,
            attrs=edge_data.attrs if edge_data else [],
        )
        g.add_edge(chain_prev, tgt_id, data=last_segment)

        dummy_edges.append(
            DummyEdge(
                original_src=src_id,
                original_tgt=tgt_id,
                dummy_ids=dummy_ids,
                edge_data=edge_data,
            )
        )

    layer_count = (max(layers.values()) + 1) if layers else 1

    return AugmentedGraph(
        graph=g,
        layers=layers,
        layer_count=layer_count,
        dummy_edges=dummy_edges,
    )


# ─── Crossing Minimization (Barycenter) ───────────────────────────────────────

# Character-unit geometry constants (TD layout).
NODE_PADDING: int = 1  # spaces inside brackets on each side of label
H_GAP: int = 4  # horizontal gap (chars) between nodes in same layer
V_GAP: int = 3  # vertical gap (rows) between adjacent layers
NODE_HEIGHT: int = 3  # top-border + text-row + bottom-border


@dataclass
class LayoutNode:
    """A positioned node in the layout."""

    id: str
    layer: int
    order: int
    x: int
    y: int
    width: int
    height: int


def minimise_crossings(aug: AugmentedGraph) -> list[list[str]]:
    """Minimise edge crossings using the barycenter heuristic.

    Takes the augmented graph (with dummy nodes) and returns an ordering for
    each layer that reduces edge crossings. Multiple top-down + bottom-up passes
    are run until the crossing count stops improving (or a pass limit is hit).

    Returns a list[list[str]] — one inner list per layer, in minimised order.
    """
    layer_count = aug.layer_count

    # Initial ordering: group by layer, sort alphabetically for determinism.
    ordering: list[list[str]] = [[] for _ in range(layer_count)]
    for node_id in sorted(aug.layers.keys()):
        layer = aug.layers[node_id]
        ordering[layer].append(node_id)

    max_passes = 24
    best = count_crossings(ordering, aug.graph)

    for _pass in range(max_passes):
        # Top-down sweep: use predecessor positions as barycenter weights.
        for layer_idx in range(1, layer_count):
            prev_ids = ordering[layer_idx - 1]
            prev: dict[str, float] = {nid: float(i) for i, nid in enumerate(prev_ids)}
            ordering[layer_idx].sort(key=lambda a, p=prev: _barycenter(a, aug.graph, p, "incoming"))

        # Bottom-up sweep: use successor positions as barycenter weights.
        for layer_idx in range(max(0, layer_count - 2), -1, -1):
            next_ids = ordering[layer_idx + 1]
            nxt: dict[str, float] = {nid: float(i) for i, nid in enumerate(next_ids)}
            ordering[layer_idx].sort(key=lambda a, n=nxt: _barycenter(a, aug.graph, n, "outgoing"))

        new = count_crossings(ordering, aug.graph)
        if new >= best:
            break
        best = new

    return ordering


def _barycenter(
    node_id: str,
    graph: nx.DiGraph,
    neighbor_pos: dict[str, float],
    direction: str,
) -> float:
    """Average position of a node's neighbours in the adjacent layer (barycenter weight).

    direction: "incoming" to look at predecessors, "outgoing" for successors.
    Returns float('inf') if the node has no neighbours in the adjacent layer.
    """
    if node_id not in graph:
        return float("inf")

    neighbors = list(graph.predecessors(node_id)) if direction == "incoming" else list(graph.successors(node_id))

    positions = [neighbor_pos[nb] for nb in neighbors if nb in neighbor_pos]
    if not positions:
        return float("inf")
    return sum(positions) / len(positions)


def count_crossings(ordering: list[list[str]], graph: nx.DiGraph) -> int:
    """Count edge crossings between consecutive layers (inversion count heuristic)."""
    total = 0
    for l_idx in range(len(ordering) - 1):
        tgt_pos: dict[str, int] = {nid: i for i, nid in enumerate(ordering[l_idx + 1])}
        edges: list[tuple[int, int]] = []
        for sp, src_id in enumerate(ordering[l_idx]):
            if src_id in graph:
                for nb in graph.successors(src_id):
                    if nb in tgt_pos:
                        edges.append((sp, tgt_pos[nb]))
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                ei, ej = edges[i], edges[j]
                if (ei[0] < ej[0] and ei[1] > ej[1]) or (ei[0] > ej[0] and ei[1] < ej[1]):
                    total += 1
    return total


# ─── Coordinate Assignment ────────────────────────────────────────────────────


def label_dimensions(label: str) -> tuple[int, int]:
    """Compute (max_line_width, line_count) for a label that may contain newlines."""
    if not label:
        return (0, 1)
    lines = label.split("\n")
    max_w = max(len(line) for line in lines)
    return (max_w, len(lines))


def assign_coordinates(ordering: list[list[str]], aug: AugmentedGraph) -> list[LayoutNode]:
    """Assign (x, y) character coordinates to every node in the augmented graph.

    Layout is top-down (TD): x = column, y = row. The renderer transposes for LR.
    Dummy nodes are given width 1 to minimise horizontal space consumption.
    """
    from mermaid_ascii import ast as mast

    return assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, mast.Direction.TD)


def assign_coordinates_padded(
    ordering: list[list[str]],
    aug: AugmentedGraph,
    padding: int,
    size_overrides: dict[str, tuple[int, int]],
    direction: object,
) -> list[LayoutNode]:
    """Internal: coordinate assignment with caller-specified padding value and direction.

    size_overrides maps node id → (width, height) for compound nodes or
    other nodes whose dimensions can't be computed from the label alone.

    For LR/RL directions, node width↔height are swapped before layout so
    Sugiyama produces a rotated arrangement. The renderer then transposes
    (x↔y) to produce the final left-to-right output.
    """
    from mermaid_ascii import ast as mast

    is_lr_or_rl = direction in (mast.Direction.LR, mast.Direction.RL)

    # When LR/RL: swap H_GAP↔V_GAP so inter-layer spacing is applied correctly.
    h_gap = V_GAP if is_lr_or_rl else H_GAP
    v_gap = H_GAP if is_lr_or_rl else V_GAP

    # Build label info map: id -> (max_line_width, line_count)
    id_to_label_info: dict[str, tuple[int, int]] = {}
    for node_id in aug.graph.nodes:
        node_attrs = aug.graph.nodes[node_id]
        node_data: NodeData | None = node_attrs.get("data")
        if node_data is not None:
            id_to_label_info[node_id] = label_dimensions(node_data.label)
        else:
            id_to_label_info[node_id] = (len(node_id), 1)

    def node_dims(node_id: str) -> tuple[int, int]:
        """Compute (width, height) for a node, respecting overrides."""
        if node_id in size_overrides:
            dims = size_overrides[node_id]
            return (dims[1], dims[0]) if is_lr_or_rl else dims
        max_line_w, line_count = id_to_label_info.get(node_id, (0, 1))
        is_dummy = max_line_w == 0 and node_id.startswith(DUMMY_PREFIX)
        width = 1 if is_dummy else max_line_w + 2 + 2 * padding
        height = NODE_HEIGHT if is_dummy else 2 + line_count
        # For LR/RL: swap so layout treats width as the cross-axis dimension.
        if is_lr_or_rl:
            return (height, width)
        return (width, height)

    # First pass: compute per-layer max height.
    layer_max_height: list[int] = [NODE_HEIGHT] * len(ordering)
    for layer_idx, layer_nodes in enumerate(ordering):
        for node_id in layer_nodes:
            _, h = node_dims(node_id)
            if h > layer_max_height[layer_idx]:
                layer_max_height[layer_idx] = h

    # Compute layer Y offsets using actual max heights.
    layer_y: list[int] = []
    y = 0
    for h in layer_max_height:
        layer_y.append(y)
        y += h + v_gap

    # Compute total width per layer for centering.
    layer_total_widths: list[int] = []
    for layer_nodes in ordering:
        w_sum = sum(node_dims(nid)[0] for nid in layer_nodes)
        gaps = (len(layer_nodes) - 1) * h_gap if len(layer_nodes) > 1 else 0
        layer_total_widths.append(w_sum + gaps)

    max_layer_w = max(layer_total_widths, default=0)
    center_col = max_layer_w // 2

    nodes: list[LayoutNode] = []
    for layer_idx, layer_nodes in enumerate(ordering):
        # Center this layer's midpoint on center_col.
        offset = max(0, center_col - layer_total_widths[layer_idx] // 2)
        x = offset
        for order, node_id in enumerate(layer_nodes):
            width, height = node_dims(node_id)
            nodes.append(
                LayoutNode(
                    id=node_id,
                    layer=layer_idx,
                    order=order,
                    x=x,
                    y=layer_y[layer_idx],
                    width=width,
                    height=height,
                )
            )
            x += width + h_gap

    # ── Barycenter refinement: align layers with parent/child centers ──
    #
    # After initial center-based placement, shift each layer so the average
    # center of its nodes aligns with the average center of their parents
    # (top-down) or children (bottom-up). Only small shifts (≤ h_gap) are
    # applied to correct integer-rounding misalignment.

    node_idx: dict[str, int] = {n.id: i for i, n in enumerate(nodes)}

    # Top-down pass: align each layer under its parent centers.
    for layer_idx in range(1, len(ordering)):
        sum_child = 0
        sum_parent = 0
        count = 0

        for node_id in ordering[layer_idx]:
            ni = node_idx[node_id]
            child_center = nodes[ni].x + nodes[ni].width // 2

            for src, tgt in aug.graph.edges():
                if tgt == node_id and not src.startswith(DUMMY_PREFIX) and src in node_idx:
                    pi = node_idx[src]
                    if nodes[pi].layer + 1 == layer_idx:
                        parent_center = nodes[pi].x + nodes[pi].width // 2
                        sum_child += child_center
                        sum_parent += parent_center
                        count += 1

        if count == 0:
            continue
        shift = sum_parent // count - sum_child // count
        if abs(shift) > h_gap:
            continue

        for node_id in ordering[layer_idx]:
            ni = node_idx[node_id]
            nodes[ni].x = max(0, nodes[ni].x + shift)

    # Bottom-up pass: align each layer over its child centers.
    for layer_idx in range(max(0, len(ordering) - 2), -1, -1):
        sum_node = 0
        sum_child = 0
        count = 0

        for node_id in ordering[layer_idx]:
            ni = node_idx[node_id]
            node_center = nodes[ni].x + nodes[ni].width // 2

            for src, tgt in aug.graph.edges():
                if src == node_id and not tgt.startswith(DUMMY_PREFIX) and tgt in node_idx:
                    ci = node_idx[tgt]
                    if nodes[ci].layer == layer_idx + 1:
                        child_center = nodes[ci].x + nodes[ci].width // 2
                        sum_node += node_center
                        sum_child += child_center
                        count += 1

        if count == 0:
            continue
        shift = sum_child // count - sum_node // count
        if abs(shift) > h_gap:
            continue

        for node_id in ordering[layer_idx]:
            ni = node_idx[node_id]
            nodes[ni].x = max(0, nodes[ni].x + shift)

    # Normalize: shift everything so the leftmost node starts at x=0.
    if nodes:
        min_x = min(n.x for n in nodes)
        if min_x > 0:
            for n in nodes:
                n.x -= min_x

    return nodes
