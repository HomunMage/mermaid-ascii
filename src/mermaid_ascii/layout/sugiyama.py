"""Sugiyama-style layered graph layout engine.

Phases:
  1. Cycle removal (greedy-FAS)
  2. Layer assignment
  3. Dummy node insertion
  4. Crossing minimization (barycenter)
  5. Coordinate assignment
  6. Edge routing (orthogonal)
  7. Subgraph collapse/expand
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import networkx as nx

from mermaid_ascii.layout.types import COMPOUND_PREFIX, DUMMY_PREFIX, LayoutNode, Point, RoutedEdge
from mermaid_ascii.syntax.graph import EdgeData, GraphIR, NodeData
from mermaid_ascii.types import Direction, EdgeType, NodeShape

# ─── Geometry constants ──────────────────────────────────────────────────────

NODE_PADDING: int = 1
H_GAP: int = 4
V_GAP: int = 3
NODE_HEIGHT: int = 3


# ─── Cycle Removal (Greedy-FAS) ─────────────────────────────────────────────


@dataclass
class CycleRemovalResult:
    reversed_edges: set[tuple[str, str]] = field(default_factory=set)


def greedy_fas_ordering(graph: nx.DiGraph) -> list[str]:
    """Compute a node ordering using the greedy-FAS heuristic."""
    active: set[str] = set(graph.nodes)
    out_deg: dict[str, int] = {}
    in_deg: dict[str, int] = {}
    for node in graph.nodes:
        out_deg[node] = graph.out_degree(node)
        in_deg[node] = graph.in_degree(node)

    s1: list[str] = []
    s2: list[str] = []

    while active:
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

    s2.reverse()
    s1.extend(s2)
    return s1


def remove_cycles(graph: nx.DiGraph) -> tuple[nx.DiGraph, set[tuple[str, str]]]:
    """Remove cycles using greedy-FAS. Returns (dag, reversed_edges)."""
    if graph.number_of_nodes() == 0:
        return graph.copy(), set()

    ordering = greedy_fas_ordering(graph)
    position: dict[str, int] = {node: pos for pos, node in enumerate(ordering)}

    reversed_edges: set[tuple[str, str]] = set()
    for src, tgt in graph.edges():
        is_self_loop = src == tgt
        if is_self_loop or position[src] > position[tgt]:
            reversed_edges.add((src, tgt))

    new_graph: nx.DiGraph = nx.DiGraph()
    for node_id in graph.nodes:
        new_graph.add_node(node_id, **graph.nodes[node_id])

    for src, tgt, edge_attrs in graph.edges(data=True):
        if src == tgt:
            continue
        if (src, tgt) in reversed_edges:
            new_graph.add_edge(tgt, src, **edge_attrs)
        else:
            new_graph.add_edge(src, tgt, **edge_attrs)

    return new_graph, reversed_edges


# ─── Layer Assignment ────────────────────────────────────────────────────────


class LayerAssignment:
    def __init__(self, layers: dict[str, int], layer_count: int, reversed_edges: set[tuple[str, str]]) -> None:
        self.layers = layers
        self.layer_count = layer_count
        self.reversed_edges = reversed_edges

    @classmethod
    def assign(cls, gir: GraphIR) -> LayerAssignment:
        dag, reversed_edges = remove_cycles(gir.digraph)
        layers: dict[str, int] = {node_id: 0 for node_id in gir.digraph.nodes}

        changed = True
        while changed:
            changed = False
            for src, tgt in dag.edges():
                if layers[tgt] < layers[src] + 1:
                    layers[tgt] = layers[src] + 1
                    changed = True

        layer_count = (max(layers.values()) + 1) if layers else 1
        return cls(layers=layers, layer_count=layer_count, reversed_edges=reversed_edges)


# ─── Dummy Node Insertion ────────────────────────────────────────────────────


@dataclass
class DummyEdge:
    original_src: str
    original_tgt: str
    dummy_ids: list[str]
    edge_data: EdgeData


@dataclass
class AugmentedGraph:
    graph: nx.DiGraph
    layers: dict[str, int]
    layer_count: int
    dummy_edges: list[DummyEdge]


def insert_dummy_nodes(dag: nx.DiGraph, la: LayerAssignment) -> AugmentedGraph:
    """Insert dummy nodes for edges spanning multiple layers."""
    g: nx.DiGraph = nx.DiGraph()
    for node_id in dag.nodes:
        g.add_node(node_id, **dag.nodes[node_id])

    layers: dict[str, int] = copy.copy(la.layers)
    dummy_edges: list[DummyEdge] = []
    edge_counter = 0

    all_edges: list[tuple[str, str, EdgeData]] = [
        (src, tgt, attrs.get("data")) for src, tgt, attrs in dag.edges(data=True)
    ]

    for src_id, tgt_id, edge_data in all_edges:
        src_layer = layers[src_id]
        tgt_layer = layers[tgt_id]
        layer_diff = tgt_layer - src_layer if tgt_layer > src_layer else 1

        if layer_diff <= 1:
            g.add_edge(src_id, tgt_id, data=edge_data)
            continue

        steps = layer_diff - 1
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
                shape=NodeShape.Rectangle,
                attrs=[],
                subgraph=None,
            )
            g.add_node(dummy_id, data=dummy_data)
            layers[dummy_id] = dummy_layer
            dummy_ids.append(dummy_id)

            segment_edge = EdgeData(
                edge_type=edge_data.edge_type if edge_data else EdgeType.Arrow,
                label=None,
                attrs=[],
            )
            g.add_edge(chain_prev, dummy_id, data=segment_edge)
            chain_prev = dummy_id

        last_segment = EdgeData(
            edge_type=edge_data.edge_type if edge_data else EdgeType.Arrow,
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
    return AugmentedGraph(graph=g, layers=layers, layer_count=layer_count, dummy_edges=dummy_edges)


# ─── Crossing Minimization ───────────────────────────────────────────────────


def minimise_crossings(aug: AugmentedGraph) -> list[list[str]]:
    """Minimise edge crossings using barycenter heuristic."""
    layer_count = aug.layer_count
    ordering: list[list[str]] = [[] for _ in range(layer_count)]
    for node_id in sorted(aug.layers.keys()):
        layer = aug.layers[node_id]
        ordering[layer].append(node_id)

    max_passes = 24
    best = count_crossings(ordering, aug.graph)

    for _pass in range(max_passes):
        for layer_idx in range(1, layer_count):
            prev_ids = ordering[layer_idx - 1]
            prev: dict[str, float] = {nid: float(i) for i, nid in enumerate(prev_ids)}
            ordering[layer_idx].sort(key=lambda a, p=prev: _barycenter(a, aug.graph, p, "incoming"))

        for layer_idx in range(layer_count - 2, -1, -1):
            next_ids = ordering[layer_idx + 1]
            nxt: dict[str, float] = {nid: float(i) for i, nid in enumerate(next_ids)}
            ordering[layer_idx].sort(key=lambda a, n=nxt: _barycenter(a, aug.graph, n, "outgoing"))

        new = count_crossings(ordering, aug.graph)
        if new >= best:
            break
        best = new

    return ordering


def _barycenter(node_id: str, graph: nx.DiGraph, neighbor_pos: dict[str, float], direction: str) -> float:
    if node_id not in graph:
        return float("inf")
    neighbors = list(graph.predecessors(node_id)) if direction == "incoming" else list(graph.successors(node_id))
    positions = [neighbor_pos[nb] for nb in neighbors if nb in neighbor_pos]
    if not positions:
        return float("inf")
    return sum(positions) / len(positions)


def count_crossings(ordering: list[list[str]], graph: nx.DiGraph) -> int:
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


# ─── Coordinate Assignment ───────────────────────────────────────────────────


def label_dimensions(label: str) -> tuple[int, int]:
    if not label:
        return (0, 1)
    lines = label.split("\n")
    max_w = max(len(line) for line in lines)
    return (max_w, len(lines))


def assign_coordinates_padded(
    ordering: list[list[str]],
    aug: AugmentedGraph,
    padding: int,
    size_overrides: dict[str, tuple[int, int]],
    direction: object,
) -> list[LayoutNode]:
    """Assign (x, y) character coordinates to every node."""
    is_lr_or_rl = direction in (Direction.LR, Direction.RL)
    h_gap = V_GAP if is_lr_or_rl else H_GAP
    v_gap = H_GAP if is_lr_or_rl else V_GAP

    id_to_label_info: dict[str, tuple[int, int]] = {}
    for node_id in aug.graph.nodes:
        node_attrs = aug.graph.nodes[node_id]
        node_data: NodeData | None = node_attrs.get("data")
        if node_data is not None:
            id_to_label_info[node_id] = label_dimensions(node_data.label)
        else:
            id_to_label_info[node_id] = (len(node_id), 1)

    def node_dims(node_id: str) -> tuple[int, int]:
        if node_id in size_overrides:
            dims = size_overrides[node_id]
            return (dims[1], dims[0]) if is_lr_or_rl else dims
        max_line_w, line_count = id_to_label_info.get(node_id, (0, 1))
        is_dummy = max_line_w == 0 and node_id.startswith(DUMMY_PREFIX)
        width = 1 if is_dummy else max_line_w + 2 + 2 * padding
        height = NODE_HEIGHT if is_dummy else 2 + line_count
        if is_lr_or_rl:
            return (height, width)
        return (width, height)

    layer_max_height: list[int] = [NODE_HEIGHT] * len(ordering)
    for layer_idx, layer_nodes in enumerate(ordering):
        for node_id in layer_nodes:
            _, h = node_dims(node_id)
            if h > layer_max_height[layer_idx]:
                layer_max_height[layer_idx] = h

    layer_y: list[int] = []
    y = 0
    for h in layer_max_height:
        layer_y.append(y)
        y += h + v_gap

    layer_total_widths: list[int] = []
    for layer_nodes in ordering:
        w_sum = sum(node_dims(nid)[0] for nid in layer_nodes)
        gaps = (len(layer_nodes) - 1) * h_gap if len(layer_nodes) > 1 else 0
        layer_total_widths.append(w_sum + gaps)

    max_layer_w = max(layer_total_widths, default=0)
    center_col = max_layer_w // 2

    nodes: list[LayoutNode] = []
    for layer_idx, layer_nodes in enumerate(ordering):
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

    # Barycenter refinement
    node_idx: dict[str, int] = {n.id: i for i, n in enumerate(nodes)}

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

    if nodes:
        min_x = min(n.x for n in nodes)
        if min_x > 0:
            for n in nodes:
                n.x -= min_x

    return nodes


# ─── Edge Routing ────────────────────────────────────────────────────────────


def route_edges(
    gir: GraphIR,
    layout_nodes: list[LayoutNode],
    aug: AugmentedGraph,
    reversed_edges: set[tuple[str, str]],
) -> list[RoutedEdge]:
    """Route all edges orthogonally through inter-layer gaps."""
    node_map: dict[str, LayoutNode] = {n.id: n for n in layout_nodes}

    layer_count = max((n.layer for n in layout_nodes), default=-1) + 1
    layer_top_y: list[int] = [10**9] * max(layer_count, 1)
    layer_bottom_y: list[int] = [0] * max(layer_count, 1)
    for n in layout_nodes:
        if n.y < layer_top_y[n.layer]:
            layer_top_y[n.layer] = n.y
        bot = n.y + n.height
        if bot > layer_bottom_y[n.layer]:
            layer_bottom_y[n.layer] = bot

    dummy_xs_map: dict[tuple[str, str], list[int]] = {}
    for de in aug.dummy_edges:
        xs = [node_map[did].x + node_map[did].width // 2 for did in de.dummy_ids if did in node_map]
        dummy_xs_map[(de.original_src, de.original_tgt)] = xs

    routes: list[RoutedEdge] = []

    for src, tgt, edge_attrs in gir.digraph.edges(data=True):
        if src == tgt:
            continue

        edge_data = edge_attrs.get("data")
        is_reversed = (src, tgt) in reversed_edges

        if is_reversed:
            vis_from, vis_to = tgt, src
        else:
            vis_from, vis_to = src, tgt

        from_node = node_map.get(vis_from)
        to_node = node_map.get(vis_to)
        if from_node is None or to_node is None:
            continue

        dummy_xs = dummy_xs_map.get((vis_from, vis_to), [])

        waypoints = _compute_orthogonal_waypoints(from_node, to_node, layer_top_y, layer_bottom_y, dummy_xs)

        label = edge_data.label if edge_data else None
        edge_type = edge_data.edge_type if edge_data else None

        routes.append(RoutedEdge(from_id=vis_from, to_id=vis_to, label=label, edge_type=edge_type, waypoints=waypoints))

    return routes


def _compute_orthogonal_waypoints(
    from_node: LayoutNode,
    to_node: LayoutNode,
    layer_top_y: list[int],
    layer_bottom_y: list[int],
    dummy_xs: list[int],
) -> list[Point]:
    exit_x = from_node.x + from_node.width // 2
    exit_y = from_node.y + from_node.height - 1
    entry_x = to_node.x + to_node.width // 2
    entry_y = to_node.y

    src_layer = from_node.layer
    tgt_layer = to_node.layer

    if src_layer == tgt_layer:
        bot = layer_bottom_y[src_layer] if src_layer < len(layer_bottom_y) else exit_y + 1
        below_y = bot + V_GAP // 2
        return [
            Point(x=exit_x, y=exit_y),
            Point(x=exit_x, y=below_y),
            Point(x=entry_x, y=below_y),
            Point(x=entry_x, y=entry_y),
        ]

    low_layer = min(src_layer, tgt_layer)
    high_layer = max(src_layer, tgt_layer)

    waypoints: list[Point] = [Point(x=exit_x, y=exit_y)]

    gaps = high_layer - low_layer
    for gap_idx in range(gaps):
        gap = low_layer + gap_idx
        gap_start = layer_bottom_y[gap] if gap < len(layer_bottom_y) else exit_y + 1
        gap_end = layer_top_y[gap + 1] if gap + 1 < len(layer_top_y) else gap_start + V_GAP
        mid_y = gap_start + max(0, gap_end - gap_start) // 2

        if gap_idx < len(dummy_xs):
            gap_x = dummy_xs[gap_idx]
        elif gap_idx == 0:
            gap_x = exit_x
        else:
            gap_x = entry_x

        last_wp = waypoints[-1]
        if last_wp.x != gap_x:
            waypoints.append(Point(x=gap_x, y=last_wp.y))
        waypoints.append(Point(x=gap_x, y=mid_y))

    last_wp = waypoints[-1]
    if last_wp.x != entry_x:
        waypoints.append(Point(x=entry_x, y=last_wp.y))
    waypoints.append(Point(x=entry_x, y=entry_y))

    return waypoints


# ─── Compound Node (Subgraph Collapse/Expand) ───────────────────────────────

_SG_INNER_GAP: int = 1
_SG_PAD_X: int = 1


@dataclass
class CompoundInfo:
    sg_name: str
    compound_id: str
    member_ids: list[str]
    member_widths: list[int]
    member_heights: list[int]
    max_member_height: int
    description: str | None


def collapse_subgraphs(gir: GraphIR, padding: int) -> tuple[GraphIR, list[CompoundInfo]]:
    """Collapse subgraphs into compound nodes for layout."""
    member_to_sg: dict[str, str] = {}
    compounds: list[CompoundInfo] = []

    for sg_name, members in gir.subgraph_members:
        compound_id = f"{COMPOUND_PREFIX}{sg_name}"
        member_widths: list[int] = []
        member_heights: list[int] = []

        for mid in members:
            if mid in gir.digraph.nodes:
                node_attrs = gir.digraph.nodes[mid]
                data: NodeData | None = node_attrs.get("data")
                if data is not None:
                    max_line_w, line_count = label_dimensions(data.label)
                    member_widths.append(max_line_w + 2 + 2 * padding)
                    member_heights.append(2 + line_count)
                else:
                    member_widths.append(3 + 2 * padding)
                    member_heights.append(NODE_HEIGHT)
            else:
                member_widths.append(3 + 2 * padding)
                member_heights.append(NODE_HEIGHT)
            member_to_sg[mid] = sg_name

        max_member_height = max(member_heights, default=NODE_HEIGHT)
        description = gir.subgraph_descriptions.get(sg_name)

        compounds.append(
            CompoundInfo(
                sg_name=sg_name,
                compound_id=compound_id,
                member_ids=list(members),
                member_widths=member_widths,
                member_heights=member_heights,
                max_member_height=max_member_height,
                description=description,
            )
        )

    sg_to_compound: dict[str, str] = {c.sg_name: c.compound_id for c in compounds}

    def resolve_endpoint(node_id: str) -> str:
        if node_id in member_to_sg:
            return sg_to_compound[member_to_sg[node_id]]
        if node_id in sg_to_compound:
            return sg_to_compound[node_id]
        return node_id

    new_digraph: nx.DiGraph = nx.DiGraph()

    for node_id in gir.digraph.nodes:
        if node_id in member_to_sg:
            continue
        if node_id in sg_to_compound:
            continue
        new_digraph.add_node(node_id, **gir.digraph.nodes[node_id])

    for ci in compounds:
        compound_data = NodeData(
            id=ci.compound_id,
            label=ci.sg_name,
            shape=NodeShape.Rectangle,
            attrs=[],
            subgraph=None,
        )
        new_digraph.add_node(ci.compound_id, data=compound_data)

    added_edges: set[tuple[str, str]] = set()
    for src, tgt, edge_attrs in gir.digraph.edges(data=True):
        actual_src = resolve_endpoint(src)
        actual_tgt = resolve_endpoint(tgt)
        if actual_src == actual_tgt:
            continue
        key = (actual_src, actual_tgt)
        if key in added_edges:
            continue
        added_edges.add(key)
        new_digraph.add_edge(actual_src, actual_tgt, **edge_attrs)

    collapsed = GraphIR(
        digraph=new_digraph,
        direction=gir.direction,
        subgraph_members=[],
        subgraph_descriptions={},
    )

    return collapsed, compounds


def compute_compound_dimensions(compounds: list[CompoundInfo], padding: int) -> dict[str, tuple[int, int]]:
    overrides: dict[str, tuple[int, int]] = {}
    for ci in compounds:
        total_member_w = sum(ci.member_widths)
        gaps = (len(ci.member_ids) - 1) * _SG_INNER_GAP if len(ci.member_ids) > 1 else 0
        content_w = total_member_w + gaps
        title_w = len(ci.sg_name) + 4
        desc_w = len(ci.description) + 4 if ci.description else 0
        inner_w = max(content_w, title_w, desc_w)
        width = 2 + 2 * _SG_PAD_X + inner_w
        desc_rows = 1 if ci.description else 0
        height = 3 + desc_rows if not ci.member_ids else 2 + 1 + ci.max_member_height + desc_rows
        _ = padding
        overrides[ci.compound_id] = (width, height)
    return overrides


def expand_compound_nodes(layout_nodes: list[LayoutNode], compounds: list[CompoundInfo]) -> list[LayoutNode]:
    compound_map: dict[str, CompoundInfo] = {c.compound_id: c for c in compounds}
    result: list[LayoutNode] = []

    for ln in layout_nodes:
        result.append(ln)
        ci = compound_map.get(ln.id)
        if ci is not None:
            member_x = ln.x + 1 + _SG_PAD_X
            member_y = ln.y + 2
            for i, mid in enumerate(ci.member_ids):
                result.append(
                    LayoutNode(
                        id=mid,
                        layer=ln.layer,
                        order=ln.order,
                        x=member_x,
                        y=member_y,
                        width=ci.member_widths[i],
                        height=ci.member_heights[i],
                    )
                )
                member_x += ci.member_widths[i] + _SG_INNER_GAP

    return result


# ─── SugiyamaLayout Engine ───────────────────────────────────────────────────


class SugiyamaLayout:
    """Sugiyama layered layout engine."""

    def layout(self, gir: GraphIR, padding: int) -> tuple[list[LayoutNode], list[RoutedEdge]]:
        has_subgraphs = bool(gir.subgraph_members)

        if not has_subgraphs:
            la = LayerAssignment.assign(gir)
            dag, reversed_edges = remove_cycles(gir.digraph)
            aug = insert_dummy_nodes(dag, la)
            ordering = minimise_crossings(aug)
            layout_nodes = assign_coordinates_padded(ordering, aug, padding, {}, gir.direction)
            routed_edges = route_edges(gir, layout_nodes, aug, reversed_edges)
            return layout_nodes, routed_edges

        collapsed, compounds = collapse_subgraphs(gir, padding)
        dim_overrides = compute_compound_dimensions(compounds, padding)

        la = LayerAssignment.assign(collapsed)
        dag, reversed_edges = remove_cycles(collapsed.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates_padded(ordering, aug, padding, dim_overrides, gir.direction)

        expanded = expand_compound_nodes(layout_nodes, compounds)
        routed_edges = route_edges(collapsed, expanded, aug, reversed_edges)

        return expanded, routed_edges
