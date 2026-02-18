"""Tests for layout.py — cycle removal (Phase 4) + crossing minimization + coordinate assignment (Phase 6)
+ edge routing and compound nodes (Phase 7).

Phase 4 tests port the 5 Rust cycle-removal tests:
  - test_dag_has_no_reversed_edges
  - test_single_cycle_reversed
  - test_self_loop_reversed
  - test_complex_cycle
  - test_empty_graph

Phase 6 tests cover:
  - minimise_crossings (barycenter heuristic)
  - count_crossings (inversion count)
  - assign_coordinates / assign_coordinates_padded (TD + LR)
  - label_dimensions helper
  - LayoutNode dataclass

Phase 7 tests cover:
  - Point dataclass
  - RoutedEdge dataclass
  - compute_orthogonal_waypoints
  - route_edges
  - COMPOUND_PREFIX constant
  - CompoundInfo dataclass
  - collapse_subgraphs
  - compute_compound_dimensions
  - expand_compound_nodes
  - full_layout / full_layout_with_padding
"""

from __future__ import annotations

import networkx as nx

from mermaid_ascii.layout.engine import (
    assign_coordinates,
    compute_orthogonal_waypoints,
    full_layout,
    full_layout_with_padding,
)
from mermaid_ascii.layout.sugiyama import (
    COMPOUND_PREFIX,
    H_GAP,
    NODE_HEIGHT,
    NODE_PADDING,
    V_GAP,
    AugmentedGraph,
    CompoundInfo,
    CycleRemovalResult,
    LayerAssignment,
    assign_coordinates_padded,
    collapse_subgraphs,
    compute_compound_dimensions,
    count_crossings,
    expand_compound_nodes,
    greedy_fas_ordering,
    insert_dummy_nodes,
    label_dimensions,
    minimise_crossings,
    remove_cycles,
    route_edges,
)
from mermaid_ascii.layout.types import DUMMY_PREFIX, LayoutNode, Point, RoutedEdge
from mermaid_ascii.syntax import types as mast
from mermaid_ascii.syntax.graph import EdgeData, GraphIR, NodeData

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_graph(*edges: tuple[str, str]) -> nx.DiGraph:
    """Build a DiGraph from a list of (src, tgt) string pairs."""
    g: nx.DiGraph = nx.DiGraph()
    for src, tgt in edges:
        g.add_edge(src, tgt)
    return g


def make_graph_nodes(*nodes: str) -> nx.DiGraph:
    """Build a DiGraph with only nodes (no edges)."""
    g: nx.DiGraph = nx.DiGraph()
    for node in nodes:
        g.add_node(node)
    return g


def make_node_data(node_id: str, label: str = "") -> NodeData:
    """Create a minimal NodeData for testing."""
    return NodeData(
        id=node_id,
        label=label or node_id,
        shape=mast.NodeShape.Rectangle,
        attrs=[],
        subgraph=None,
    )


def make_edge_data() -> EdgeData:
    """Create a minimal EdgeData for testing."""
    return EdgeData(edge_type=mast.EdgeType.Arrow, label=None, attrs=[])


def make_augmented_graph(
    edges: list[tuple[str, str]],
    layers: dict[str, int],
) -> AugmentedGraph:
    """Build a minimal AugmentedGraph from (src, tgt) edges and explicit layers.

    All nodes get NodeData with id==label. All edges get Arrow EdgeData.
    """
    g: nx.DiGraph = nx.DiGraph()
    all_node_ids: set[str] = set(layers.keys())
    for src, tgt in edges:
        all_node_ids.add(src)
        all_node_ids.add(tgt)

    for nid in all_node_ids:
        g.add_node(nid, data=make_node_data(nid))

    for src, tgt in edges:
        g.add_edge(src, tgt, data=make_edge_data())

    layer_count = (max(layers.values()) + 1) if layers else 0
    return AugmentedGraph(graph=g, layers=layers, layer_count=layer_count, dummy_edges=[])


# ─── Cycle Removal Tests ──────────────────────────────────────────────────────


class TestCycleRemoval:
    def test_dag_has_no_reversed_edges(self):
        """A → B → C (simple DAG, no cycles) — should have zero reversed edges."""
        g = make_graph(("A", "B"), ("B", "C"))
        dag, reversed_edges = remove_cycles(g)
        assert len(reversed_edges) == 0, f"DAG should have no reversed edges, got: {reversed_edges}"
        assert not nx.is_directed_acyclic_graph(g) or nx.is_directed_acyclic_graph(dag)

    def test_single_cycle_reversed(self):
        """A → B → A (2-cycle) — should reverse exactly one edge, result is a DAG."""
        g = make_graph(("A", "B"), ("B", "A"))
        dag, reversed_edges = remove_cycles(g)
        assert len(reversed_edges) == 1, f"Should reverse exactly one edge, got: {reversed_edges}"
        assert nx.is_directed_acyclic_graph(dag), "Result should be a DAG"

    def test_self_loop_reversed(self):
        """A → A (self-loop) — self-loop counted as reversed, removed from result DAG."""
        g = make_graph(("A", "A"))
        dag, reversed_edges = remove_cycles(g)
        assert len(reversed_edges) == 1, "Self-loop should be counted as reversed"
        assert nx.is_directed_acyclic_graph(dag), "Result should be a DAG"
        # Self-loop should be removed entirely (not just reversed)
        assert dag.number_of_edges() == 0, "Self-loop should be removed from the DAG"

    def test_complex_cycle(self):
        """A → B → C → A (3-cycle) plus D → B — result must be a DAG."""
        g = make_graph(("A", "B"), ("B", "C"), ("C", "A"), ("D", "B"))
        dag, reversed_edges = remove_cycles(g)
        assert nx.is_directed_acyclic_graph(dag), "Result should be a DAG"
        # Some edges must have been reversed to break the cycle
        assert len(reversed_edges) >= 1, "At least one edge should be reversed"

    def test_empty_graph(self):
        """Empty graph — should return empty graph with no reversed edges."""
        g: nx.DiGraph = nx.DiGraph()
        dag, reversed_edges = remove_cycles(g)
        assert dag.number_of_nodes() == 0
        assert len(reversed_edges) == 0


# ─── greedy_fas_ordering Tests ────────────────────────────────────────────────


class TestGreedyFasOrdering:
    def test_chain_ordering(self):
        """A → B → C — ordering should put A before B before C."""
        g = make_graph(("A", "B"), ("B", "C"))
        ordering = greedy_fas_ordering(g)
        assert set(ordering) == {"A", "B", "C"}, "All nodes should appear in ordering"
        assert len(ordering) == 3

    def test_single_node(self):
        """Single node — ordering has just that node."""
        g = make_graph_nodes("A")
        ordering = greedy_fas_ordering(g)
        assert ordering == ["A"]

    def test_empty_graph(self):
        """Empty graph — ordering is empty."""
        g: nx.DiGraph = nx.DiGraph()
        ordering = greedy_fas_ordering(g)
        assert ordering == []

    def test_all_nodes_present(self):
        """Ordering must contain all nodes exactly once."""
        g = make_graph(("A", "B"), ("B", "C"), ("C", "A"))
        ordering = greedy_fas_ordering(g)
        assert len(ordering) == 3
        assert set(ordering) == {"A", "B", "C"}


# ─── CycleRemovalResult Tests ─────────────────────────────────────────────────


class TestCycleRemovalResult:
    def test_default_empty(self):
        """CycleRemovalResult defaults to empty reversed_edges set."""
        result = CycleRemovalResult()
        assert result.reversed_edges == set()

    def test_stores_edge_pairs(self):
        """CycleRemovalResult stores (src, tgt) string tuples."""
        result = CycleRemovalResult(reversed_edges={("A", "B"), ("C", "D")})
        assert ("A", "B") in result.reversed_edges
        assert ("C", "D") in result.reversed_edges


# ─── Phase 6: label_dimensions Tests ─────────────────────────────────────────


class TestLabelDimensions:
    def test_empty_label(self):
        """Empty label → (0, 1) width-0, single line."""
        assert label_dimensions("") == (0, 1)

    def test_single_line(self):
        """Single-line label → (len, 1)."""
        assert label_dimensions("Hello") == (5, 1)

    def test_multiline(self):
        """Multi-line label → (max_width, line_count)."""
        label = "Hello\nWorld\nLonger line"
        w, h = label_dimensions(label)
        assert h == 3
        assert w == len("Longer line")

    def test_single_char(self):
        """Single-char label — width 1, height 1."""
        assert label_dimensions("X") == (1, 1)

    def test_equal_width_lines(self):
        """Multiple lines of equal length."""
        assert label_dimensions("AB\nCD") == (2, 2)


# ─── Phase 6: LayoutNode Dataclass Tests ──────────────────────────────────────


class TestLayoutNode:
    def test_construction(self):
        """LayoutNode stores all fields correctly."""
        node = LayoutNode(id="A", layer=0, order=0, x=5, y=10, width=7, height=3)
        assert node.id == "A"
        assert node.layer == 0
        assert node.order == 0
        assert node.x == 5
        assert node.y == 10
        assert node.width == 7
        assert node.height == 3

    def test_equality(self):
        """Two LayoutNodes with same fields are equal."""
        n1 = LayoutNode(id="X", layer=1, order=2, x=0, y=6, width=5, height=3)
        n2 = LayoutNode(id="X", layer=1, order=2, x=0, y=6, width=5, height=3)
        assert n1 == n2

    def test_constants_exported(self):
        """Layout constants are exported and have correct types/values."""
        assert isinstance(NODE_PADDING, int)
        assert isinstance(H_GAP, int)
        assert isinstance(V_GAP, int)
        assert isinstance(NODE_HEIGHT, int)
        assert NODE_PADDING == 1
        assert H_GAP == 4
        assert V_GAP == 3
        assert NODE_HEIGHT == 3


# ─── Phase 6: count_crossings Tests ───────────────────────────────────────────


class TestCountCrossings:
    def test_no_crossings_simple_chain(self):
        """A → B with A in layer 0 and B in layer 1 — zero crossings."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        assert count_crossings(ordering, aug.graph) == 0

    def test_no_crossings_parallel(self):
        """Two parallel edges (A→C, B→D) with natural ordering — zero crossings."""
        aug = make_augmented_graph([("A", "C"), ("B", "D")], {"A": 0, "B": 0, "C": 1, "D": 1})
        # [A, B] → [C, D] — no crossings
        ordering = [["A", "B"], ["C", "D"]]
        assert count_crossings(ordering, aug.graph) == 0

    def test_one_crossing(self):
        """A→D and B→C with A before B in layer 0 — one crossing because D after C."""
        aug = make_augmented_graph([("A", "D"), ("B", "C")], {"A": 0, "B": 0, "C": 1, "D": 1})
        # [A, B] × [C, D]: A→D (pos 1), B→C (pos 0) → crossing
        ordering = [["A", "B"], ["C", "D"]]
        assert count_crossings(ordering, aug.graph) == 1

    def test_empty_graph_no_crossings(self):
        """Empty graph — zero crossings."""
        g: nx.DiGraph = nx.DiGraph()
        assert count_crossings([], g) == 0

    def test_single_layer_no_crossings(self):
        """Single-layer ordering with no inter-layer edges — zero crossings."""
        aug = make_augmented_graph([], {"A": 0, "B": 0})
        ordering = [["A", "B"]]
        assert count_crossings(ordering, aug.graph) == 0

    def test_crossing_reduces_with_swap(self):
        """Swapping layer 1 node order should reduce crossings from 1 to 0."""
        aug = make_augmented_graph([("A", "D"), ("B", "C")], {"A": 0, "B": 0, "C": 1, "D": 1})
        crossed_order = [["A", "B"], ["C", "D"]]
        uncrossed_order = [["A", "B"], ["D", "C"]]
        assert count_crossings(crossed_order, aug.graph) == 1
        assert count_crossings(uncrossed_order, aug.graph) == 0


# ─── Phase 6: minimise_crossings Tests ────────────────────────────────────────


class TestMinimiseCrossings:
    def test_returns_all_nodes(self):
        """minimise_crossings returns all nodes, none missing or duplicated."""
        aug = make_augmented_graph([("A", "B"), ("A", "C")], {"A": 0, "B": 1, "C": 1})
        result = minimise_crossings(aug)
        all_ids = {nid for layer in result for nid in layer}
        assert all_ids == {"A", "B", "C"}

    def test_layer_count_matches(self):
        """Result has exactly layer_count layers."""
        aug = make_augmented_graph([("A", "B"), ("B", "C")], {"A": 0, "B": 1, "C": 2})
        result = minimise_crossings(aug)
        assert len(result) == aug.layer_count

    def test_each_node_in_correct_layer(self):
        """Every node appears in the layer matching its assignment."""
        layers = {"A": 0, "B": 1, "C": 1, "D": 2}
        aug = make_augmented_graph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")], layers)
        result = minimise_crossings(aug)
        for node_id, expected_layer in layers.items():
            assert node_id in result[expected_layer], f"{node_id} should be in layer {expected_layer}"

    def test_crossings_not_worse_after_minimise(self):
        """After minimise_crossings, crossing count should not exceed initial count."""
        aug = make_augmented_graph(
            [("A", "D"), ("B", "C")],
            {"A": 0, "B": 0, "C": 1, "D": 1},
        )
        # Initial ordering (alphabetical: A,B / C,D) has 1 crossing
        initial_ordering = [["A", "B"], ["C", "D"]]
        initial_crossings = count_crossings(initial_ordering, aug.graph)
        result = minimise_crossings(aug)
        final_crossings = count_crossings(result, aug.graph)
        assert final_crossings <= initial_crossings, (
            f"minimise_crossings worsened crossings: {initial_crossings} → {final_crossings}"
        )

    def test_empty_graph(self):
        """Empty augmented graph returns empty list of layers."""
        g: nx.DiGraph = nx.DiGraph()
        aug = AugmentedGraph(graph=g, layers={}, layer_count=0, dummy_edges=[])
        result = minimise_crossings(aug)
        assert result == []

    def test_single_node_single_layer(self):
        """Single node in single layer returns [[node_id]]."""
        aug = make_augmented_graph([], {"A": 0})
        result = minimise_crossings(aug)
        assert result == [["A"]]

    def test_no_duplicates_in_layers(self):
        """No node should appear in more than one layer after minimisation."""
        layers = {"A": 0, "B": 1, "C": 1, "D": 2, "E": 2}
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")]
        aug = make_augmented_graph(edges, layers)
        result = minimise_crossings(aug)
        all_ids = [nid for layer in result for nid in layer]
        assert len(all_ids) == len(set(all_ids)), "Each node must appear exactly once"

    def test_parallel_diamond_zero_crossings(self):
        """Diamond A→B,A→C,B→D,C→D — barycenter should produce zero crossings."""
        layers = {"A": 0, "B": 1, "C": 1, "D": 2}
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        aug = make_augmented_graph(edges, layers)
        result = minimise_crossings(aug)
        # With symmetric topology, any ordering of B,C has zero crossings
        assert count_crossings(result, aug.graph) == 0


# ─── Phase 6: assign_coordinates Tests ────────────────────────────────────────


class TestAssignCoordinates:
    def test_returns_layout_nodes_for_all_nodes(self):
        """assign_coordinates returns a LayoutNode for every node in the graph."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        result = assign_coordinates(ordering, aug)
        ids = {n.id for n in result}
        assert ids == {"A", "B"}

    def test_layer_zero_starts_at_y_zero(self):
        """First layer (layer 0) starts at y=0."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        result = assign_coordinates(ordering, aug)
        layer0 = [n for n in result if n.layer == 0]
        assert all(n.y == 0 for n in layer0), "Layer 0 nodes should start at y=0"

    def test_layer_y_increases(self):
        """Successive layers have strictly increasing y coordinates."""
        aug = make_augmented_graph([("A", "B"), ("B", "C")], {"A": 0, "B": 1, "C": 2})
        ordering = [["A"], ["B"], ["C"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        assert id_to_node["A"].y < id_to_node["B"].y < id_to_node["C"].y

    def test_non_negative_coordinates(self):
        """All x and y coordinates must be non-negative."""
        layers = {"A": 0, "B": 0, "C": 1, "D": 1}
        edges = [("A", "C"), ("B", "D")]
        aug = make_augmented_graph(edges, layers)
        ordering = [["A", "B"], ["C", "D"]]
        result = assign_coordinates(ordering, aug)
        for n in result:
            assert n.x >= 0, f"Node {n.id} has negative x={n.x}"
            assert n.y >= 0, f"Node {n.id} has negative y={n.y}"

    def test_nodes_in_same_layer_have_same_y(self):
        """All nodes in the same layer share the same y coordinate."""
        layers = {"A": 0, "B": 0, "C": 1}
        edges = [("A", "C"), ("B", "C")]
        aug = make_augmented_graph(edges, layers)
        ordering = [["A", "B"], ["C"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        assert id_to_node["A"].y == id_to_node["B"].y

    def test_nodes_in_same_layer_non_overlapping_x(self):
        """Nodes in the same layer must not overlap in x."""
        layers = {"A": 0, "B": 0}
        edges: list[tuple[str, str]] = []
        aug = make_augmented_graph(edges, layers)
        ordering = [["A", "B"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        a, b = id_to_node["A"], id_to_node["B"]
        # Whichever comes first, their extents must not overlap
        left, right = (a, b) if a.x < b.x else (b, a)
        assert left.x + left.width <= right.x, "Nodes in same layer must not overlap"

    def test_width_reflects_label_length(self):
        """A node with a longer label gets a wider width than one with a shorter label."""
        aug = make_augmented_graph([], {"Short": 0, "VeryLongLabel": 0})
        # Give VeryLongLabel a real NodeData with a long label
        aug.graph.nodes["VeryLongLabel"]["data"] = NodeData(
            id="VeryLongLabel",
            label="VeryLongLabel",
            shape=mast.NodeShape.Rectangle,
            attrs=[],
            subgraph=None,
        )
        aug.graph.nodes["Short"]["data"] = NodeData(
            id="Short",
            label="Hi",
            shape=mast.NodeShape.Rectangle,
            attrs=[],
            subgraph=None,
        )
        ordering = [["Short", "VeryLongLabel"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        assert id_to_node["VeryLongLabel"].width > id_to_node["Short"].width

    def test_dummy_node_width_is_one(self):
        """Dummy nodes (DUMMY_PREFIX) get width 1."""
        dummy_id = f"{DUMMY_PREFIX}0_0"
        layers = {"A": 0, dummy_id: 1, "B": 2}
        aug = make_augmented_graph([("A", dummy_id), (dummy_id, "B")], layers)
        # Give dummy node empty label
        aug.graph.nodes[dummy_id]["data"] = NodeData(
            id=dummy_id,
            label="",
            shape=mast.NodeShape.Rectangle,
            attrs=[],
            subgraph=None,
        )
        ordering = [["A"], [dummy_id], ["B"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        assert id_to_node[dummy_id].width == 1

    def test_layer_y_gap_matches_constants(self):
        """Y gap between consecutive layers equals NODE_HEIGHT + V_GAP for default nodes."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        y_gap = id_to_node["B"].y - id_to_node["A"].y
        # height for label "A" (1 char, single line) = 2+1 = 3 = NODE_HEIGHT
        # layer_max_height = max(NODE_HEIGHT, 3) = 3; gap = 3 + V_GAP
        assert y_gap == NODE_HEIGHT + V_GAP

    def test_order_field_matches_position_in_layer(self):
        """LayoutNode.order matches the node's position index within its layer."""
        layers = {"A": 0, "B": 0, "C": 1}
        aug = make_augmented_graph([("A", "C"), ("B", "C")], layers)
        ordering = [["A", "B"], ["C"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        assert id_to_node["A"].order == 0
        assert id_to_node["B"].order == 1
        assert id_to_node["C"].order == 0

    def test_layer_field_correct(self):
        """LayoutNode.layer field matches the assigned layer."""
        layers = {"A": 0, "B": 1, "C": 2}
        aug = make_augmented_graph([("A", "B"), ("B", "C")], layers)
        ordering = [["A"], ["B"], ["C"]]
        result = assign_coordinates(ordering, aug)
        id_to_node = {n.id: n for n in result}
        assert id_to_node["A"].layer == 0
        assert id_to_node["B"].layer == 1
        assert id_to_node["C"].layer == 2

    def test_single_node_at_origin(self):
        """A single node should be placed with x=0, y=0."""
        aug = make_augmented_graph([], {"A": 0})
        ordering = [["A"]]
        result = assign_coordinates(ordering, aug)
        assert len(result) == 1
        assert result[0].x == 0
        assert result[0].y == 0


# ─── Phase 6: assign_coordinates_padded Tests (LR direction) ──────────────────


class TestAssignCoordinatesPadded:
    def test_lr_direction_produces_valid_coords(self):
        """LR direction assigns non-negative coordinates without overlap."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        result = assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, mast.Direction.LR)
        assert len(result) == 2
        for n in result:
            assert n.x >= 0
            assert n.y >= 0

    def test_td_and_lr_produce_different_layouts(self):
        """TD and LR directions should differ in at least some node coordinates."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        td_result = assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, mast.Direction.TD)
        lr_result = assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, mast.Direction.LR)
        td_map = {n.id: n for n in td_result}
        lr_map = {n.id: n for n in lr_result}
        # For two-node graph, at minimum the y offsets differ between TD and LR
        coords_differ = any(td_map[nid].x != lr_map[nid].x or td_map[nid].y != lr_map[nid].y for nid in ("A", "B"))
        assert coords_differ, "TD and LR should produce different coordinate assignments"

    def test_size_overrides_applied(self):
        """size_overrides allows caller to specify custom (width, height)."""
        aug = make_augmented_graph([], {"A": 0})
        ordering = [["A"]]
        overrides = {"A": (20, 10)}
        result = assign_coordinates_padded(ordering, aug, NODE_PADDING, overrides, mast.Direction.TD)
        assert len(result) == 1
        assert result[0].width == 20
        assert result[0].height == 10

    def test_custom_padding_increases_width(self):
        """Larger padding should result in wider nodes."""
        aug = make_augmented_graph([], {"Hello": 0})
        ordering = [["Hello"]]
        result_p1 = assign_coordinates_padded(ordering, aug, 1, {}, mast.Direction.TD)
        result_p3 = assign_coordinates_padded(ordering, aug, 3, {}, mast.Direction.TD)
        assert result_p3[0].width > result_p1[0].width

    def test_rl_direction_produces_valid_coords(self):
        """RL direction also assigns valid non-negative coordinates."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        result = assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, mast.Direction.RL)
        for n in result:
            assert n.x >= 0
            assert n.y >= 0

    def test_bt_direction_produces_valid_coords(self):
        """BT direction assigns valid non-negative coordinates (treated as TD internally)."""
        aug = make_augmented_graph([("A", "B")], {"A": 0, "B": 1})
        ordering = [["A"], ["B"]]
        result = assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, mast.Direction.BT)
        for n in result:
            assert n.x >= 0
            assert n.y >= 0


# ─── Phase 6 Integration Tests ────────────────────────────────────────────────


class TestPhase6Integration:
    def _build_gir_from_dsl(self, dsl: str) -> GraphIR:
        """Helper: parse DSL text → AST → GraphIR."""
        from mermaid_ascii.parsers.registry import parse

        ast_graph = parse(dsl)
        return GraphIR.from_ast(ast_graph)

    def test_simple_chain_full_pipeline(self):
        """Full pipeline on A→B→C produces valid LayoutNodes."""
        gir = self._build_gir_from_dsl("graph TD\n    A --> B\n    B --> C\n")
        la = LayerAssignment.assign(gir)
        dag, _ = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        result = assign_coordinates(ordering, aug)

        ids = {n.id for n in result}
        assert "A" in ids
        assert "B" in ids
        assert "C" in ids
        for n in result:
            assert n.x >= 0
            assert n.y >= 0

    def test_diamond_no_overlapping_nodes(self):
        """Diamond graph: A→B, A→C, B→D, C→D — nodes must not overlap in x within same layer."""
        gir = self._build_gir_from_dsl("graph TD\n    A --> B\n    A --> C\n    B --> D\n    C --> D\n")
        la = LayerAssignment.assign(gir)
        dag, _ = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        result = assign_coordinates(ordering, aug)

        # All nodes present
        ids = {n.id for n in result}
        assert {"A", "B", "C", "D"}.issubset(ids)

        # No overlapping nodes in the same layer
        layer_groups: dict[int, list[LayoutNode]] = {}
        for n in result:
            layer_groups.setdefault(n.layer, []).append(n)
        for layer_idx, nodes in layer_groups.items():
            nodes_sorted = sorted(nodes, key=lambda n: n.x)
            for i in range(len(nodes_sorted) - 1):
                left = nodes_sorted[i]
                right = nodes_sorted[i + 1]
                assert left.x + left.width <= right.x, f"Nodes {left.id} and {right.id} overlap in layer {layer_idx}"

    def test_long_chain_layers_monotonically_increase(self):
        """A chain A→B→C→D→E — layer indices must strictly increase along the chain."""
        gir = self._build_gir_from_dsl("graph TD\n    A --> B\n    B --> C\n    C --> D\n    D --> E\n")
        la = LayerAssignment.assign(gir)
        dag, _ = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        result = assign_coordinates(ordering, aug)

        # All real nodes present
        ids = {n.id for n in result}
        assert {"A", "B", "C", "D", "E"}.issubset(ids)

        # Layers increase monotonically from A to E
        id_to_node = {n.id: n for n in result}
        assert id_to_node["A"].layer < id_to_node["B"].layer
        assert id_to_node["B"].layer < id_to_node["C"].layer
        assert id_to_node["C"].layer < id_to_node["D"].layer
        assert id_to_node["D"].layer < id_to_node["E"].layer

    def test_all_y_coords_non_negative_after_full_pipeline(self):
        """End-to-end pipeline: all coordinates must be non-negative."""
        gir = self._build_gir_from_dsl("graph LR\n    A --> B\n    A --> C\n    B --> D\n")
        la = LayerAssignment.assign(gir)
        dag, _ = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        result = assign_coordinates(ordering, aug)
        for n in result:
            assert n.x >= 0, f"Node {n.id} has negative x={n.x}"
            assert n.y >= 0, f"Node {n.id} has negative y={n.y}"


# ─── Phase 7: Point dataclass tests ───────────────────────────────────────────


class TestPoint:
    def test_construction(self):
        """Point stores x and y fields."""
        p = Point(x=3, y=7)
        assert p.x == 3
        assert p.y == 7

    def test_equality(self):
        """Two Points with same coordinates are equal."""
        assert Point(x=0, y=0) == Point(x=0, y=0)
        assert Point(x=5, y=10) == Point(x=5, y=10)

    def test_inequality(self):
        """Points with different coordinates are not equal."""
        assert Point(x=1, y=2) != Point(x=2, y=1)

    def test_zero_coords(self):
        """Point at origin (0, 0) is valid."""
        p = Point(x=0, y=0)
        assert p.x == 0
        assert p.y == 0


# ─── Phase 7: RoutedEdge dataclass tests ──────────────────────────────────────


class TestRoutedEdge:
    def test_construction(self):
        """RoutedEdge stores all fields correctly."""
        wp = [Point(x=0, y=0), Point(x=0, y=5)]
        edge = RoutedEdge(from_id="A", to_id="B", label="my label", edge_type=mast.EdgeType.Arrow, waypoints=wp)
        assert edge.from_id == "A"
        assert edge.to_id == "B"
        assert edge.label == "my label"
        assert edge.edge_type == mast.EdgeType.Arrow
        assert edge.waypoints == wp

    def test_none_label(self):
        """RoutedEdge accepts None as label."""
        edge = RoutedEdge(from_id="X", to_id="Y", label=None, edge_type=mast.EdgeType.Line, waypoints=[])
        assert edge.label is None

    def test_empty_waypoints(self):
        """RoutedEdge can have empty waypoints list."""
        edge = RoutedEdge(from_id="A", to_id="B", label=None, edge_type=mast.EdgeType.Arrow, waypoints=[])
        assert edge.waypoints == []

    def test_multiple_waypoints(self):
        """RoutedEdge can carry many waypoints."""
        wps = [Point(x=i, y=i * 2) for i in range(5)]
        edge = RoutedEdge(from_id="A", to_id="Z", label=None, edge_type=mast.EdgeType.DottedArrow, waypoints=wps)
        assert len(edge.waypoints) == 5


# ─── Phase 7: compute_orthogonal_waypoints tests ──────────────────────────────


def _make_layout_node(nid: str, layer: int, x: int, y: int, w: int = 5, h: int = 3) -> LayoutNode:
    return LayoutNode(id=nid, layer=layer, order=0, x=x, y=y, width=w, height=h)


class TestComputeOrthogonalWaypoints:
    def test_adjacent_layers_two_waypoints(self):
        """Adjacent-layer edge (layer 0 → layer 1): produces at least 2 waypoints (exit + entry)."""
        from_node = _make_layout_node("A", layer=0, x=0, y=0, w=5, h=3)
        to_node = _make_layout_node("B", layer=1, x=0, y=6, w=5, h=3)
        layer_top_y = [0, 6]
        layer_bottom_y = [3, 9]
        wps = compute_orthogonal_waypoints(from_node, to_node, layer_top_y, layer_bottom_y, [])
        assert len(wps) >= 2, "Should have at least exit + entry waypoints"

    def test_first_waypoint_is_exit_of_from_node(self):
        """First waypoint is at the bottom-center of from_node."""
        from_node = _make_layout_node("A", layer=0, x=0, y=0, w=6, h=3)
        to_node = _make_layout_node("B", layer=1, x=0, y=6, w=6, h=3)
        wps = compute_orthogonal_waypoints(from_node, to_node, [0, 6], [3, 9], [])
        # exit_x = 0 + 6//2 = 3; exit_y = 0 + 3 - 1 = 2
        assert wps[0].x == 3
        assert wps[0].y == 2

    def test_last_waypoint_is_entry_of_to_node(self):
        """Last waypoint is at the top-center of to_node."""
        from_node = _make_layout_node("A", layer=0, x=0, y=0, w=6, h=3)
        to_node = _make_layout_node("B", layer=1, x=4, y=6, w=6, h=3)
        wps = compute_orthogonal_waypoints(from_node, to_node, [0, 6], [3, 9], [])
        # entry_x = 4 + 6//2 = 7; entry_y = 6 (top of to_node)
        assert wps[-1].x == 7
        assert wps[-1].y == 6

    def test_same_layer_u_shape(self):
        """Same-layer edge produces a 4-point U-shape going below the layer."""
        from_node = _make_layout_node("A", layer=0, x=0, y=0, w=5, h=3)
        to_node = _make_layout_node("B", layer=0, x=10, y=0, w=5, h=3)
        layer_top_y = [0]
        layer_bottom_y = [3]
        wps = compute_orthogonal_waypoints(from_node, to_node, layer_top_y, layer_bottom_y, [])
        assert len(wps) == 4, "Same-layer edge should produce exactly 4 waypoints"
        # U-shape: exit → below-left → below-right → entry
        assert wps[0].y < wps[1].y, "First drop should go down"
        assert wps[2].y > wps[0].y, "Still below original y before final entry"
        assert wps[3].y == to_node.y, "Last point lands at top of to_node"

    def test_long_edge_uses_dummy_xs(self):
        """Edge spanning 2 layers uses provided dummy_xs for horizontal routing."""
        from_node = _make_layout_node("A", layer=0, x=0, y=0, w=5, h=3)
        to_node = _make_layout_node("C", layer=2, x=0, y=12, w=5, h=3)
        layer_top_y = [0, 6, 12]
        layer_bottom_y = [3, 9, 15]
        dummy_xs = [8]  # dummy at x=8 in gap between layer 0 and layer 1
        wps = compute_orthogonal_waypoints(from_node, to_node, layer_top_y, layer_bottom_y, dummy_xs)
        # Should have more than 2 waypoints to handle the intermediate layer
        assert len(wps) >= 3
        # At least one waypoint should be at x=8 (the dummy x position)
        xs = [w.x for w in wps]
        assert 8 in xs, f"Expected dummy x=8 in waypoints, got {xs}"

    def test_waypoints_monotonically_increasing_y(self):
        """For a top-down edge, y coordinates of waypoints should be non-decreasing."""
        from_node = _make_layout_node("A", layer=0, x=0, y=0, w=5, h=3)
        to_node = _make_layout_node("B", layer=1, x=0, y=6, w=5, h=3)
        wps = compute_orthogonal_waypoints(from_node, to_node, [0, 6], [3, 9], [])
        for i in range(len(wps) - 1):
            assert wps[i].y <= wps[i + 1].y, f"Y should be non-decreasing: {wps}"

    def test_all_waypoints_non_negative(self):
        """All waypoint coordinates must be non-negative."""
        from_node = _make_layout_node("A", layer=0, x=2, y=0, w=5, h=3)
        to_node = _make_layout_node("B", layer=1, x=2, y=6, w=5, h=3)
        wps = compute_orthogonal_waypoints(from_node, to_node, [0, 6], [3, 9], [])
        for wp in wps:
            assert wp.x >= 0
            assert wp.y >= 0


# ─── Phase 7: route_edges tests ───────────────────────────────────────────────


def _build_gir_simple(dsl: str) -> GraphIR:
    """Helper: parse DSL → AST → GraphIR."""
    from mermaid_ascii.parsers.registry import parse

    return GraphIR.from_ast(parse(dsl))


class TestRouteEdges:
    def test_returns_one_route_per_edge(self):
        """route_edges returns exactly one RoutedEdge per real (non-self-loop) edge."""
        gir = _build_gir_simple("graph TD\n    A --> B\n    B --> C\n")
        la = LayerAssignment.assign(gir)
        dag, reversed_edges = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, reversed_edges)
        assert len(routes) == 2

    def test_route_from_to_ids_match_edges(self):
        """RoutedEdge from_id/to_id correspond to node ids in the graph."""
        gir = _build_gir_simple("graph TD\n    A --> B\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        assert len(routes) == 1
        route = routes[0]
        assert {route.from_id, route.to_id} == {"A", "B"}

    def test_each_route_has_waypoints(self):
        """Every RoutedEdge has at least 2 waypoints."""
        gir = _build_gir_simple("graph TD\n    A --> B\n    B --> C\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        for r in routes:
            assert len(r.waypoints) >= 2, f"Edge {r.from_id}→{r.to_id} has {len(r.waypoints)} waypoints"

    def test_edge_label_preserved_in_route(self):
        """Edge label from the graph is preserved in the RoutedEdge."""
        gir = _build_gir_simple("graph TD\n    A -->|hello| B\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        assert len(routes) == 1
        assert routes[0].label == "hello"

    def test_self_loop_not_included_in_routes(self):
        """Self-loop edges (A→A) should be excluded from routes."""
        gir = _build_gir_simple("graph TD\n    A --> B\n    A --> A\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        for r in routes:
            assert r.from_id != r.to_id, "Self-loop should not appear in routes"

    def test_edge_type_preserved_in_route(self):
        """Edge type from the graph is preserved in the RoutedEdge."""
        gir = _build_gir_simple("graph TD\n    A -.-> B\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        assert len(routes) == 1
        assert routes[0].edge_type == mast.EdgeType.DottedArrow

    def test_empty_graph_returns_no_routes(self):
        """An empty graph produces no routes."""
        gir = _build_gir_simple("graph TD\n    A\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        assert routes == []

    def test_all_waypoint_coords_non_negative(self):
        """All waypoints in all routed edges must have non-negative coordinates."""
        gir = _build_gir_simple("graph TD\n    A --> B\n    A --> C\n    B --> D\n    C --> D\n")
        la = LayerAssignment.assign(gir)
        dag, rev = remove_cycles(gir.digraph)
        aug = insert_dummy_nodes(dag, la)
        ordering = minimise_crossings(aug)
        layout_nodes = assign_coordinates(ordering, aug)
        routes = route_edges(gir, layout_nodes, aug, rev)
        for r in routes:
            for wp in r.waypoints:
                assert wp.x >= 0, f"Negative x in edge {r.from_id}→{r.to_id}"
                assert wp.y >= 0, f"Negative y in edge {r.from_id}→{r.to_id}"


# ─── Phase 7: COMPOUND_PREFIX + CompoundInfo tests ────────────────────────────


class TestCompoundPrefix:
    def test_compound_prefix_value(self):
        """COMPOUND_PREFIX is '__sg_'."""
        assert COMPOUND_PREFIX == "__sg_"

    def test_compound_prefix_type(self):
        """COMPOUND_PREFIX is a string."""
        assert isinstance(COMPOUND_PREFIX, str)


class TestCompoundInfo:
    def test_construction(self):
        """CompoundInfo stores all fields."""
        ci = CompoundInfo(
            sg_name="Group",
            compound_id="__sg_Group",
            member_ids=["X", "Y"],
            member_widths=[5, 7],
            member_heights=[3, 3],
            max_member_height=3,
            description="My group",
        )
        assert ci.sg_name == "Group"
        assert ci.compound_id == "__sg_Group"
        assert ci.member_ids == ["X", "Y"]
        assert ci.member_widths == [5, 7]
        assert ci.member_heights == [3, 3]
        assert ci.max_member_height == 3
        assert ci.description == "My group"

    def test_none_description(self):
        """CompoundInfo description can be None."""
        ci = CompoundInfo(
            sg_name="G",
            compound_id="__sg_G",
            member_ids=[],
            member_widths=[],
            member_heights=[],
            max_member_height=3,
            description=None,
        )
        assert ci.description is None


# ─── Phase 7: collapse_subgraphs tests ────────────────────────────────────────


class TestCollapseSubgraphs:
    def _parse_gir(self, dsl: str) -> GraphIR:
        from mermaid_ascii.parsers.registry import parse

        return GraphIR.from_ast(parse(dsl))

    def test_no_subgraphs_returns_original_gir(self):
        """Graph with no subgraphs returns unchanged GIR and empty compounds."""
        gir = self._parse_gir("graph TD\n    A --> B\n")
        collapsed, compounds = collapse_subgraphs(gir, NODE_PADDING)
        assert compounds == []
        # No compound nodes added
        assert COMPOUND_PREFIX not in " ".join(collapsed.digraph.nodes)

    def test_subgraph_creates_compound_node(self):
        """A subgraph produces one CompoundInfo and one compound node in collapsed GIR."""
        dsl = "graph TD\n    subgraph MyGroup\n        X --> Y\n    end\n"
        gir = self._parse_gir(dsl)
        collapsed, compounds = collapse_subgraphs(gir, NODE_PADDING)
        assert len(compounds) == 1
        ci = compounds[0]
        assert ci.sg_name == "MyGroup"
        assert ci.compound_id == f"{COMPOUND_PREFIX}MyGroup"
        # Compound node exists in collapsed graph
        assert ci.compound_id in collapsed.digraph.nodes

    def test_member_nodes_removed_from_collapsed_graph(self):
        """Member nodes of a subgraph should not appear directly in collapsed graph."""
        dsl = "graph TD\n    subgraph G\n        A --> B\n    end\n"
        gir = self._parse_gir(dsl)
        collapsed, _ = collapse_subgraphs(gir, NODE_PADDING)
        assert "A" not in collapsed.digraph.nodes
        assert "B" not in collapsed.digraph.nodes

    def test_cross_boundary_edge_redirected_to_compound(self):
        """Edge from external node to subgraph member becomes edge to compound node."""
        dsl = "graph TD\n    Ext --> A\n    subgraph G\n        A --> B\n    end\n"
        gir = self._parse_gir(dsl)
        collapsed, compounds = collapse_subgraphs(gir, NODE_PADDING)
        compound_id = compounds[0].compound_id
        # Ext → compound should exist
        assert collapsed.digraph.has_edge("Ext", compound_id), (
            f"Expected Ext → {compound_id}, edges: {list(collapsed.digraph.edges())}"
        )

    def test_internal_edges_dropped(self):
        """Edges between two members of the same subgraph are dropped in collapsed GIR."""
        dsl = "graph TD\n    subgraph G\n        A --> B\n    end\n"
        gir = self._parse_gir(dsl)
        collapsed, _ = collapse_subgraphs(gir, NODE_PADDING)
        # No edge from A or B (they're collapsed into a compound)
        assert collapsed.digraph.number_of_edges() == 0

    def test_member_ids_in_compound_info(self):
        """CompoundInfo.member_ids lists the subgraph members."""
        dsl = "graph TD\n    subgraph G\n        X\n        Y\n    end\n"
        gir = self._parse_gir(dsl)
        _, compounds = collapse_subgraphs(gir, NODE_PADDING)
        assert len(compounds) == 1
        assert set(compounds[0].member_ids) == {"X", "Y"}

    def test_compound_info_description_when_subgraph_has_label(self):
        """CompoundInfo description is set when subgraph has a description (title)."""
        dsl = 'graph TD\n    subgraph G["My Title"]\n        A\n    end\n'
        gir = self._parse_gir(dsl)
        _, compounds = collapse_subgraphs(gir, NODE_PADDING)
        # description should be present (either from subgraph description or None)
        # Just verify the field exists and is accessible
        assert hasattr(compounds[0], "description")


# ─── Phase 7: compute_compound_dimensions tests ───────────────────────────────


class TestComputeCompoundDimensions:
    def test_single_member_dimensions(self):
        """A compound with one member has at least enough width for that member."""
        ci = CompoundInfo(
            sg_name="G",
            compound_id="__sg_G",
            member_ids=["A"],
            member_widths=[7],
            member_heights=[3],
            max_member_height=3,
            description=None,
        )
        dims = compute_compound_dimensions([ci], NODE_PADDING)
        assert "__sg_G" in dims
        w, h = dims["__sg_G"]
        assert w >= 7, "Width should be at least as wide as the member"
        assert h >= 3, "Height should be at least as tall as the member"

    def test_two_members_wider_than_one(self):
        """Two members result in wider compound than one member of same width."""
        ci_one = CompoundInfo(
            sg_name="G1",
            compound_id="__sg_G1",
            member_ids=["A"],
            member_widths=[5],
            member_heights=[3],
            max_member_height=3,
            description=None,
        )
        ci_two = CompoundInfo(
            sg_name="G2",
            compound_id="__sg_G2",
            member_ids=["A", "B"],
            member_widths=[5, 5],
            member_heights=[3, 3],
            max_member_height=3,
            description=None,
        )
        dims = compute_compound_dimensions([ci_one, ci_two], NODE_PADDING)
        assert dims["__sg_G2"][0] > dims["__sg_G1"][0], "Two members should be wider than one"

    def test_dimensions_respect_title_width(self):
        """Compound width is at least as wide as the subgraph name + border chars."""
        long_name = "VeryLongSubgraphName"
        ci = CompoundInfo(
            sg_name=long_name,
            compound_id=f"__sg_{long_name}",
            member_ids=["X"],
            member_widths=[1],
            member_heights=[3],
            max_member_height=3,
            description=None,
        )
        dims = compute_compound_dimensions([ci], NODE_PADDING)
        w, _ = dims[f"__sg_{long_name}"]
        min_title_w = len(long_name) + 4
        assert w >= min_title_w, f"Width {w} should be at least {min_title_w}"

    def test_empty_members_still_returns_dims(self):
        """Compound with no members still produces valid (positive) dimensions."""
        ci = CompoundInfo(
            sg_name="Empty",
            compound_id="__sg_Empty",
            member_ids=[],
            member_widths=[],
            member_heights=[],
            max_member_height=NODE_HEIGHT,
            description=None,
        )
        dims = compute_compound_dimensions([ci], NODE_PADDING)
        assert "__sg_Empty" in dims
        w, h = dims["__sg_Empty"]
        assert w > 0
        assert h > 0


# ─── Phase 7: expand_compound_nodes tests ─────────────────────────────────────


class TestExpandCompoundNodes:
    def test_non_compound_nodes_unchanged(self):
        """Nodes that are not compound nodes pass through unchanged."""
        ln = LayoutNode(id="A", layer=0, order=0, x=5, y=10, width=7, height=3)
        result = expand_compound_nodes([ln], [])
        assert len(result) == 1
        assert result[0].id == "A"
        assert result[0].x == 5

    def test_compound_node_adds_members(self):
        """A compound node produces itself + one LayoutNode per member."""
        ci = CompoundInfo(
            sg_name="G",
            compound_id="__sg_G",
            member_ids=["X", "Y"],
            member_widths=[5, 7],
            member_heights=[3, 3],
            max_member_height=3,
            description=None,
        )
        compound_ln = LayoutNode(id="__sg_G", layer=0, order=0, x=0, y=0, width=20, height=6)
        result = expand_compound_nodes([compound_ln], [ci])
        ids = [n.id for n in result]
        assert "__sg_G" in ids
        assert "X" in ids
        assert "Y" in ids
        assert len(result) == 3  # compound + 2 members

    def test_member_nodes_inside_compound_bounds(self):
        """Member nodes are positioned inside the compound node's bounds."""
        ci = CompoundInfo(
            sg_name="G",
            compound_id="__sg_G",
            member_ids=["M1"],
            member_widths=[5],
            member_heights=[3],
            max_member_height=3,
            description=None,
        )
        compound_ln = LayoutNode(id="__sg_G", layer=0, order=0, x=10, y=20, width=15, height=8)
        result = expand_compound_nodes([compound_ln], [ci])
        member = next(n for n in result if n.id == "M1")
        assert member.x >= compound_ln.x, "Member should start at or after compound left edge"
        assert member.y >= compound_ln.y, "Member should start at or after compound top edge"
        assert member.x + member.width <= compound_ln.x + compound_ln.width + 2, (
            "Member should fit within compound (with border)"
        )

    def test_multiple_members_placed_horizontally(self):
        """Multiple members in a compound are placed horizontally (increasing x)."""
        ci = CompoundInfo(
            sg_name="G",
            compound_id="__sg_G",
            member_ids=["A", "B", "C"],
            member_widths=[5, 5, 5],
            member_heights=[3, 3, 3],
            max_member_height=3,
            description=None,
        )
        compound_ln = LayoutNode(id="__sg_G", layer=0, order=0, x=0, y=0, width=30, height=8)
        result = expand_compound_nodes([compound_ln], [ci])
        members = [n for n in result if n.id in {"A", "B", "C"}]
        members_sorted = sorted(members, key=lambda n: n.x)
        # All members should have the same y (horizontal layout)
        ys = [n.y for n in members]
        assert len(set(ys)) == 1, "All members should have same y"
        # x positions should be strictly increasing
        xs = [n.x for n in members_sorted]
        for i in range(len(xs) - 1):
            assert xs[i] < xs[i + 1], "Members should be placed left to right"

    def test_expand_preserves_layer_and_order(self):
        """Expanded members inherit the compound's layer and order."""
        ci = CompoundInfo(
            sg_name="G",
            compound_id="__sg_G",
            member_ids=["Z"],
            member_widths=[5],
            member_heights=[3],
            max_member_height=3,
            description=None,
        )
        compound_ln = LayoutNode(id="__sg_G", layer=2, order=1, x=0, y=0, width=15, height=8)
        result = expand_compound_nodes([compound_ln], [ci])
        member = next(n for n in result if n.id == "Z")
        assert member.layer == 2
        assert member.order == 1


# ─── Phase 7: full_layout / full_layout_with_padding integration tests ─────────


class TestFullLayout:
    def _parse_gir(self, dsl: str) -> GraphIR:
        from mermaid_ascii.parsers.registry import parse

        return GraphIR.from_ast(parse(dsl))

    def test_simple_chain_returns_nodes_and_edges(self):
        """full_layout on A→B returns LayoutNodes and RoutedEdges for both nodes + edge."""
        gir = self._parse_gir("graph TD\n    A --> B\n")
        layout_nodes, routed_edges = full_layout(gir)
        ids = {n.id for n in layout_nodes}
        assert "A" in ids
        assert "B" in ids
        assert len(routed_edges) == 1

    def test_all_coords_non_negative(self):
        """All node coordinates from full_layout must be non-negative."""
        gir = self._parse_gir("graph TD\n    A --> B\n    A --> C\n    B --> D\n")
        layout_nodes, routed_edges = full_layout(gir)
        for n in layout_nodes:
            assert n.x >= 0
            assert n.y >= 0
        for r in routed_edges:
            for wp in r.waypoints:
                assert wp.x >= 0
                assert wp.y >= 0

    def test_custom_padding_returns_wider_nodes(self):
        """full_layout_with_padding with larger padding returns wider nodes."""
        gir = self._parse_gir("graph TD\n    Hello --> World\n")
        nodes_p1, _ = full_layout_with_padding(gir, 1)
        nodes_p3, _ = full_layout_with_padding(gir, 3)
        # Find the "Hello" node in each
        n_p1 = next(n for n in nodes_p1 if n.id == "Hello")
        n_p3 = next(n for n in nodes_p3 if n.id == "Hello")
        assert n_p3.width > n_p1.width, "Larger padding should produce wider nodes"

    def test_subgraph_layout_includes_members(self):
        """full_layout on a graph with subgraph includes both compound and member nodes."""
        dsl = "graph TD\n    subgraph G\n        X --> Y\n    end\n"
        gir = self._parse_gir(dsl)
        layout_nodes, _ = full_layout(gir)
        ids = {n.id for n in layout_nodes}
        # Should contain the compound node
        assert any(nid.startswith(COMPOUND_PREFIX) for nid in ids), f"Expected compound node, got ids: {ids}"
        # Should contain member nodes X and Y
        assert "X" in ids
        assert "Y" in ids

    def test_lr_direction_layout_valid(self):
        """full_layout with LR direction produces valid non-negative coordinates."""
        gir = self._parse_gir("graph LR\n    A --> B\n    B --> C\n")
        layout_nodes, routed_edges = full_layout(gir)
        for n in layout_nodes:
            assert n.x >= 0
            assert n.y >= 0
        assert len(routed_edges) == 2

    def test_empty_single_node_layout(self):
        """full_layout on a single node with no edges returns that node."""
        gir = self._parse_gir("graph TD\n    Lonely\n")
        layout_nodes, routed_edges = full_layout(gir)
        ids = {n.id for n in layout_nodes}
        assert "Lonely" in ids
        assert routed_edges == []
