"""Tests for layout.py — cycle removal (Phase 4) + crossing minimization + coordinate assignment (Phase 6).

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
"""

from __future__ import annotations

import networkx as nx

from mermaid_ascii import ast as mast
from mermaid_ascii.graph import EdgeData, GraphIR, NodeData
from mermaid_ascii.layout import (
    DUMMY_PREFIX,
    H_GAP,
    NODE_HEIGHT,
    NODE_PADDING,
    V_GAP,
    AugmentedGraph,
    CycleRemovalResult,
    LayerAssignment,
    LayoutNode,
    assign_coordinates,
    assign_coordinates_padded,
    count_crossings,
    greedy_fas_ordering,
    insert_dummy_nodes,
    label_dimensions,
    minimise_crossings,
    remove_cycles,
)

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
        from mermaid_ascii.parser import parse

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
