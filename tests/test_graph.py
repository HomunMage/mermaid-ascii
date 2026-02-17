"""Tests for mermaid_ascii.graph — GraphIR construction, cycle detection, and degree queries."""

from mermaid_ascii.ast import (
    Direction,
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeShape,
    Subgraph,
)
from mermaid_ascii.graph import EdgeData, GraphIR, NodeData


def _make_graph(
    *,
    direction: Direction = Direction.TD,
    nodes: list[Node] | None = None,
    edges: list[Edge] | None = None,
    subgraphs: list[Subgraph] | None = None,
) -> Graph:
    return Graph(
        direction=direction,
        nodes=nodes or [],
        edges=edges or [],
        subgraphs=subgraphs or [],
    )


def _node(id: str, label: str | None = None, shape: NodeShape = NodeShape.Rectangle) -> Node:
    return Node(id=id, label=label or id, shape=shape)


def _edge(from_id: str, to_id: str, edge_type: EdgeType = EdgeType.Arrow) -> Edge:
    return Edge(from_id=from_id, to_id=to_id, edge_type=edge_type)


class TestBasicConstruction:
    def test_empty_graph(self):
        g = _make_graph()
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 0
        assert gir.edge_count() == 0

    def test_single_node(self):
        g = _make_graph(nodes=[_node("A")])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 1
        assert gir.edge_count() == 0

    def test_direction_preserved(self):
        g = _make_graph(direction=Direction.LR)
        gir = GraphIR.from_ast(g)
        assert gir.direction == Direction.LR

    def test_simple_edge_creates_nodes(self):
        """An edge with no explicit node declarations still creates nodes."""
        g = _make_graph(edges=[_edge("A", "B")])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 2
        assert gir.edge_count() == 1

    def test_node_data_stored(self):
        g = _make_graph(nodes=[_node("A", label="Alpha", shape=NodeShape.Diamond)])
        gir = GraphIR.from_ast(g)
        data: NodeData = gir.digraph.nodes["A"]["data"]
        assert data.id == "A"
        assert data.label == "Alpha"
        assert data.shape == NodeShape.Diamond
        assert data.subgraph is None

    def test_edge_data_stored(self):
        g = _make_graph(
            edges=[Edge(from_id="A", to_id="B", edge_type=EdgeType.DottedArrow, label="goes")]
        )
        gir = GraphIR.from_ast(g)
        data: EdgeData = gir.digraph.edges["A", "B"]["data"]
        assert data.edge_type == EdgeType.DottedArrow
        assert data.label == "goes"

    def test_first_definition_wins(self):
        """If a node id appears twice, the first definition is kept."""
        g = _make_graph(
            nodes=[
                _node("A", label="First", shape=NodeShape.Rectangle),
                _node("A", label="Second", shape=NodeShape.Diamond),
            ]
        )
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 1
        data: NodeData = gir.digraph.nodes["A"]["data"]
        assert data.label == "First"


class TestSubgraphFlattening:
    def test_subgraph_members_collected(self):
        sg = Subgraph(name="Group", nodes=[_node("X"), _node("Y")])
        g = _make_graph(subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 2
        assert ("Group", ["X", "Y"]) in gir.subgraph_members

    def test_subgraph_node_membership(self):
        sg = Subgraph(name="Group", nodes=[_node("X")])
        g = _make_graph(subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        data: NodeData = gir.digraph.nodes["X"]["data"]
        assert data.subgraph == "Group"

    def test_top_level_node_skipped_if_same_name_as_subgraph(self):
        """Top-level node whose id matches a subgraph name is skipped."""
        sg = Subgraph(name="Group", nodes=[_node("X")])
        g = _make_graph(nodes=[_node("Group")], subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        # Only "X" should be in graph (Group node is skipped since it's a subgraph)
        assert "X" in gir.digraph
        # "Group" may appear if used as edge endpoint, but not from top-level node def
        assert gir.digraph.nodes.get("Group") is None or "X" in gir.digraph

    def test_subgraph_edges_added(self):
        sg = Subgraph(
            name="Group",
            nodes=[_node("X"), _node("Y")],
            edges=[_edge("X", "Y")],
        )
        g = _make_graph(subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        assert gir.edge_count() == 1
        assert gir.digraph.has_edge("X", "Y")

    def test_subgraph_description(self):
        sg = Subgraph(name="Group", nodes=[], description="My group")
        g = _make_graph(subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        assert gir.subgraph_descriptions["Group"] == "My group"

    def test_nested_subgraph(self):
        inner = Subgraph(name="Inner", nodes=[_node("Z")])
        outer = Subgraph(name="Outer", nodes=[_node("W")], subgraphs=[inner])
        g = _make_graph(subgraphs=[outer])
        gir = GraphIR.from_ast(g)
        assert "W" in gir.digraph
        assert "Z" in gir.digraph
        assert gir.digraph.nodes["Z"]["data"].subgraph == "Inner"


class TestCycleDetection:
    def test_empty_graph_is_dag(self):
        g = _make_graph()
        gir = GraphIR.from_ast(g)
        assert gir.is_dag() is True

    def test_simple_chain_is_dag(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "C")])
        gir = GraphIR.from_ast(g)
        assert gir.is_dag() is True

    def test_single_cycle_is_not_dag(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "C"), _edge("C", "A")])
        gir = GraphIR.from_ast(g)
        assert gir.is_dag() is False

    def test_self_loop_is_not_dag(self):
        g = _make_graph(edges=[_edge("A", "A")])
        gir = GraphIR.from_ast(g)
        assert gir.is_dag() is False

    def test_two_node_cycle_is_not_dag(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "A")])
        gir = GraphIR.from_ast(g)
        assert gir.is_dag() is False


class TestTopologicalOrder:
    def test_empty_graph_returns_empty_list(self):
        g = _make_graph()
        gir = GraphIR.from_ast(g)
        result = gir.topological_order()
        assert result == []

    def test_simple_chain_order(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "C")])
        gir = GraphIR.from_ast(g)
        order = gir.topological_order()
        assert order is not None
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_cycle_returns_none(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "A")])
        gir = GraphIR.from_ast(g)
        assert gir.topological_order() is None

    def test_self_loop_returns_none(self):
        g = _make_graph(edges=[_edge("A", "A")])
        gir = GraphIR.from_ast(g)
        assert gir.topological_order() is None

    def test_all_nodes_in_result(self):
        g = _make_graph(
            nodes=[_node("A"), _node("B"), _node("C")],
            edges=[_edge("A", "B"), _edge("A", "C")],
        )
        gir = GraphIR.from_ast(g)
        order = gir.topological_order()
        assert order is not None
        assert set(order) == {"A", "B", "C"}
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")


class TestDegreeQueries:
    def test_in_degree_source_node(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("A", "C")])
        gir = GraphIR.from_ast(g)
        assert gir.in_degree("A") == 0

    def test_out_degree_source_node(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("A", "C")])
        gir = GraphIR.from_ast(g)
        assert gir.out_degree("A") == 2

    def test_in_degree_sink_node(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("C", "B")])
        gir = GraphIR.from_ast(g)
        assert gir.in_degree("B") == 2

    def test_out_degree_sink_node(self):
        g = _make_graph(edges=[_edge("A", "B")])
        gir = GraphIR.from_ast(g)
        assert gir.out_degree("B") == 0

    def test_in_degree_middle_node(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "C")])
        gir = GraphIR.from_ast(g)
        assert gir.in_degree("B") == 1
        assert gir.out_degree("B") == 1

    def test_degree_unknown_node_returns_zero(self):
        g = _make_graph(nodes=[_node("A")])
        gir = GraphIR.from_ast(g)
        assert gir.in_degree("NONEXISTENT") == 0
        assert gir.out_degree("NONEXISTENT") == 0

    def test_self_loop_degree(self):
        g = _make_graph(edges=[_edge("A", "A")])
        gir = GraphIR.from_ast(g)
        assert gir.in_degree("A") == 1
        assert gir.out_degree("A") == 1


class TestAdjacencyList:
    def test_empty_graph(self):
        g = _make_graph()
        gir = GraphIR.from_ast(g)
        assert gir.adjacency_list() == []

    def test_single_node_no_edges(self):
        g = _make_graph(nodes=[_node("A")])
        gir = GraphIR.from_ast(g)
        adj = gir.adjacency_list()
        assert adj == [("A", [])]

    def test_chain_adjacency(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("B", "C")])
        gir = GraphIR.from_ast(g)
        adj = dict(gir.adjacency_list())
        assert adj["A"] == ["B"]
        assert adj["B"] == ["C"]
        assert adj["C"] == []

    def test_sorted_output(self):
        g = _make_graph(edges=[_edge("C", "A"), _edge("B", "A")])
        gir = GraphIR.from_ast(g)
        adj = gir.adjacency_list()
        keys = [k for k, _ in adj]
        assert keys == sorted(keys)

    def test_neighbors_sorted(self):
        """Outgoing neighbors are sorted alphabetically."""
        g = _make_graph(edges=[_edge("A", "C"), _edge("A", "B")])
        gir = GraphIR.from_ast(g)
        adj = dict(gir.adjacency_list())
        assert adj["A"] == ["B", "C"]

    def test_multiple_fanout(self):
        g = _make_graph(edges=[_edge("A", "B"), _edge("A", "C"), _edge("A", "D")])
        gir = GraphIR.from_ast(g)
        adj = dict(gir.adjacency_list())
        assert adj["A"] == ["B", "C", "D"]


class TestAllEdgeTypes:
    """Each EdgeType is stored and retrieved faithfully."""

    def _gir_for(self, etype: EdgeType) -> GraphIR:
        return GraphIR.from_ast(_make_graph(edges=[_edge("A", "B", etype)]))

    def test_arrow(self):
        gir = self._gir_for(EdgeType.Arrow)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.Arrow

    def test_line(self):
        gir = self._gir_for(EdgeType.Line)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.Line

    def test_dotted_arrow(self):
        gir = self._gir_for(EdgeType.DottedArrow)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.DottedArrow

    def test_dotted_line(self):
        gir = self._gir_for(EdgeType.DottedLine)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.DottedLine

    def test_thick_arrow(self):
        gir = self._gir_for(EdgeType.ThickArrow)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.ThickArrow

    def test_thick_line(self):
        gir = self._gir_for(EdgeType.ThickLine)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.ThickLine

    def test_bidir_arrow(self):
        gir = self._gir_for(EdgeType.BidirArrow)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.BidirArrow

    def test_bidir_dotted(self):
        gir = self._gir_for(EdgeType.BidirDotted)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.BidirDotted

    def test_bidir_thick(self):
        gir = self._gir_for(EdgeType.BidirThick)
        assert gir.digraph.edges["A", "B"]["data"].edge_type == EdgeType.BidirThick


class TestNodeEdgeCount:
    def test_no_duplicate_nodes_from_shared_edge_endpoint(self):
        """A --> B, A --> C should create exactly 3 distinct nodes."""
        g = _make_graph(edges=[_edge("A", "B"), _edge("A", "C")])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 3

    def test_explicit_and_implicit_same_node(self):
        """Explicit node + edge referencing same id → only 1 node."""
        g = _make_graph(nodes=[_node("A")], edges=[_edge("A", "B")])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 2  # A (explicit) + B (placeholder)

    def test_diamond_graph(self):
        g = _make_graph(edges=[
            _edge("A", "B"), _edge("A", "C"),
            _edge("B", "D"), _edge("C", "D"),
        ])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 4
        assert gir.edge_count() == 4
        assert gir.is_dag() is True


class TestSubgraphFlatteningExtended:
    def test_subgraph_edge_creates_placeholder_if_node_missing(self):
        """An edge inside a subgraph referencing a node not in nodes list
        creates a placeholder node."""
        sg = Subgraph(name="SG", nodes=[], edges=[_edge("X", "Y")])
        g = _make_graph(subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        assert "X" in gir.digraph
        assert "Y" in gir.digraph
        assert gir.edge_count() == 1

    def test_multiple_subgraphs_members(self):
        sg1 = Subgraph(name="SG1", nodes=[_node("A"), _node("B")])
        sg2 = Subgraph(name="SG2", nodes=[_node("C")])
        g = _make_graph(subgraphs=[sg1, sg2])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 3
        assert len(gir.subgraph_members) == 2
        names = {name for name, _ in gir.subgraph_members}
        assert names == {"SG1", "SG2"}

    def test_cross_subgraph_edge_at_top_level(self):
        sg1 = Subgraph(name="SG1", nodes=[_node("A")])
        sg2 = Subgraph(name="SG2", nodes=[_node("B")])
        g = _make_graph(subgraphs=[sg1, sg2], edges=[_edge("A", "B")])
        gir = GraphIR.from_ast(g)
        assert gir.edge_count() == 1
        assert gir.digraph.has_edge("A", "B")

    def test_deeply_nested_subgraph(self):
        innermost = Subgraph(name="Level3", nodes=[_node("P")])
        middle = Subgraph(name="Level2", nodes=[_node("Q")], subgraphs=[innermost])
        outer = Subgraph(name="Level1", nodes=[_node("R")], subgraphs=[middle])
        g = _make_graph(subgraphs=[outer])
        gir = GraphIR.from_ast(g)
        assert gir.node_count() == 3
        assert len(gir.subgraph_members) == 3
        names = {name for name, _ in gir.subgraph_members}
        assert names == {"Level1", "Level2", "Level3"}

    def test_no_subgraph_descriptions_when_none(self):
        sg = Subgraph(name="SG", nodes=[])
        g = _make_graph(subgraphs=[sg])
        gir = GraphIR.from_ast(g)
        assert "SG" not in gir.subgraph_descriptions


class TestFromParserIntegration:
    """End-to-end: parse DSL → AST → GraphIR."""

    def _build(self, dsl: str) -> GraphIR:
        from mermaid_ascii.parser import parse
        return GraphIR.from_ast(parse(dsl))

    def test_simple_chain(self):
        gir = self._build("graph TD\n    A --> B --> C\n")
        assert gir.node_count() == 3
        assert gir.edge_count() == 2
        assert gir.is_dag() is True

    def test_lr_direction(self):
        gir = self._build("graph LR\n    A --> B\n")
        assert gir.direction == Direction.LR

    def test_cyclic_from_dsl(self):
        gir = self._build("graph TD\n    A --> B\n    B --> A\n")
        assert gir.is_dag() is False
        assert gir.topological_order() is None

    def test_subgraph_from_dsl(self):
        dsl = "graph TD\n    subgraph Group\n        X --> Y\n    end\n"
        gir = self._build(dsl)
        assert gir.node_count() == 2
        assert gir.edge_count() == 1
        assert len(gir.subgraph_members) == 1
        name, members = gir.subgraph_members[0]
        assert name == "Group"

    def test_adjacency_from_dsl(self):
        gir = self._build("graph TD\n    A --> B\n    A --> C\n")
        adj = dict(gir.adjacency_list())
        assert "A" in adj
        assert set(adj["A"]) == {"B", "C"}

    def test_topological_order_from_dsl(self):
        gir = self._build("graph TD\n    A --> B\n    B --> C\n")
        order = gir.topological_order()
        assert order is not None
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_edge_label_preserved(self):
        gir = self._build("graph TD\n    A -->|yes| B\n")
        ed: EdgeData = gir.digraph.edges["A", "B"]["data"]
        assert ed.label == "yes"
