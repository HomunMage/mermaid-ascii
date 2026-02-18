"""Tests for mermaid_ascii.ast — port of Rust ast tests."""

from mermaid_ascii.syntax.types import (
    Attr,
    Edge,
    Graph,
    Node,
    Subgraph,
)
from mermaid_ascii.types import Direction, EdgeType, NodeShape


class TestDirection:
    def test_default_is_td(self):
        assert Direction.default() == Direction.TD

    def test_all_variants(self):
        assert Direction.LR != Direction.RL
        assert Direction.TD != Direction.BT


class TestNodeShape:
    def test_default_is_rectangle(self):
        assert NodeShape.default() == NodeShape.Rectangle

    def test_all_variants_distinct(self):
        shapes = [NodeShape.Rectangle, NodeShape.Rounded, NodeShape.Diamond, NodeShape.Circle]
        assert len(set(shapes)) == 4


class TestEdgeType:
    def test_all_edge_types_distinct(self):
        types = [
            EdgeType.Arrow,
            EdgeType.Line,
            EdgeType.DottedArrow,
            EdgeType.DottedLine,
            EdgeType.ThickArrow,
            EdgeType.ThickLine,
            EdgeType.BidirArrow,
            EdgeType.BidirDotted,
            EdgeType.BidirThick,
        ]
        assert len(set(types)) == 9


class TestNode:
    def test_bare_node(self):
        n = Node.bare("A")
        assert n.id == "A"
        assert n.label == "A"
        assert n.shape == NodeShape.Rectangle
        assert n.attrs == []

    def test_new_node(self):
        n = Node.new("B", "Hello", NodeShape.Diamond)
        assert n.id == "B"
        assert n.label == "Hello"
        assert n.shape == NodeShape.Diamond

    def test_node_with_attrs(self):
        n = Node(id="C", label="C", shape=NodeShape.Rounded, attrs=[Attr("color", "red")])
        assert len(n.attrs) == 1
        assert n.attrs[0].key == "color"
        assert n.attrs[0].value == "red"

    def test_node_default_shape(self):
        n = Node(id="D", label="D")
        assert n.shape == NodeShape.Rectangle


class TestEdge:
    def test_new_edge(self):
        e = Edge.new("A", "B", EdgeType.Arrow)
        assert e.from_id == "A"
        assert e.to_id == "B"
        assert e.edge_type == EdgeType.Arrow
        assert e.label is None
        assert e.attrs == []

    def test_edge_with_label(self):
        e = Edge(from_id="A", to_id="B", edge_type=EdgeType.Line, label="my label")
        assert e.label == "my label"

    def test_edge_equality(self):
        e1 = Edge.new("X", "Y", EdgeType.ThickArrow)
        e2 = Edge.new("X", "Y", EdgeType.ThickArrow)
        assert e1 == e2


class TestSubgraph:
    def test_new_subgraph(self):
        sg = Subgraph.new("Group")
        assert sg.name == "Group"
        assert sg.nodes == []
        assert sg.edges == []
        assert sg.subgraphs == []
        assert sg.description is None
        assert sg.direction is None

    def test_subgraph_with_direction(self):
        sg = Subgraph(name="G", direction=Direction.LR)
        assert sg.direction == Direction.LR

    def test_nested_subgraph(self):
        inner = Subgraph.new("Inner")
        outer = Subgraph(name="Outer", subgraphs=[inner])
        assert len(outer.subgraphs) == 1
        assert outer.subgraphs[0].name == "Inner"


class TestGraph:
    def test_new_graph_defaults(self):
        g = Graph.new()
        assert g.direction == Direction.TD
        assert g.nodes == []
        assert g.edges == []
        assert g.subgraphs == []

    def test_graph_construction(self):
        n = Node.bare("A")
        e = Edge.new("A", "B", EdgeType.Arrow)
        g = Graph(direction=Direction.LR, nodes=[n], edges=[e])
        assert g.direction == Direction.LR
        assert len(g.nodes) == 1
        assert len(g.edges) == 1
