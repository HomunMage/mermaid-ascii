"""AST data structures for Mermaid flowchart syntax.

These types represent the parsed form of the input DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mermaid_ascii.types import Direction, EdgeType, NodeShape


@dataclass
class Attr:
    key: str
    value: str


@dataclass
class Node:
    id: str
    label: str
    shape: NodeShape = field(default_factory=NodeShape.default)
    attrs: list[Attr] = field(default_factory=list)

    @classmethod
    def new(cls, id: str, label: str, shape: NodeShape) -> Node:
        return cls(id=id, label=label, shape=shape)

    @classmethod
    def bare(cls, id: str) -> Node:
        """Create a bare node (id = label, default Rectangle shape)."""
        return cls(id=id, label=id, shape=NodeShape.Rectangle)


@dataclass
class Edge:
    from_id: str
    to_id: str
    edge_type: EdgeType
    label: str | None = None
    attrs: list[Attr] = field(default_factory=list)

    @classmethod
    def new(cls, from_id: str, to_id: str, edge_type: EdgeType) -> Edge:
        return cls(from_id=from_id, to_id=to_id, edge_type=edge_type)


@dataclass
class Subgraph:
    name: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    subgraphs: list[Subgraph] = field(default_factory=list)
    description: str | None = None
    direction: Direction | None = None

    @classmethod
    def new(cls, name: str) -> Subgraph:
        return cls(name=name)


@dataclass
class Graph:
    direction: Direction = field(default_factory=Direction.default)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    subgraphs: list[Subgraph] = field(default_factory=list)

    @classmethod
    def new(cls) -> Graph:
        return cls()
