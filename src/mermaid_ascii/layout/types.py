"""Layout types shared across layout engines and renderers."""

from __future__ import annotations

from dataclasses import dataclass, field

from mermaid_ascii.syntax.types import Direction, EdgeType, NodeShape


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
    label: str = ""
    shape: NodeShape = NodeShape.Rectangle


@dataclass
class Point:
    """A 2D point in character coordinates (column, row)."""

    x: int
    y: int


@dataclass
class RoutedEdge:
    """A routed edge with orthogonal waypoints."""

    from_id: str
    to_id: str
    label: str | None
    edge_type: EdgeType
    waypoints: list[Point]


@dataclass
class LayoutResult:
    """Self-contained layout output — everything renderers need."""

    nodes: list[LayoutNode]
    edges: list[RoutedEdge]
    direction: Direction
    subgraph_members: list[tuple[str, list[str]]] = field(default_factory=list)
    subgraph_descriptions: dict[str, str] = field(default_factory=dict)


# Prefix constants
DUMMY_PREFIX = "__dummy_"
COMPOUND_PREFIX = "__sg_"
