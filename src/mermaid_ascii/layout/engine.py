"""Layout engine convenience functions."""

from __future__ import annotations

from mermaid_ascii.layout.sugiyama import (
    NODE_PADDING,
    SugiyamaLayout,
    _compute_orthogonal_waypoints,
    assign_coordinates_padded,
)
from mermaid_ascii.layout.types import LayoutNode, RoutedEdge
from mermaid_ascii.syntax.graph import GraphIR
from mermaid_ascii.types import Direction


def full_layout(gir: GraphIR) -> tuple[list[LayoutNode], list[RoutedEdge]]:
    """Run the full layout pipeline with default padding."""
    return full_layout_with_padding(gir, NODE_PADDING)


def full_layout_with_padding(gir: GraphIR, padding: int) -> tuple[list[LayoutNode], list[RoutedEdge]]:
    """Run the default (Sugiyama) layout pipeline."""
    engine = SugiyamaLayout()
    return engine.layout(gir, padding)


def assign_coordinates(ordering, aug):
    """Assign coordinates with default padding and TD direction."""
    return assign_coordinates_padded(ordering, aug, NODE_PADDING, {}, Direction.TD)


def compute_orthogonal_waypoints(from_node, to_node, layer_top_y, layer_bottom_y, dummy_xs):
    """Compute orthogonal waypoints for edge routing."""
    return _compute_orthogonal_waypoints(from_node, to_node, layer_top_y, layer_bottom_y, dummy_xs)
