"""Base renderer protocol."""

from __future__ import annotations

from typing import Protocol

from mermaid_ascii.layout.types import LayoutNode, RoutedEdge
from mermaid_ascii.syntax.graph import GraphIR


class Renderer(Protocol):
    """Protocol that all renderers must implement."""

    def render(
        self,
        gir: GraphIR,
        layout_nodes: list[LayoutNode],
        routed_edges: list[RoutedEdge],
    ) -> str:
        """Render a laid-out graph to an output string."""
        ...
