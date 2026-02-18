"""Base renderer protocol."""

from __future__ import annotations

from typing import Protocol

from mermaid_ascii.layout.types import LayoutResult


class Renderer(Protocol):
    """Protocol that all renderers must implement."""

    def render(self, result: LayoutResult) -> str:
        """Render a laid-out graph to an output string."""
        ...
