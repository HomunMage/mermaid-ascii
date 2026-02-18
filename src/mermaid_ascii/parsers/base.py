"""Base parser protocol."""

from __future__ import annotations

from typing import Protocol

from mermaid_ascii.syntax.types import Graph


class Parser(Protocol):
    """Protocol that all diagram parsers must implement."""

    def parse(self, src: str) -> Graph:
        """Parse source text into an AST Graph."""
        ...
