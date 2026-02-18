"""Parser registry — auto-detect diagram type and dispatch to the right parser."""

from __future__ import annotations

from mermaid_ascii.parsers.flowchart import FlowchartParser
from mermaid_ascii.syntax.types import Graph


def detect_type(src: str) -> str:
    """Detect the diagram type from source text. Returns 'flowchart' etc."""
    stripped = src.strip()
    for line in stripped.split("\n"):
        line = line.strip()
        if not line or line.startswith("%%"):
            continue
        lower = line.lower()
        if lower.startswith("flowchart") or lower.startswith("graph"):
            return "flowchart"
        break
    return "flowchart"


_PARSERS = {
    "flowchart": FlowchartParser,
}


def parse(src: str) -> Graph:
    """Auto-detect diagram type and parse to AST."""
    diagram_type = detect_type(src)
    parser_cls = _PARSERS.get(diagram_type)
    if parser_cls is None:
        raise ValueError(f"Unsupported diagram type: {diagram_type}")
    return parser_cls().parse(src)
