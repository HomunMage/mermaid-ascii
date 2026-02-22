"""mermaid-ascii: Mermaid flowchart syntax to ASCII/Unicode text output."""

from mermaid_ascii.layout.engine import full_layout_with_padding
from mermaid_ascii.layout.graph import GraphIR
from mermaid_ascii.parsers.registry import parse
from mermaid_ascii.renderers.ascii import AsciiRenderer
from mermaid_ascii.renderers.svg import SvgRenderer
from mermaid_ascii.syntax.types import Direction

_DIRECTION_MAP: dict[str, Direction] = {
    "LR": Direction.LR,
    "RL": Direction.RL,
    "TD": Direction.TD,
    "TB": Direction.TD,
    "BT": Direction.BT,
}


def _apply_direction(ast_graph, direction: str | None) -> None:
    """Override ast_graph.direction from a string, if provided."""
    if direction is None:
        return
    key = direction.upper()
    if key not in _DIRECTION_MAP:
        raise ValueError(f"Unknown direction '{direction}'; use LR, RL, TD, or BT")
    ast_graph.direction = _DIRECTION_MAP[key]


def render_dsl(src: str, unicode: bool = True, padding: int = 1, direction: str | None = None) -> str:
    """Parse a Mermaid flowchart string and render it to ASCII/Unicode art."""
    ast_graph = parse(src)
    _apply_direction(ast_graph, direction)
    gir = GraphIR.from_ast(ast_graph)
    if gir.node_count() == 0 and not gir.subgraph_members:
        return ""
    layout_result = full_layout_with_padding(gir, padding)
    renderer = AsciiRenderer(unicode=unicode)
    return renderer.render(layout_result)


def render_svg(src: str, padding: int = 1, direction: str | None = None) -> str:
    """Parse a Mermaid flowchart string and render it to SVG."""
    ast_graph = parse(src)
    _apply_direction(ast_graph, direction)
    gir = GraphIR.from_ast(ast_graph)
    if gir.node_count() == 0 and not gir.subgraph_members:
        return ""
    layout_result = full_layout_with_padding(gir, padding)
    renderer = SvgRenderer()
    return renderer.render(layout_result)
