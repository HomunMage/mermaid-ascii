"""SVG renderer — renders Layout IR to SVG string."""

from __future__ import annotations

from mermaid_ascii.layout.types import (
    COMPOUND_PREFIX,
    DUMMY_PREFIX,
    LayoutNode,
    LayoutResult,
    RoutedEdge,
)
from mermaid_ascii.syntax.types import Direction, EdgeType, NodeShape

# ─── Constants ──────────────────────────────────────────────────────────────

CELL_W = 10  # pixels per character column
CELL_H = 20  # pixels per character row
FONT_SIZE = 14
FONT_FAMILY = "monospace"
PADDING = 20  # canvas padding in pixels

_STROKE_STYLES: dict[EdgeType, str] = {
    EdgeType.Arrow: "",
    EdgeType.Line: "",
    EdgeType.DottedArrow: 'stroke-dasharray="6 4"',
    EdgeType.DottedLine: 'stroke-dasharray="6 4"',
    EdgeType.ThickArrow: 'stroke-width="3"',
    EdgeType.ThickLine: 'stroke-width="3"',
    EdgeType.BidirArrow: "",
    EdgeType.BidirDotted: 'stroke-dasharray="6 4"',
    EdgeType.BidirThick: 'stroke-width="3"',
}

_ARROW_TYPES = {
    EdgeType.Arrow,
    EdgeType.DottedArrow,
    EdgeType.ThickArrow,
    EdgeType.BidirArrow,
    EdgeType.BidirDotted,
    EdgeType.BidirThick,
}

_BIDIR_TYPES = {EdgeType.BidirArrow, EdgeType.BidirDotted, EdgeType.BidirThick}

_FILL_STROKE = 'fill="white" stroke="black" stroke-width="1.5"'
_SG_STROKE = 'fill="none" stroke="#888" stroke-width="1" stroke-dasharray="4 2"'


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _font(size: int = FONT_SIZE) -> str:
    return f'font-family="{FONT_FAMILY}" font-size="{size}"'


# ─── Coordinate Helpers ─────────────────────────────────────────────────────


def _px(col: int) -> int:
    return PADDING + col * CELL_W


def _py(row: int) -> int:
    return PADDING + row * CELL_H


def _node_rect(ln: LayoutNode) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) in pixel coordinates."""
    return (_px(ln.x), _py(ln.y), ln.width * CELL_W, ln.height * CELL_H)


# ─── Shape Rendering ────────────────────────────────────────────────────────


def _render_node(ln: LayoutNode) -> str:
    x, y, w, h = _node_rect(ln)
    cx, cy = x + w // 2, y + h // 2
    label = _escape(ln.label)
    lines = label.split("\n")

    font = _font()
    if len(lines) == 1:
        label_svg = f'<text x="{cx}" y="{cy}" dominant-baseline="central" text-anchor="middle" {font}>{lines[0]}</text>'
    else:
        total_h = len(lines) * (FONT_SIZE + 2)
        start_y = cy - total_h // 2 + FONT_SIZE // 2
        tspans = "".join(
            f'<tspan x="{cx}" y="{start_y + i * (FONT_SIZE + 2)}">{line}</tspan>' for i, line in enumerate(lines)
        )
        label_svg = f'<text text-anchor="middle" {font}>{tspans}</text>'

    fs = _FILL_STROKE
    if ln.shape == NodeShape.Rectangle:
        shape_svg = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="0" {fs}/>'
    elif ln.shape == NodeShape.Rounded:
        r = min(w, h) // 4
        shape_svg = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" {fs}/>'
    elif ln.shape == NodeShape.Diamond:
        pts = f"{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}"
        shape_svg = f'<polygon points="{pts}" {fs}/>'
    elif ln.shape == NodeShape.Circle:
        shape_svg = f'<ellipse cx="{cx}" cy="{cy}" rx="{w // 2}" ry="{h // 2}" {fs}/>'
    else:
        shape_svg = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {fs}/>'

    return f"{shape_svg}\n{label_svg}"


def _render_compound(
    ln: LayoutNode,
    sg_name: str,
    description: str | None,
) -> str:
    x, y, w, h = _node_rect(ln)
    font = _font(FONT_SIZE - 2)
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" {_SG_STROKE}/>',
        f'<text x="{x + 8}" y="{y + FONT_SIZE + 4}" {font} fill="#666">{sg_name}</text>',
    ]
    if description is not None:
        desc = _escape(description)
        parts.append(f'<text x="{x + 8}" y="{y + h - 6}" {font} fill="#666">{desc}</text>')
    return "\n".join(parts)


# ─── Edge Rendering ─────────────────────────────────────────────────────────


def _render_edge(
    re: RoutedEdge,
    marker_end: str,
    marker_start: str,
) -> str:
    if len(re.waypoints) < 2:
        return ""

    style = _STROKE_STYLES.get(re.edge_type, "")
    markers = ""
    if re.edge_type in _ARROW_TYPES:
        markers += f' marker-end="url(#{marker_end})"'
    if re.edge_type in _BIDIR_TYPES:
        markers += f' marker-start="url(#{marker_start})"'

    pts = " ".join(f"{_px(p.x)},{_py(p.y)}" for p in re.waypoints)
    parts = [
        f'<polyline points="{pts}" fill="none" stroke="black" stroke-width="1.5" {style}{markers}/>',
    ]

    if re.label is not None:
        mid = len(re.waypoints) // 2
        lp = re.waypoints[mid]
        lx, ly = _px(lp.x), _py(lp.y) - 8
        font = _font(FONT_SIZE - 2)
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" {font} fill="#333">{_escape(re.label)}</text>')

    return "\n".join(parts)


# ─── Subgraph Borders ───────────────────────────────────────────────────────


def _render_subgraph_borders(
    subgraph_members: list[tuple[str, list[str]]],
    nodes: list[LayoutNode],
) -> str:
    node_pos: dict[str, LayoutNode] = {n.id: n for n in nodes}
    parts: list[str] = []

    for sg_name, members in subgraph_members:
        if not members:
            continue

        min_x = min_y = 10**9
        max_x = max_y = -(10**9)

        for member_id in members:
            ln = node_pos.get(member_id)
            if ln is None:
                continue
            px, py, pw, ph = _node_rect(ln)
            min_x, min_y = min(min_x, px), min(min_y, py)
            max_x, max_y = max(max_x, px + pw), max(max_y, py + ph)

        if min_x == 10**9:
            continue

        margin = 15
        bx, by = min_x - margin, min_y - margin
        bw = max_x - min_x + 2 * margin
        bh = max_y - min_y + 2 * margin
        font = _font(FONT_SIZE - 2)

        parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" {_SG_STROKE}/>')
        parts.append(f'<text x="{bx + 8}" y="{by + FONT_SIZE + 2}" {font} fill="#666">{sg_name}</text>')

    return "\n".join(parts)


# ─── Direction Transforms ───────────────────────────────────────────────────


def _transpose_node(ln: LayoutNode) -> LayoutNode:
    return LayoutNode(
        id=ln.id,
        layer=ln.layer,
        order=ln.order,
        x=ln.y,
        y=ln.x,
        width=ln.height,
        height=ln.width,
        label=ln.label,
        shape=ln.shape,
    )


def _transpose_edge(re: RoutedEdge) -> RoutedEdge:
    from mermaid_ascii.layout.types import Point

    return RoutedEdge(
        from_id=re.from_id,
        to_id=re.to_id,
        label=re.label,
        edge_type=re.edge_type,
        waypoints=[Point(x=p.y, y=p.x) for p in re.waypoints],
    )


# ─── Public Renderer ────────────────────────────────────────────────────────


class SvgRenderer:
    """SVG renderer — consumes Layout IR, produces SVG string."""

    def render(self, result: LayoutResult) -> str:
        if result.direction in (Direction.LR, Direction.RL):
            nodes = [_transpose_node(n) for n in result.nodes]
            edges = [_transpose_edge(re) for re in result.edges]
        else:
            nodes = list(result.nodes)
            edges = list(result.edges)

        has_compounds = any(n.id.startswith(COMPOUND_PREFIX) for n in nodes)
        real_nodes = [n for n in nodes if not n.id.startswith(DUMMY_PREFIX) and not n.id.startswith(COMPOUND_PREFIX)]
        compound_nodes = [n for n in nodes if n.id.startswith(COMPOUND_PREFIX)]

        if not real_nodes and not compound_nodes:
            return ""

        # Compute canvas size
        max_col = max_row = 0
        for n in nodes:
            if n.id.startswith(DUMMY_PREFIX):
                continue
            max_col = max(max_col, n.x + n.width + 2)
            max_row = max(max_row, n.y + n.height + 2)
        for re in edges:
            for p in re.waypoints:
                max_col = max(max_col, p.x + 2)
                max_row = max(max_row, p.y + 2)

        svg_w = PADDING * 2 + max_col * CELL_W
        svg_h = PADDING * 2 + max_row * CELL_H

        transform = ""
        if result.direction == Direction.BT:
            transform = f'<g transform="translate(0,{svg_h}) scale(1,-1)">'
        elif result.direction == Direction.RL:
            transform = f'<g transform="translate({svg_w},0) scale(-1,1)">'

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
            "<defs>",
            '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">',
            '    <polygon points="0 0, 10 3.5, 0 7" fill="black"/>',
            "  </marker>",
            '  <marker id="arrowhead-rev" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">',
            '    <polygon points="10 0, 0 3.5, 10 7" fill="black"/>',
            "  </marker>",
            "</defs>",
            f'<rect width="{svg_w}" height="{svg_h}" fill="white"/>',
        ]

        if transform:
            parts.append(transform)

        # Subgraph borders
        if has_compounds:
            for ln in compound_nodes:
                sg_name = ln.id[len(COMPOUND_PREFIX) :]
                desc = result.subgraph_descriptions.get(sg_name)
                parts.append(_render_compound(ln, sg_name, desc))
        else:
            borders = _render_subgraph_borders(
                result.subgraph_members,
                nodes,
            )
            if borders:
                parts.append(borders)

        # Edges (behind nodes) — sort for deterministic output across implementations
        edges.sort(key=lambda e: (e.from_id, e.to_id))
        for re in edges:
            parts.append(
                _render_edge(re, "arrowhead", "arrowhead-rev"),
            )

        # Nodes (on top)
        for ln in real_nodes:
            parts.append(_render_node(ln))

        if transform:
            parts.append("</g>")

        parts.append("</svg>")
        return "\n".join(parts)
