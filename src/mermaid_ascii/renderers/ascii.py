"""ASCII/Unicode text renderer."""

from __future__ import annotations

from mermaid_ascii.layout.types import COMPOUND_PREFIX, DUMMY_PREFIX, LayoutNode, LayoutResult, Point, RoutedEdge
from mermaid_ascii.renderers.canvas import Canvas, Rect
from mermaid_ascii.renderers.charset import BoxChars, CharSet
from mermaid_ascii.types import Direction, EdgeType, NodeShape

# ─── Node Rendering ──────────────────────────────────────────────────────────


def _box_chars_for_shape(shape: NodeShape, cs: CharSet) -> BoxChars:
    if shape == NodeShape.Rectangle:
        return BoxChars.for_charset(cs)
    elif shape == NodeShape.Rounded:
        if cs == CharSet.Ascii:
            return BoxChars.ascii()
        bc = BoxChars.unicode()
        bc.top_left = "╭"
        bc.top_right = "╮"
        bc.bottom_left = "╰"
        bc.bottom_right = "╯"
        return bc
    elif shape == NodeShape.Diamond:
        bc = BoxChars.for_charset(cs)
        bc.top_left = "/"
        bc.top_right = "\\"
        bc.bottom_left = "\\"
        bc.bottom_right = "/"
        return bc
    else:  # Circle
        bc = BoxChars.for_charset(cs)
        bc.top_left = "("
        bc.top_right = ")"
        bc.bottom_left = "("
        bc.bottom_right = ")"
        bc.vertical = " "
        return bc


def _paint_node(canvas: Canvas, ln: LayoutNode, shape: NodeShape, label: str) -> None:
    bc = _box_chars_for_shape(shape, canvas.charset)
    rect = Rect(ln.x, ln.y, ln.width, ln.height)
    canvas.draw_box(rect, bc)

    inner_w = max(0, ln.width - 2)
    lines = label.split("\n")
    for i, line in enumerate(lines):
        label_row = ln.y + 1 + i
        line_len = len(line)
        pad = max(0, inner_w - line_len) // 2
        col_start = ln.x + 1 + pad
        canvas.write_str(col_start, label_row, line)


def _paint_compound_node(canvas: Canvas, ln: LayoutNode, sg_name: str, description: str | None) -> None:
    bc = BoxChars.for_charset(canvas.charset)
    rect = Rect(ln.x, ln.y, ln.width, ln.height)
    canvas.draw_box(rect, bc)

    inner_w = max(0, ln.width - 2)
    title_pad = max(0, inner_w - len(sg_name)) // 2
    title_col = ln.x + 1 + title_pad
    title_row = ln.y + 1
    canvas.write_str(title_col, title_row, sg_name)

    if description is not None:
        desc_row = ln.y + ln.height - 2
        desc_pad = max(0, inner_w - len(description)) // 2
        desc_col = ln.x + 1 + desc_pad
        canvas.write_str(desc_col, desc_row, description)


def _paint_subgraph_borders(
    subgraph_members: list[tuple[str, list[str]]], layout_nodes: list[LayoutNode], canvas: Canvas
) -> None:
    node_pos: dict[str, LayoutNode] = {n.id: n for n in layout_nodes}
    bc = BoxChars.for_charset(canvas.charset)

    for sg_name, members in subgraph_members:
        if not members:
            continue

        min_x = min_y = 10**9
        max_x = max_y = 0

        for member_id in members:
            ln = node_pos.get(member_id)
            if ln is None:
                continue
            if ln.x < min_x:
                min_x = ln.x
            if ln.y < min_y:
                min_y = ln.y
            right = ln.x + ln.width
            bottom = ln.y + ln.height
            if right > max_x:
                max_x = right
            if bottom > max_y:
                max_y = bottom

        if min_x == 10**9:
            continue

        margin_x = 2
        margin_y = 1
        bx = max(0, min_x - margin_x)
        by = max(0, min_y - margin_y)
        bw = (max_x + margin_x) - bx
        bh = (max_y + margin_y) - by

        rect = Rect(bx, by, bw, bh)
        canvas.draw_box(rect, bc)

        label = f" {sg_name} "
        label_col = bx + 2
        if len(label) + 4 <= bw:
            canvas.write_str(label_col, by, label)


# ─── Edge Rendering ──────────────────────────────────────────────────────────


def _line_chars_for(edge_type: EdgeType, cs: CharSet) -> tuple[str, str]:
    bc = BoxChars.for_charset(cs)
    if edge_type in (EdgeType.ThickArrow, EdgeType.ThickLine, EdgeType.BidirThick):
        return ("═", "║")
    elif edge_type in (EdgeType.DottedArrow, EdgeType.DottedLine, EdgeType.BidirDotted):
        return ("╌", "╎")
    else:
        return (bc.horizontal, bc.vertical)


def _paint_edge(canvas: Canvas, re: RoutedEdge, edge_type: EdgeType) -> None:
    if len(re.waypoints) < 2:
        return

    cs = canvas.charset
    h_ch, v_ch = _line_chars_for(edge_type, cs)
    bc = BoxChars.for_charset(cs)

    for i in range(len(re.waypoints) - 1):
        p0 = re.waypoints[i]
        p1 = re.waypoints[i + 1]
        if p0.y == p1.y:
            canvas.hline(p0.y, p0.x, p1.x, h_ch)
        elif p0.x == p1.x:
            canvas.vline(p0.x, p0.y, p1.y, v_ch)

    arrow_types = {
        EdgeType.Arrow,
        EdgeType.DottedArrow,
        EdgeType.ThickArrow,
        EdgeType.BidirArrow,
        EdgeType.BidirDotted,
        EdgeType.BidirThick,
    }
    bidir_types = {EdgeType.BidirArrow, EdgeType.BidirDotted, EdgeType.BidirThick}

    if edge_type in arrow_types:
        last = re.waypoints[-1]
        prev = re.waypoints[-2]
        if last.y < prev.y:
            arrow = bc.arrow_up
        elif last.y > prev.y:
            arrow = bc.arrow_down
        elif last.x > prev.x:
            arrow = bc.arrow_right
        else:
            arrow = bc.arrow_left
        canvas.set(last.x, last.y, arrow)

    if edge_type in bidir_types:
        first = re.waypoints[0]
        second = re.waypoints[1]
        if first.y < second.y:
            start_arrow = bc.arrow_up
        elif first.y > second.y:
            start_arrow = bc.arrow_down
        elif first.x > second.x:
            start_arrow = bc.arrow_right
        else:
            start_arrow = bc.arrow_left
        canvas.set(first.x, first.y, start_arrow)

    if re.label is not None:
        mid = len(re.waypoints) // 2
        lp = re.waypoints[mid]
        label_y = max(0, lp.y - 1)
        canvas.write_str(lp.x, label_y, re.label)


# ─── Direction Transforms ────────────────────────────────────────────────────


def _transpose_layout(nodes: list[LayoutNode], edges: list[RoutedEdge]) -> None:
    for n in nodes:
        n.x, n.y = n.y, n.x
        n.width, n.height = n.height, n.width
    for re in edges:
        for p in re.waypoints:
            p.x, p.y = p.y, p.x


def remap_char_vertical(c: str) -> str:
    table = {
        "▼": "▲",
        "▲": "▼",
        "v": "^",
        "^": "v",
        "┌": "└",
        "└": "┌",
        "┐": "┘",
        "┘": "┐",
        "╭": "╰",
        "╰": "╭",
        "╮": "╯",
        "╯": "╮",
        "┬": "┴",
        "┴": "┬",
    }
    return table.get(c, c)


def remap_char_horizontal(c: str) -> str:
    table = {
        "►": "◄",
        "◄": "►",
        ">": "<",
        "<": ">",
        "┌": "┐",
        "┐": "┌",
        "└": "┘",
        "┘": "└",
        "╭": "╮",
        "╮": "╭",
        "╰": "╯",
        "╯": "╰",
        "├": "┤",
        "┤": "├",
    }
    return table.get(c, c)


def flip_vertical(s: str) -> str:
    lines = s.splitlines()
    flipped = ["".join(remap_char_vertical(c) for c in line) for line in reversed(lines)]
    return "\n".join(flipped) + "\n"


def flip_horizontal(s: str) -> str:
    lines = s.splitlines()
    max_width = max((len(line) for line in lines), default=0)
    flipped = []
    for line in lines:
        chars = list(line)
        pad = max_width - len(chars)
        chars.extend([" "] * pad)
        chars.reverse()
        remapped = "".join(remap_char_horizontal(c) for c in chars)
        flipped.append(remapped.rstrip())
    return "\n".join(flipped) + "\n"


# ─── Canvas Sizing ───────────────────────────────────────────────────────────


def _canvas_dimensions(layout_nodes: list[LayoutNode], routed_edges: list[RoutedEdge]) -> tuple[int, int]:
    max_col = 40
    max_row = 10
    for n in layout_nodes:
        if n.id.startswith(DUMMY_PREFIX):
            continue
        max_col = max(max_col, n.x + n.width + 2)
        max_row = max(max_row, n.y + n.height + 4)
    for re in routed_edges:
        for p in re.waypoints:
            max_col = max(max_col, p.x + 4)
            max_row = max(max_row, p.y + 4)
    return (max_col, max_row)


# ─── Public Renderer ─────────────────────────────────────────────────────────


class AsciiRenderer:
    """ASCII/Unicode text renderer."""

    def __init__(self, unicode: bool = True) -> None:
        self.unicode = unicode

    def render(self, result: LayoutResult) -> str:
        cs = CharSet.Unicode if self.unicode else CharSet.Ascii

        if result.direction in (Direction.TD, Direction.BT):
            nodes = list(result.nodes)
            edges = list(result.edges)
        else:  # LR or RL
            nodes = [
                LayoutNode(
                    id=n.id,
                    layer=n.layer,
                    order=n.order,
                    x=n.x,
                    y=n.y,
                    width=n.width,
                    height=n.height,
                    label=n.label,
                    shape=n.shape,
                )
                for n in result.nodes
            ]
            edges = [
                RoutedEdge(
                    from_id=re.from_id,
                    to_id=re.to_id,
                    label=re.label,
                    edge_type=re.edge_type,
                    waypoints=[Point(x=p.x, y=p.y) for p in re.waypoints],
                )
                for re in result.edges
            ]
            _transpose_layout(nodes, edges)

        has_compounds = any(n.id.startswith(COMPOUND_PREFIX) for n in nodes)
        real_nodes = [n for n in nodes if not n.id.startswith(DUMMY_PREFIX) and not n.id.startswith(COMPOUND_PREFIX)]
        compound_nodes = [n for n in nodes if n.id.startswith(COMPOUND_PREFIX)]

        if not real_nodes and not compound_nodes:
            return ""

        width, height = _canvas_dimensions(nodes, edges)
        canvas = Canvas(width, height, cs)

        if has_compounds:
            for ln in compound_nodes:
                sg_name = ln.id[len(COMPOUND_PREFIX) :]
                desc = result.subgraph_descriptions.get(sg_name)
                _paint_compound_node(canvas, ln, sg_name, desc)
        else:
            _paint_subgraph_borders(result.subgraph_members, nodes, canvas)

        for ln in real_nodes:
            _paint_node(canvas, ln, ln.shape, ln.label)

        for re in edges:
            _paint_edge(canvas, re, re.edge_type)

        rendered = canvas.to_string()

        if result.direction == Direction.BT:
            return flip_vertical(rendered)
        elif result.direction == Direction.RL:
            return flip_horizontal(rendered)
        return rendered
