"""Flowchart parser — hand-rolled recursive descent.

Parses Mermaid flowchart/graph DSL into the AST types from ast.types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mermaid_ascii.syntax.types import Edge, Graph, Node, Subgraph
from mermaid_ascii.types import Direction, EdgeType, NodeShape

# ─── Tokenizer ───────────────────────────────────────────────────────────────

_COMMENT_RE = re.compile(r"%%[^\n]*")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\r\n|\n|\r")

_EDGE_PATTERNS: list[tuple[str, EdgeType]] = [
    ("<-.->", EdgeType.BidirDotted),
    ("<==>", EdgeType.BidirThick),
    ("<-->", EdgeType.BidirArrow),
    ("-.->", EdgeType.DottedArrow),
    ("==>", EdgeType.ThickArrow),
    ("-->", EdgeType.Arrow),
    ("-.-", EdgeType.DottedLine),
    ("===", EdgeType.ThickLine),
    ("---", EdgeType.Line),
]

_NODE_ID_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_-]*")
_DIRECTION_RE = re.compile(r"TD|TB|LR|RL|BT")
_BARE_LABEL_RE = re.compile(r"[^\]\)\}\n]+")
_LABEL_TEXT_RE = re.compile(r"[^|\n]+")
_BARE_SUBGRAPH_LABEL_RE = re.compile(r"[^\n]+")


@dataclass
class _Cursor:
    """Stateful parser cursor over the input string."""

    src: str
    pos: int = 0

    def eof(self) -> bool:
        return self.pos >= len(self.src)

    def peek(self, s: str) -> bool:
        return self.src.startswith(s, self.pos)

    def consume(self, s: str) -> bool:
        if self.peek(s):
            self.pos += len(s)
            return True
        return False

    def match_re(self, pattern: re.Pattern[str]) -> str | None:
        m = pattern.match(self.src, self.pos)
        if m:
            self.pos = m.end()
            return m.group(0)
        return None

    def skip_ws(self) -> None:
        while True:
            m = _WHITESPACE_RE.match(self.src, self.pos)
            if m:
                self.pos = m.end()
                continue
            m = _COMMENT_RE.match(self.src, self.pos)
            if m:
                self.pos = m.end()
                continue
            break

    def skip_ws_and_newlines(self) -> None:
        while True:
            m = _WHITESPACE_RE.match(self.src, self.pos)
            if m:
                self.pos = m.end()
                continue
            m = _COMMENT_RE.match(self.src, self.pos)
            if m:
                self.pos = m.end()
                continue
            m = _NEWLINE_RE.match(self.src, self.pos)
            if m:
                self.pos = m.end()
                continue
            break

    def consume_newline(self) -> bool:
        m = _NEWLINE_RE.match(self.src, self.pos)
        if m:
            self.pos = m.end()
            return True
        return False

    def parse_direction_value(self) -> Direction:
        d = self.match_re(_DIRECTION_RE)
        if d in ("TD", "TB"):
            return Direction.TD
        if d == "LR":
            return Direction.LR
        if d == "RL":
            return Direction.RL
        if d == "BT":
            return Direction.BT
        return Direction.TD

    def try_parse_header(self) -> Direction | None:
        saved = self.pos
        self.skip_ws_and_newlines()
        if self.consume("flowchart") or self.consume("graph"):
            self.skip_ws()
            d = self.parse_direction_value()
            self.skip_ws()
            self.skip_ws()
            m = _COMMENT_RE.match(self.src, self.pos)
            if m:
                self.pos = m.end()
            self.skip_ws()
            self.consume_newline()
            return d
        self.pos = saved
        return None

    def parse_quoted_string(self) -> str:
        assert self.src[self.pos] == '"'
        self.pos += 1
        buf: list[str] = []
        while self.pos < len(self.src):
            ch = self.src[self.pos]
            if ch == '"':
                self.pos += 1
                break
            if ch == "\\" and self.pos + 1 < len(self.src):
                nxt = self.src[self.pos + 1]
                if nxt == "n":
                    buf.append("\n")
                elif nxt == '"':
                    buf.append('"')
                elif nxt == "\\":
                    buf.append("\\")
                else:
                    buf.append(nxt)
                self.pos += 2
            else:
                buf.append(ch)
                self.pos += 1
        return "".join(buf)

    def parse_node_label(self) -> str:
        self.skip_ws()
        if self.pos < len(self.src) and self.src[self.pos] == '"':
            return self.parse_quoted_string()
        label = self.match_re(_BARE_LABEL_RE)
        return (label or "").strip()

    def parse_node_shape(self) -> tuple[NodeShape, str] | None:
        if self.peek("(("):
            self.pos += 2
            label = self.parse_node_label()
            self.consume("))")
            return (NodeShape.Circle, label)
        if self.peek("(") and not self.src.startswith("((", self.pos):
            self.pos += 1
            label = self.parse_node_label()
            self.consume(")")
            return (NodeShape.Rounded, label)
        if self.peek("{"):
            self.pos += 1
            label = self.parse_node_label()
            self.consume("}")
            return (NodeShape.Diamond, label)
        if self.peek("["):
            self.pos += 1
            label = self.parse_node_label()
            self.consume("]")
            return (NodeShape.Rectangle, label)
        return None

    def parse_node_ref(self) -> Node | None:
        self.skip_ws()
        node_id = self.match_re(_NODE_ID_RE)
        if not node_id:
            return None
        shape_result = self.parse_node_shape()
        if shape_result:
            shape, label = shape_result
            return Node.new(node_id, label, shape)
        return Node.bare(node_id)

    def parse_edge_connector(self) -> EdgeType | None:
        self.skip_ws()
        for token, etype in _EDGE_PATTERNS:
            if self.peek(token):
                self.pos += len(token)
                return etype
        return None

    def try_parse_edge_label(self) -> str | None:
        self.skip_ws()
        if not self.consume("|"):
            return None
        text = self.match_re(_LABEL_TEXT_RE)
        self.consume("|")
        return (text or "").strip()

    def parse_edge_chain(self) -> list[tuple[EdgeType, str | None, Node]]:
        segments: list[tuple[EdgeType, str | None, Node]] = []
        while True:
            saved = self.pos
            etype = self.parse_edge_connector()
            if etype is None:
                self.pos = saved
                break
            label = self.try_parse_edge_label()
            node = self.parse_node_ref()
            if node is None:
                self.pos = saved
                break
            segments.append((etype, label, node))
        return segments

    def try_parse_edge_stmt(self) -> tuple[list[Node], list[Edge]] | None:
        saved = self.pos
        source = self.parse_node_ref()
        if source is None:
            self.pos = saved
            return None
        segments = self.parse_edge_chain()
        if not segments:
            self.pos = saved
            return None
        nodes: list[Node] = [source]
        edges: list[Edge] = []
        prev_id = source.id
        for etype, label, target in segments:
            e = Edge.new(prev_id, target.id, etype)
            e.label = label
            prev_id = target.id
            nodes.append(target)
            edges.append(e)
        return (nodes, edges)

    def try_parse_node_stmt(self) -> Node | None:
        saved = self.pos
        node = self.parse_node_ref()
        if node is None:
            self.pos = saved
            return None
        return node

    def try_parse_subgraph_direction(self) -> Direction | None:
        saved = self.pos
        self.skip_ws()
        if self.consume("direction"):
            self.skip_ws()
            d = self.parse_direction_value()
            self.skip_ws()
            self.consume_newline()
            return d
        self.pos = saved
        return None

    def parse_subgraph_label(self) -> str:
        self.skip_ws()
        if self.pos < len(self.src) and self.src[self.pos] == '"':
            return self.parse_quoted_string()
        label = self.match_re(_BARE_SUBGRAPH_LABEL_RE)
        return (label or "").strip()

    def at_end_keyword(self) -> bool:
        if not self.src.startswith("end", self.pos):
            return False
        after = self.pos + 3
        if after >= len(self.src):
            return True
        ch = self.src[after]
        return not (ch.isalnum() or ch in ("_", "-"))

    def parse_subgraph_block(self) -> Subgraph | None:
        saved = self.pos
        self.skip_ws()
        if not self.consume("subgraph"):
            self.pos = saved
            return None
        if self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] in ("_", "-")):
            self.pos = saved
            return None
        name = self.parse_subgraph_label()
        self.skip_ws()
        self.consume_newline()
        sg = Subgraph.new(name)
        d = self.try_parse_subgraph_direction()
        if d is not None:
            sg.direction = d
        while not self.eof():
            self.skip_ws()
            if self.at_end_keyword():
                self.pos += 3
                self.skip_ws()
                self.consume_newline()
                break
            if not self.parse_statement_into(sg.nodes, sg.edges, sg.subgraphs) and not self.consume_newline():
                self.pos += 1
        return sg

    def parse_statement_into(
        self,
        nodes: list[Node],
        edges: list[Edge],
        subgraphs: list[Subgraph],
    ) -> bool:
        self.skip_ws()
        if self.eof():
            return False

        sg = self.parse_subgraph_block()
        if sg is not None:
            subgraphs.append(sg)
            return True

        result = self.try_parse_edge_stmt()
        if result is not None:
            stmt_nodes, stmt_edges = result
            for n in stmt_nodes:
                _upsert_node(nodes, n)
            edges.extend(stmt_edges)
            self.skip_ws()
            self.consume_newline()
            return True

        node = self.try_parse_node_stmt()
        if node is not None:
            _upsert_node(nodes, node)
            self.skip_ws()
            self.consume_newline()
            return True

        return False

    def parse_graph(self) -> Graph:
        graph = Graph.new()
        direction = self.try_parse_header()
        if direction is not None:
            graph.direction = direction

        while not self.eof():
            self.skip_ws()
            if self.eof():
                break
            if self.consume_newline():
                continue
            if not self.parse_statement_into(graph.nodes, graph.edges, graph.subgraphs):
                self.pos += 1

        return graph


def _upsert_node(nodes: list[Node], node: Node) -> None:
    """First-definition-wins: insert node only if id not already present."""
    if not any(n.id == node.id for n in nodes):
        nodes.append(node)


class FlowchartParser:
    """Flowchart/graph diagram parser."""

    def parse(self, src: str) -> Graph:
        cursor = _Cursor(src=src)
        return cursor.parse_graph()
