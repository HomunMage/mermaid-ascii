# Mermaid ASCII python

A Python CLI that renders Mermaid flowchart syntax as ASCII/Unicode art.

```
echo 'graph TD
    A --> B --> C' | mermaid-ascii

┌───┐
│ A │
└─┼─┘
  │
  │
  │
┌─▼─┐
│ B │
└─┼─┘
  │
  │
  │
┌─▼─┐
│ C │
└───┘
```

## Install

```sh
pip install mermaid-ascii
```

Or with uv:

```sh
uv pip install mermaid-ascii
```

## Usage

```
mermaid-ascii [OPTIONS] [INPUT]

Arguments:
  [INPUT]  Input file (reads from stdin if omitted)

Options:
  -a, --ascii            Use plain ASCII characters instead of Unicode
  -d, --direction <DIR>  Override graph direction (LR, RL, TD, BT)
  -p, --padding <N>      Node padding [default: 1]
  -o, --output <FILE>    Write output to file instead of stdout
```

Read from file:

```sh
mermaid-ascii examples/flowchart.mm.md
```

Pipe from stdin:

```sh
echo 'graph LR
    A --> B' | mermaid-ascii
```

ASCII mode:

```
echo 'graph TD
    A --> B --> C' | mermaid-ascii --ascii

+---+
| A |
+-+-+
  |
  |
  |
+-v-+
| B |
+-+-+
  |
  |
  |
+-v-+
| C |
+---+
```

## Mermaid Syntax

Standard [Mermaid flowchart](https://mermaid.js.org/syntax/flowchart.html) syntax. Designed to align with [mermaid-ascii](https://github.com/AlexanderGrooff/mermaid-ascii) and [beautiful-mermaid](https://github.com/lukilabs/beautiful-mermaid).

### Header

```
graph TD        %% top-down (default)
flowchart LR    %% left-to-right
graph BT        %% bottom-to-top
graph RL        %% right-to-left
```

### Nodes

```
A               %% bare node (rectangle, label = "A")
A[Rectangle]    %% rectangle with label
B(Rounded)      %% rounded rectangle
C{Diamond}      %% diamond / decision
D((Circle))     %% circle
```

### Edges

```
A --> B           %% solid arrow
A --- B           %% solid line (no arrow)
A -.-> B          %% dotted arrow
A -.- B           %% dotted line
A ==> B           %% thick arrow
A === B           %% thick line
A <--> B          %% bidirectional arrow
A -->|label| B    %% edge with label
A --> B --> C     %% chained edges
```

### Subgraphs

```
subgraph Backend
    API --> DB
end
```

### Multi-line labels

```
A["Line 1\nLine 2"]
```

### Comments

```
%% This is a comment
A --> B  %% inline comment
```

## Examples

### Flowchart with shapes and labels

```
cat <<'EOF' | mermaid-ascii
graph TD
    Start[Start] --> Decision{Decision}
    Decision -->|yes| ProcessA[Process A]
    Decision -->|no| ProcessB[Process B]
    ProcessA --> End[End]
    ProcessB --> End
EOF

          ┌───────┐
          │ Start │
          └───┼───┘
              │
              │
              │
        /─────▼────\
        │ Decision │
        \─────┼────/
      yes     │        no
      ┼───────┼────────┼
      │                │
┌─────▼─────┐    ┌─────▼─────┐
│ Process A │    │ Process B │
└─────┼─────┘    └─────┼─────┘
      │                │
      ┼───────┼────────┼
              │
           ┌──▼──┐
           │ End │
           └─────┘
```

### Left-to-right pipeline

```
cat <<'EOF' | mermaid-ascii
flowchart LR
    Source --> Build --> Test --> Deploy
    Build --> Lint
    Lint --> Test
EOF
```

Generate all example outputs:

```sh
bash examples/gen.sh
```

## Compiler Design

The system is structured as a **multi-phase compiler pipeline**, where each phase transforms data from one representation to the next. No `__init__.py` files — all packages are namespace packages with explicit module imports.

```
Mermaid DSL text
  │
  ▼
┌──────────────────────────────────────────────┐
│  1. PARSER  (parsers/)                       │
│     Recursive descent → AST                  │
│     parsers/registry.py  — type detection    │
│     parsers/flowchart.py — flowchart grammar │
└──────────────┬───────────────────────────────┘
               │  AST (syntax/types.py)
               │  Graph, Node, Edge, Subgraph
               ▼
┌──────────────────────────────────────────────┐
│  2. GRAPH BUILDER  (syntax/)                 │
│     AST → networkx DiGraph                   │
│     syntax/graph.py — GraphIR topology ops   │
└──────────────┬───────────────────────────────┘
               │  GraphIR (networkx DiGraph
               │    + node/edge metadata
               │    + subgraph membership)
               ▼
┌──────────────────────────────────────────────┐
│  3. LAYOUT ENGINE  (layout/)                 │
│     Sugiyama hierarchical layout             │
│     layout/sugiyama.py — full algorithm      │
│     layout/engine.py   — convenience API     │
│     layout/types.py    — LayoutNode, Point   │
│                                              │
│     3a. Cycle removal    (Greedy-FAS)        │
│     3b. Layer assignment (longest path)      │
│     3c. Dummy node insertion                 │
│     3d. Crossing minimization (barycenter)   │
│     3e. Coordinate assignment + refinement   │
│     3f. Subgraph collapse / expand           │
│     3g. Orthogonal edge routing              │
└──────────────┬───────────────────────────────┘
               │  LayoutNode[] + RoutedEdge[]
               │  (x, y, width, height per node
               │   + waypoints per edge)
               ▼
┌──────────────────────────────────────────────┐
│  4. RENDERER  (renderers/)                   │
│     Layout → 2D character canvas → string    │
│     renderers/ascii.py   — paint pipeline    │
│     renderers/canvas.py  — 2D char grid      │
│     renderers/charset.py — junction merging  │
│                                              │
│     4a. Direction transform (LR/RL/BT)       │
│     4b. Paint subgraph borders               │
│     4c. Paint node boxes (shape-aware)       │
│     4d. Paint edges (with junction merging)  │
│     4e. Paint arrowheads + edge labels       │
│     4f. Direction post-processing (flip)     │
└──────────────┬───────────────────────────────┘
               │
               ▼
         ASCII/Unicode string
```

### Module Structure

```
mermaid_ascii/
├── api.py                  # Public API: render_dsl()
├── __main__.py             # CLI entry point (click)
├── types.py                # Shared enums: Direction, NodeShape, EdgeType
├── config.py               # RenderConfig dataclass
├── parsers/
│   ├── registry.py         # detect_type() + parse() dispatcher
│   ├── base.py             # Parser protocol
│   └── flowchart.py        # Recursive descent flowchart parser
├── syntax/
│   ├── types.py            # AST dataclasses (Graph, Node, Edge, Subgraph)
│   └── graph.py            # GraphIR (networkx DiGraph wrapper)
├── layout/
│   ├── engine.py           # Layout convenience functions
│   ├── sugiyama.py         # Sugiyama algorithm (1500+ lines)
│   └── types.py            # LayoutNode, RoutedEdge, Point
└── renderers/
    ├── base.py             # Renderer protocol
    ├── ascii.py            # ASCII/Unicode renderer
    ├── canvas.py           # 2D character grid with draw ops
    └── charset.py          # BoxChars, Arms junction merging
```

### Key Algorithms

| Phase | Algorithm | Purpose |
|-------|-----------|---------|
| Cycle Removal | Greedy Feedback Arc Set | Convert cyclic graph to DAG by reversing minimum edges |
| Layer Assignment | Longest-path fixed point | Assign each node to a layer (rank) |
| Dummy Insertion | Chain splitting | Break multi-layer edges into unit-distance segments |
| Crossing Minimization | Barycenter heuristic (24 passes) | Reorder nodes within layers to reduce edge crossings |
| Coordinate Assignment | Layer centering + barycenter refinement | Assign pixel x,y positions to nodes |
| Edge Routing | Orthogonal waypoints | Route edges through layer gaps using dummy node positions |
| Junction Merging | Arms (up/down/left/right) OR-merge | Combine overlapping box-drawing characters (e.g. `─` + `│` = `┼`) |

### Design Decisions

**Sugiyama over grid-based layout.** Reference projects like [mermaid-ascii](https://github.com/AlexanderGrooff/mermaid-ascii) (Go) and [ascii-mermaid](https://github.com/kais-radwan/ascii-mermaid) (TS) use a simpler grid-based BFS + A* pathfinding approach. We use the Sugiyama layered algorithm (same as D2's Dagre engine) which produces better results for complex graphs through crossing minimization.

**Recursive descent parser.** Our parser is a hand-rolled recursive descent parser with longest-match edge pattern matching, matching the Rust reference's PEG grammar behavior. This gives us full control over error recovery and makes it easy to extend with new diagram types.

**Protocol-based extensibility.** Parser and Renderer are defined as protocols (`parsers/base.py`, `renderers/base.py`), making it straightforward to add new diagram types (sequence, class, ER) or output formats (SVG) without modifying existing code.

**Direction via transform, not re-layout.** LR/RL directions are handled by transposing coordinates before rendering. BT is handled by flipping the canvas vertically after rendering. This avoids duplicating layout logic for each direction.

### Dependencies

- [networkx](https://networkx.org/) — directed graph (equivalent to Rust's petgraph)
- [parsimonious](https://github.com/erikrose/parsimonious) — PEG grammar utilities
- [click](https://click.palletsprojects.com/) — CLI framework

### Reference Implementations

This is a 1:1 port of [mermaid-ascii-rust](https://github.com/HomunMage/mermaid-ascii-rust). Design influenced by:

- [mermaid-ascii](https://github.com/AlexanderGrooff/mermaid-ascii) (Go) — grid-based BFS layout + A* edge routing
- [ascii-mermaid](https://github.com/kais-radwan/ascii-mermaid) (TS) — extended node shapes, classDef support
- [D2](https://github.com/terrastruct/d2) (Go) — pluggable layout engine architecture

## License

MIT



## Reference

  ┌──────────────┬──────────────────────┬──────────────────────┬────────────────────┬────────────────────┬────────────────────────┐
  │              │      Our Python      │ Rust (ground truth)  │ Go (mermaid-ascii) │ TS (ascii-mermaid) │           D2           │
  ├──────────────┼──────────────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
  │ Parser       │ Recursive descent    │ PEG (pest)           │ Regex line-by-line │ Regex line-by-line │ Custom DSL parser      │
  ├──────────────┼──────────────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
  │ Layout       │ Sugiyama (full)      │ Sugiyama (full)      │ Grid BFS + A*      │ Grid BFS + A*      │ Dagre (Sugiyama) / ELK │
  ├──────────────┼──────────────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
  │ Crossing Min │ Barycenter 24-pass   │ Barycenter 24-pass   │ None               │ None               │ Barycenter (via Dagre) │
  ├──────────────┼──────────────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
  │ Edge Routing │ Orthogonal waypoints │ Orthogonal waypoints │ A* pathfinding     │ A* pathfinding     │ Spline curves          │
  ├──────────────┼──────────────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
  │ Node Shapes  │ 4                    │ 4                    │ 1 (rect only)      │ 13                 │ Many                   │
  ├──────────────┼──────────────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
  │ Target       │ ASCII/Unicode        │ ASCII/Unicode        │ ASCII/Unicode      │ ASCII/Unicode      │ SVG                    │
  └──────────────┴──────────────────────┴──────────────────────┴────────────────────┴────────────────────┴────────────────────────┘