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

Multi-phase compiler pipeline. Each phase transforms one representation to the next.

```
                          Mermaid DSL text
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Tokenizer + Parser   │  parsers/registry.py
                    │  (recursive descent)  │  parsers/flowchart.py
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
   ┌──────────────┐    ┌──────────────┐    ┌────────────────┐
   │  Flowchart   │    │  Sequence    │    │ Architecture   │
   │  AST         │    │  AST         │    │ AST            │
   │  (current)   │    │  (future)    │    │ (future)       │
   └──────┬───────┘    └──────┬───────┘    └───────┬────────┘
          │                   │                    │
          │ graph(Sugiyama)   │ timeline(linear)   │ grid(force)
          ▼                   ▼                    ▼
   ┌──────────────┐    ┌──────────────┐    ┌────────────────┐
   │  Sugiyama    │    │  Sequence    │    │ Architecture   │
   │  Layout      │    │  Layout      │    │ Layout         │
   │  (current)   │    │  (future)    │    │ (future)       │
   └──────┬───────┘    └──────┬───────┘    └───────┬────────┘
          │                   │                   │
          └─────────────┬─────┴───────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Layout IR   │  layout/types.py
                 │ LayoutNode[] │  x, y, width, height per node
                 │ RoutedEdge[] │  waypoints per edge
                 └───────┬──────┘
                         │
                   ┌─────┼─────┐
                   │           │
                   ▼           ▼
              ┌─────────┐ ┌─────────┐
              │  ASCII  │ │   SVG   │
              │Renderer │ │Renderer │
              │(current)│ │(future) │
              └────┬────┘ └─────────┘
                   │
                   ▼
            ASCII/Unicode string


  Sugiyama Layout Algorithm Phases:

  1. collapse_subgraphs()
     └─ replace subgraph members with compound node

  2. remove_cycles()             ← Greedy-FAS
     └─ reverse back-edges → DAG

  3. LayerAssignment.assign()    ← longest-path
     └─ assign each node a layer (rank)

  4. insert_dummy_nodes()
     └─ break multi-layer edges into unit segments

  5. minimise_crossings()        ← barycenter heuristic
     └─ 24-pass sweep reordering nodes within layers

  6. assign_coordinates_padded() ← layer centering
     └─ x,y positions + barycenter refinement

  7. expand_compound_nodes()
     └─ position member nodes inside compounds

  8. route_edges()               ← orthogonal waypoints
     └─ waypoints through layer gaps via dummy positions


  ASCII Render Phases:

  1. Direction transform (transpose for LR/RL)
  2. Paint compound/subgraph borders
  3. Paint node boxes (shape-aware: ┌┐└┘ ╭╮╰╯ /\ ())
  4. Paint edges (solid ─│, dotted ╌╎, thick ═║)
  5. Paint arrowheads (► ◄ ▼ ▲) + edge labels
  6. Junction merging (Arms OR: ─ + │ = ┼)
  7. Direction flip (BT→vertical, RL→horizontal)
```

### Module Map

```
mermaid_ascii/
├── api.py                  # render_dsl() — public API
├── __main__.py             # CLI (click)
├── types.py                # Direction, NodeShape, EdgeType enums
├── config.py               # RenderConfig dataclass
├── parsers/
│   ├── registry.py         # detect_type() → parse() dispatch
│   ├── base.py             # Parser protocol
│   └── flowchart.py        # recursive descent parser
├── syntax/
│   ├── types.py            # AST: Graph, Node, Edge, Subgraph
│   └── graph.py            # GraphIR: networkx DiGraph wrapper
├── layout/
│   ├── engine.py           # full_layout() convenience API
│   ├── sugiyama.py         # Sugiyama algorithm (8 phases)
│   └── types.py            # LayoutNode, RoutedEdge, Point
└── renderers/
    ├── base.py             # Renderer protocol
    ├── ascii.py            # ASCII/Unicode renderer (7 phases)
    ├── canvas.py           # Canvas: 2D char grid
    └── charset.py          # BoxChars, Arms junction merging
```

### Dependencies

- [networkx](https://networkx.org/) — directed graph (petgraph equivalent)
- [parsimonious](https://github.com/erikrose/parsimonious) — PEG grammar utilities
- [click](https://click.palletsprojects.com/) — CLI framework

### Reference

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