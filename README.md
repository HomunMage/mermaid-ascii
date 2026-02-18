# Mermaid ASCII rust

A compiler that renders Mermaid flowchart syntax as ASCII/Unicode art.

wasm playground at https://homun.posetmage.com/mermaid-ascii/

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

Build from source (Rust):

```sh
cargo build --release
# Binary at ./target/release/mermaid-ascii
```

Or install the Python library (no CLI — library only):

```sh
pip install mermaid-ascii
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

### Python API (library usage)

```python
from mermaid_ascii.api import render_dsl

output = render_dsl("graph TD\n    A --> B")
print(output)
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
       ┌───────────────────┼────────────────────┐
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌────────────────┐
│  Flowchart   │    │  Sequence    │    │ Architecture   │
│  AST         │    │  AST         │    │ AST            │  syntax/types.py
│  (current)   │    │  (future)    │    │ (future)       │
└──────┬───────┘    └──────┬───────┘    └───────┬────────┘
       │                   │                    │
       │ graph(Sugiyama)   │ timeline(linear)   │ grid(force)
       │ layout/graph.py   │                    │
       ▼                   ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌────────────────┐
│  Sugiyama    │    │  Sequence    │    │ Architecture   │
│  Layout      │    │  Layout      │    │ Layout         │  layout/sugiyama.py
│  (current)   │    │  (future)    │    │ (future)       │
└──────┬───────┘    └──────┬───────┘    └───────┬────────┘
       │                   │                    │
       └───────────────┬───┴────────────────────┘
                       │
                       ▼
                ┌──────────────┐
                │  Layout IR   │  layout/types.py
                │ LayoutNode[] │  x, y, width, height per node
                │ RoutedEdge[] │  waypoints per edge
                └──────┬───────┘
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
mermaid_ascii/                        # Python (library, no CLI)
├── api.py                            # render_dsl() — public API
├── config.py                         # RenderConfig dataclass
├── parsers/
│   ├── registry.py                   # detect_type() → parse() dispatch
│   ├── base.py                       # Parser protocol
│   └── flowchart.py                  # recursive descent parser
├── syntax/
│   └── types.py                      # Direction, NodeShape, EdgeType + AST
├── layout/
│   ├── engine.py                     # full_layout() convenience API
│   ├── graph.py                      # GraphIR: networkx DiGraph wrapper
│   ├── sugiyama.py                   # Sugiyama algorithm (8 phases)
│   └── types.py                      # LayoutNode, LayoutResult, RoutedEdge, Point
└── renderers/
    ├── base.py                       # Renderer protocol
    ├── ascii.py                      # ASCII/Unicode renderer (7 phases)
    ├── canvas.py                     # Canvas: 2D char grid
    └── charset.py                    # BoxChars, Arms junction merging

src/rust/                             # Rust (library + CLI binary)
├── lib.rs                            # render_dsl() — public API
├── main.rs                           # CLI entry point (clap)
├── config.rs                         # RenderConfig
├── parsers/                          # (mirrors Python parsers/)
├── syntax/                           # (mirrors Python syntax/)
├── layout/                           # (mirrors Python layout/)
└── renderers/                        # (mirrors Python renderers/)
```

### Phase Boundary Contracts

Each phase transforms one representation to the next. These type contracts are the **source of truth** — both Python and Rust must conform exactly. When adding features, update the contract first, then both implementations.

#### AST (Parser → Layout)

```
Graph {
    nodes:       Node[]          # all declared nodes
    edges:       Edge[]          # all declared edges
    subgraphs:   Subgraph[]      # nested subgraph blocks
    direction:   Direction        # TD | BT | LR | RL
}

Node {
    id:     str                  # unique identifier
    label:  str                  # display text (may contain \n)
    shape:  NodeShape            # Rectangle | Rounded | Diamond | Circle
    attrs:  Attr[]               # key-value metadata (future use)
}

Edge {
    from_id:    str
    to_id:      str
    edge_type:  EdgeType         # Arrow | Line | DottedArrow | DottedLine
                                 # ThickArrow | ThickLine | BidirArrow
                                 # BidirDotted | BidirThick
    label:      str?             # optional edge label
    attrs:      Attr[]
}

Subgraph {
    id:          str
    label:       str
    nodes:       Node[]
    edges:       Edge[]
    subgraphs:   Subgraph[]      # nested
    direction:   Direction?      # optional override
}
```

#### Layout IR (Layout → Renderer)

```
LayoutResult {
    nodes:                  LayoutNode[]
    edges:                  RoutedEdge[]
    direction:              Direction
    subgraph_members:       (str, str[])[]     # (sg_name, member_ids)
    subgraph_descriptions:  {str: str}         # sg_name → description
}

LayoutNode {
    id:      str
    layer:   uint              # Sugiyama layer index
    order:   uint              # position within layer
    x:       int               # column (char coords)
    y:       int               # row (char coords)
    width:   int               # box width in chars
    height:  int               # box height in chars
    label:   str
    shape:   NodeShape
}

RoutedEdge {
    from_id:    str
    to_id:      str
    label:      str?
    edge_type:  EdgeType
    waypoints:  Point[]        # orthogonal path segments
}

Point { x: int, y: int }      # char-grid coordinates

# Internal prefixes (not visible to renderers as real nodes):
DUMMY_PREFIX    = "__dummy_"   # dummy nodes from edge splitting
COMPOUND_PREFIX = "__sg_"      # compound nodes from subgraph collapse
```

#### Renderer Contract

Both Python and Rust renderers must follow these behavioral rules:

```
Canvas:
  - Character width = Unicode scalar count (not byte length, not display width)
  - to_string(): trim trailing whitespace per line, trim trailing empty lines,
    end with single \n
  - Negative coordinates: silently skip (don't paint)

Node painting:
  - Label centering: pad = (inner_width - char_count) / 2  (integer division)
  - inner_width = box_width - 2

Edge painting:
  - Line chars: solid ─│, dotted ╌╎, thick ═║
  - Arrow chars: ► ◄ ▼ ▲ (Unicode), > < v ^ (ASCII)
  - Junction merging: Arms OR (─ + │ = ┼)
  - Label placed at midpoint waypoint, one row above

Direction transforms:
  - LR/RL: transpose x↔y and width↔height before painting
  - BT: flip vertical after painting (reverse rows, remap ▼↔▲ etc.)
  - RL: flip horizontal after painting (reverse cols, remap ►↔◄ etc.)

Sentinel values for min/max:
  - Use language maximum (sys.maxsize / i64::MAX), not arbitrary constants
```

### Dual-Language Maintenance

**Workflow: iterate fast in Python → port to Rust → compile to binary**

```
┌─────────────────────────────────────────────────────────┐
│  1. Prototype & iterate in Python                       │
│     - Fast feedback: uv run python -m pytest            │
│     - All logic changes happen here first               │
│     - Update .expect golden files if output changes     │
│                                                         │
│  2. Port changes to Rust (1:1 module match)             │
│     - Follow the module map below                       │
│     - Rust must produce identical output                 │
│     - cd src/rust && cargo test                         │
│                                                         │
│  3. Verify parity                                       │
│     - E2E tests: uv run python -m pytest tests/e2e/    │
│     - Both languages tested against same .expect files  │
│                                                         │
│  4. Ship Rust binary                                    │
│     - cargo build --release                             │
│     - Cross-platform CI builds (linux/mac/windows)      │
└─────────────────────────────────────────────────────────┘
```

**Rules for keeping both languages in sync:**

1. **Python is ground truth** — all new features start in Python
2. **Contracts first** — update the Phase Boundary Contracts above before changing either implementation
3. **Golden tests are shared** — both languages test against `examples/*.expect` files
4. **Module map is 1:1** — every Python module has exactly one Rust counterpart (see table below)
5. **Algorithm logic must match** — same variable names, same loop structures, same formulas where possible

**When adding a new diagram type** (e.g., sequence diagrams):

1. Add new parser in `parsers/sequence.py` → `parsers/sequence.rs`
2. Add new AST types in `syntax/types.py` → `syntax/types.rs`
3. Add new layout engine in `layout/sequence.py` → `layout/sequence.rs`
4. All layout engines output the same `LayoutResult` — renderers don't change
5. Update `parsers/registry.py` → `parsers/mod.rs` to dispatch the new type

### Module Map (Python ↔ Rust)

```
Python (src/mermaid_ascii/)     Rust (src/rust/)                    Graph lib
─────────────────────────────   ──────────────────────────────────  ─────────
syntax/types.py                 syntax/types.rs                     —
config.py                       config.rs                           —
parsers/registry.py             parsers/mod.rs                      —
parsers/base.py                 parsers/base.rs                     —
parsers/flowchart.py            parsers/flowchart.rs                —
layout/engine.py                layout/mod.rs                       —
layout/graph.py                 layout/graph.rs                     networkx / petgraph
layout/sugiyama.py              layout/sugiyama.rs                  networkx / petgraph
layout/types.py                 layout/types.rs                     —
renderers/base.py               renderers/mod.rs                    —
renderers/ascii.py              renderers/ascii.rs                  —
renderers/canvas.py             renderers/canvas.rs                 —
renderers/charset.py            renderers/charset.rs                —
api.py                          lib.rs                              —
(no Python CLI)                 main.rs                             — (Rust-only CLI)
```

**Architectural note — Rust AdjGraph**: Rust's `sugiyama.rs` uses a lightweight `AdjGraph` struct (string-based adjacency list) as an intermediate representation for cycle removal, layer assignment, and crossing minimization. This exists because petgraph's index-based API is less ergonomic than networkx's string-keyed API for these algorithms. Python works directly on networkx throughout. The algorithm logic is identical — AdjGraph is a Rust-specific implementation detail that doesn't affect output.

### Dependencies

**Python:**
- [networkx](https://networkx.org/) — directed graph (petgraph equivalent)
- [click](https://click.palletsprojects.com/) — CLI framework

**Rust:**
- [petgraph](https://docs.rs/petgraph/) — directed graph (networkx equivalent)
- [clap](https://docs.rs/clap/) — CLI framework (click equivalent)

### Reference

This is a 1:1 port of [mermaid-ascii-rust](https://github.com/HomunMage/mermaid-ascii-rust). Design influenced by:

- [mermaid-ascii](https://github.com/AlexanderGrooff/mermaid-ascii) (Go) — grid-based BFS layout + A* edge routing
- [ascii-mermaid](https://github.com/kais-radwan/ascii-mermaid) (TS) — extended node shapes, classDef support
- [D2](https://github.com/terrastruct/d2) (Go) — pluggable layout engine architecture

## License

MIT



## Reference
```
┌──────────────┬──────────────────────┬────────────────────┬────────────────────┬────────────────────────┐
│              │      Ours            │ Go (mermaid-ascii) │ TS (ascii-mermaid) │           D2           │
├──────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
│ Parser       │ Recursive descent    │ Regex line-by-line │ Regex line-by-line │ Custom DSL parser      │
├──────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
│ Layout       │ Sugiyama (full)      │ Grid BFS + A*      │ Grid BFS + A*      │ Dagre (Sugiyama) / ELK │
├──────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
│ Crossing Min │ Barycenter 24-pass   │ None               │ None               │ Barycenter (via Dagre) │
├──────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
│ Edge Routing │ Orthogonal waypoints │ A* pathfinding     │ A* pathfinding     │ Spline curves          │
├──────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
│ Node Shapes  │ 4                    │ 1 (rect only)      │ 13                 │ Many                   │
├──────────────┼──────────────────────┼────────────────────┼────────────────────┼────────────────────────┤
│ Target       │ ASCII/Unicode        │ ASCII/Unicode      │ ASCII/Unicode      │ SVG                    │
└──────────────┴──────────────────────┴────────────────────┴────────────────────┴────────────────────────┘
```
