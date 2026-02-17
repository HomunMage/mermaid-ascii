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

## Architecture

Pipeline: **Mermaid text** → **PEG parser** → **AST** → **networkx IR** → **Sugiyama layout** → **edge routing** → **canvas render** → **text output**

- Parser: [parsimonious](https://github.com/erikrose/parsimonious) PEG grammar
- Graph: [networkx](https://networkx.org/) directed graph
- Layout: Sugiyama algorithm (cycle removal, layer assignment, crossing minimization, coordinate assignment with barycenter refinement)
- Rendering: 2D character canvas with Unicode box-drawing character merging

## License

MIT
