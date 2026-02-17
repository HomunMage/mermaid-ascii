# text-graph

A Rust CLI that renders directed graphs as ASCII/Unicode art from a simple DSL.

```
echo '[A] --> [B] --> [C]' | text-graph

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
cargo install --path .
```

## Usage

```
text-graph [OPTIONS] [INPUT]

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
text-graph examples/diamond.txt
```

Pipe from stdin:

```sh
echo '[A] --> [B]' | text-graph
```

ASCII mode:

```
echo '[A] --> [B] --> [C]' | text-graph --ascii

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

## DSL Syntax

### Edges

```
[A] --> [B]       # directed arrow
[A] -- [B]        # undirected line
[A] ==> [B]       # thick arrow
[A] ..> [B]       # dotted arrow
[A] --> [B] --> [C]   # chained edges
```

### Edge labels

```
[Login] --> [Dashboard] { label: "success" }
[Login] --> [Error] { label: "failed" }
```

### Node shapes

```
[Rectangle]       # square brackets
(Rounded)         # parentheses
{Diamond}         # curly braces
((Circle))        # double parens
```

### Direction

```
direction: LR     # left-to-right
direction: TD     # top-down (default)
direction: BT     # bottom-to-top
direction: RL     # right-to-left
```

### Subgraphs

```
subgraph "Backend" {
  [API] --> [DB]
}
```

### Comments

```
# This is a comment
// This is also a comment
```

## Example

```
cat <<'EOF' | text-graph
[Start] --> [Build]
[Build] --> [Test]
[Test] --> [Deploy]
[Build] --> [Lint]
[Lint] --> [Deploy]
EOF

┌───────┐
│ Start │
└───┼───┘
    │
    │
    │
┌───▼───┐
│ Build │
└───┼───┘
    │
    ┼───────────┼
    │           │
┌───▼──┐    ┌───▼──┐
│ Lint │    │ Test │
└───┼──┘    └───┼──┘
    │           │
    ┼┼──────────┼
     │
┌────▼───┐
│ Deploy │
└────────┘
```

## Architecture

Pipeline: **DSL text** → **pest parser** → **AST** → **petgraph IR** → **Sugiyama layout** → **edge routing** → **canvas render** → **text output**

- Parser: [pest](https://pest.rs/) PEG grammar
- Graph: [petgraph](https://docs.rs/petgraph/) directed graph
- Layout: Sugiyama algorithm (cycle removal, layer assignment, crossing minimization, coordinate assignment)
- Rendering: 2D character canvas with box-drawing character merging
- Testing: [insta](https://insta.rs/) snapshot tests

## License

MIT
