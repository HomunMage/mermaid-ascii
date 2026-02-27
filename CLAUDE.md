# CLAUDE.md — mermaid-ascii project instructions

## Project Overview
Mermaid flowchart syntax → Parse → Graph layout → ASCII/Unicode text output.
Written in **Homun language (.hom)** + Rust helper modules. The .hom files are compiled to .rs by `homunc`.

## Autonomous Mode
This project runs with autonomous Claude agents. **Never ask the user for permission or clarification. Just work.**

## Language Reference
- **Homun-Lang spec**: `../Homun-Lang/llm.txt` — READ THIS FIRST. It is your language reference.
- **Legacy Python/Rust reference**: `git show legacy:src/mermaid_ascii/<file>` or `git show legacy:src/rust/<file>`

## Project Structure
```
src/
  *.hom          — Core logic in Homun language (source of truth)
  dep/*.rs       — Rust helper modules (pure Rust, hand-written)
  lib.rs         — API facade (hand-written Rust)
  main.rs        — CLI entry (hand-written Rust)
tests/
  hom/*.hom      — Test files for .hom modules
  hom/*.rs       — Rust integration tests for dep/ modules
```

**IMPORTANT**: Generated `.rs` files (from `.hom`) are gitignored. `build.rs` runs `homunc` at build time to produce them. Never commit `src/types.rs`, `src/config.rs`, etc. Only commit `.hom` source files.

## Homunc Compiler
The `homunc` compiler is at: `../Homun-Lang/target/release/homunc`

Build it if needed:
```bash
cd ../Homun-Lang && cargo build --release
```

`build.rs` automatically compiles `src/*.hom` → `$OUT_DIR/*.rs` (inside `target/`) when `homunc` is in PATH. Generated `.rs` files never live in `src/` — they are build artifacts in `target/`.

To put homunc in PATH for cargo:
```bash
PATH="../Homun-Lang/target/release:$PATH" cargo build
PATH="../Homun-Lang/target/release:$PATH" cargo test
```

`cargo clean` removes all generated files.

## HOW TO WRITE .hom CODE
- Read `../Homun-Lang/llm.txt` for syntax reference
- No methods/impl blocks — use free functions: `canvas_set(c, x, y, ch)` not `c.set(x, y, ch)`
- No classes — structs for data, functions for behavior
- Use pipe `|` for chaining: `list | filter(f) | map(g)`
- Import libraries: `use std`, `use heap`, `use re`, `use chars`
- Last expression in `{}` is the return value
- Use `and`/`or`/`not` — NOT `&&`/`||`/`!` (these are lex errors)
- `?` operator works for Result unwrapping

## Development Cycle (CRITICAL — follow every time)

**Write .hom → cargo build (homunc compiles to target/) → cargo test → Commit**

1. **Write/edit a .hom module** — one function or one module at a time
2. **Build** (build.rs calls homunc automatically):
   ```bash
   PATH="../Homun-Lang/target/release:$PATH" cargo build 2>&1
   ```
   This compiles .hom → .rs into `target/` then builds the project. MUST succeed.
3. **Run tests**:
   ```bash
   PATH="../Homun-Lang/target/release:$PATH" cargo test 2>&1
   ```
   All tests MUST pass.
4. **Commit** (only .hom source, never generated .rs):
   ```bash
   git add src/ tests/ && git commit -m "mermaid: description" --no-verify
   ```

### NEVER commit code that:
- Fails `cargo build` (which includes homunc compilation)
- Fails `cargo test`
- Contains generated `.rs` files in `src/` (they belong in `target/`)

### Error Recovery
- If something breaks and can't be fixed in 3 attempts: `git reset --hard HEAD`

## Module Status

| File | Status | Notes |
|------|--------|-------|
| `src/types.hom` | compiles | Direction, NodeShape, EdgeType, Node, Edge, Subgraph, Graph |
| `src/config.hom` | compiles | RenderConfig |
| `src/layout_types.hom` | compiles | Point, LayoutNode, RoutedEdge, LayoutResult |
| `src/pathfinder.hom` | compiles | A* edge routing + occupancy grid |
| `src/canvas.hom` | **FIX NEEDED** | Uses `&&` — change to `and` |
| `src/charset.hom` | **FIX NEEDED** | Parse error: RParen/Colon mismatch |
| `src/parser.hom` | **FIX NEEDED** | Uses `!` — change to `not` |
| `src/layout.hom` | **FIX NEEDED** | Semantic: undefined references to dep/layout_state.rs functions |
| `src/render.hom` | NOT STARTED | ASCII renderer 7 phases |

## Dep Modules (pure Rust helpers)
- `src/dep/graph.rs` — petgraph DiGraph wrapper
- `src/dep/grid_data.rs` — Rc<RefCell<Vec<bool>>> occupancy grid data
- `src/dep/layout_state.rs` — Mutable state types for Sugiyama layout (DegMap, NodeSet, StrList, etc.)
- `src/dep/path_state.rs` — Point + path data structures for A* routing

## Known .hom Language Gaps
1. `.hom codegen wraps all Var args in .clone()` — for Vec<T> this clones the whole Vec and mutations are lost. Use `Rc<RefCell<...>>` dep types so `.clone()` is a pointer-bump.
2. `||`, `&&`, `!` are lex errors — use `or`, `and`, `not`
3. `homunc codegen emits EnumName.Variant` (dot) instead of `EnumName::Variant` (Rust ::) — avoid creating enum values in function bodies if the output .rs won't compile; use Rust dep helpers instead.
4. Top-level variables compile to `const X: _ = ...` which fails — define constants inline or use functions.
5. Functions from dep/*.rs are unknown to homunc's semantic checker — they appear as "undefined reference" errors. Workaround: declare them with `extern` or accept the semantic warning.

## Pipeline
```
Mermaid DSL text
  → Parser (hand-rolled recursive descent, tokenizer)
  → Graph AST (nodes, edges, subgraphs, direction)
  → Sugiyama Layout (8 phases → LayoutResult)
  → ASCII Renderer (7 phases → character grid)
  → text output
```

## Mermaid Syntax Supported
```mermaid
graph TD           %% or: flowchart LR / graph BT / etc.
    A[Rectangle]   %% id + shape bracket = node definition
    B(Rounded)
    C{Diamond}
    D((Circle))
    A --> B        %% solid arrow
    B --- C        %% solid line (no arrow)
    C -.-> D       %% dotted arrow
    D ==> A        %% thick arrow
    A <--> B       %% bidirectional
    A -->|label| B %% edge with label
    subgraph Group
        X --> Y
    end
```

## Status Files (in parent repo)
- `../.claude/llm.plan.status` — Master plan with checkboxes
- `../.claude/llm.working.status` — Current working state
- `../.claude/llm.mermaid-ascii.md` — Reference for what mermaid-ascii does

## Current Phase: 4 (Layout + Renderer)
### Remaining work:
- Fix existing .hom files that don't compile with homunc (canvas, charset, parser, layout)
- layout.hom Phase 5-8 (assign_coordinates, collapse_subgraphs, route_edges, expand_compound_nodes, full_layout)
- render.hom — ASCII renderer 7 phases
- Phase 5: Wire lib.rs + main.rs, golden file tests
