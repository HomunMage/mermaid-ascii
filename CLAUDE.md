# Dev Rules — mermaid-ascii

## On Start — Read These First

1. `README.md` — project overview
2. `.tmp/llm.plan.status` — ticket list and current status (pick `[ ]` tickets to work on)
3. `.tmp/llm.working.log` — abstract of recent completed work
4. `.tmp/llm.working.notes` — detailed working notes (if exists)
5. Any `.tmp/llm*.md` files — design docs, references

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
  lib.rs         — API facade + full pipeline (hand-written Rust)
  main.rs        — CLI entry (hand-written Rust)
tests/
  hom/*.hom      — Test files for .hom modules
  hom/*.rs       — Rust integration tests for dep/ modules
```

**IMPORTANT**: Generated `.rs` files (from `.hom`) are gitignored. `build.rs` runs `homunc` at build time to produce them. Never commit `src/types.rs`, `src/config.rs`, etc. Only commit `.hom` source files.

## Work Cycle

### Step 1: Clean Slate
```bash
git status
# If there are uncommitted changes → git reset --hard HEAD
```

### Step 2: Pick ONE Ticket
- Read `.tmp/llm.plan.status`
- Find the first `[ ]` (unchecked) ticket
- Work on ONLY that ticket — one ticket per session

### Step 3: Implement
- Make the smallest possible change to complete the ticket
- Stay in scope — don't refactor unrelated code

### Step 4: Test
```bash
cargo build 2>&1   # homunc compiles .hom → .rs, then rustc builds
cargo test 2>&1    # All tests MUST pass
```

### Step 5: Format + Lint
```bash
cargo fmt
cargo clippy -- -D warnings
```

### Step 6: Git Commit
```bash
# Acquire lock (if multi-worker)
while ! mkdir _git.lock 2>/dev/null; do sleep 2; done

git add src/ tests/
git commit -m "mermaid: <short description>"

# Release lock
rmdir _git.lock
```

### Step 7: Update Status
1. Mark the ticket `[x]` in `.tmp/llm.plan.status`
2. Append a summary to `.tmp/llm.working.log`:
   ```
   [W{id}] <what was done> — <files changed>
   ```

## Temporary Files

- **All temp/scratch work MUST go in `./.tmp/`** (project-local), never `/tmp/` or other system dirs.
- `.tmp/` is gitignored — safe for intermediate outputs, downloads, generated files, build artifacts, etc.
- Create `.tmp/` if it doesn't exist before writing to it.

## Rules

- **ONE ticket per session.** Small steps. Do not batch multiple tickets.
- **Never ask questions.** Make reasonable decisions and document them in the commit message.
- **Stay in your assigned scope.** Don't touch files outside your task boundary.
- **If stuck after 3 attempts:** `git stash`, write BLOCKED to the trigger file, stop.
- **All tests must pass** before committing.
- **NEVER commit generated `.rs` files** in `src/` (they belong in `target/`).
- **Commit messages**: `mermaid: <verb> <what>`

### Error Recovery
- If something breaks and can't be fixed in 3 attempts: `git reset --hard HEAD`

## Homunc Compiler

Install the latest `homunc` from GitHub releases:
```bash
wget -q https://github.com/HomunMage/Homun-Lang/releases/latest/download/homunc-linux-x86_64 -O ~/bin/homunc
chmod +x ~/bin/homunc
```

`build.rs` automatically compiles `src/*.hom` → `$OUT_DIR/*.rs` (inside `target/`) when `homunc` is in PATH.

## HOW TO WRITE .hom CODE

- Read `../Homun-Lang/llm.txt` for syntax reference
- No methods/impl blocks — use free functions: `canvas_set(c, x, y, ch)` not `c.set(x, y, ch)`
- No classes — structs for data, functions for behavior
- Use pipe `|` for chaining: `list | filter(f) | map(g)`
- Use `and`/`or`/`not` — NOT `&&`/`||`/`!` (these are lex errors)
- `?` operator works for Result unwrapping

## Known .hom Language Gaps

1. `.hom codegen wraps all Var args in .clone()` — for Vec<T> mutations are lost. Use `Rc<RefCell<...>>` dep types.
2. **Nested while loop counter shadowing** — `x := x + 1` generates shadow variable → infinite loop
3. `||`, `&&`, `!` are lex errors — use `or`, `and`, `not`
4. Functions from dep/*.rs are unknown to homunc's semantic checker — appear as warnings
5. `str` in .hom = `String` in Rust

## Pipeline

```
Mermaid DSL text
  → Parser (hand-rolled recursive descent)
  → Graph AST (nodes, edges, subgraphs, direction)
  → Sugiyama Layout (cycle removal → layers → ordering → coordinates → routing)
  → ASCII Renderer (canvas + box-drawing characters)
  → text output
```

## Key Files

| File | Role |
|------|------|
| `src/lib.rs` | **MAIN FILE** — entire pipeline: parser, layout, routing, rendering |
| `src/dep/graph.rs` | petgraph DiGraph wrapper (hand-written Rust) |
| `src/dep/layout_state.rs` | Rc<RefCell<...>> mutable state types |
| `src/dep/path_state.rs` | Rc<RefCell<...>> types for A* |
| `src/pathfinder.hom` | A* pathfinding (works — single-level loops + Rc types) |
| `src/canvas.hom` | Canvas/CharSet/BoxChars type definitions + pure functions |
| `src/main.rs` | CLI entry point (clap-based) |

## Build & Run

```bash
cargo build    # homunc compiles .hom → .rs, then rustc builds
cargo test     # 35 tests pass
cargo run -- input.txt           # Unicode output
cargo run -- --ascii input.txt   # ASCII output
printf 'graph TD\n  A-->B' | cargo run  # stdin
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
