# Dev Rules

## On Start — Read These First

1. `README.md` — project overview, architecture, tech stack
2. `.tmp/llm.plan.status` — ticket list and current status (pick `[ ]` tickets to work on)
3. `.tmp/llm.working.log` — abstract of recent completed work
4. `.tmp/llm.working.notes` — detailed working notes (if exists, read for more context)
5. Any `.tmp/llm*md` files — design docs, API specs, references

## Project Overview

Mermaid flowchart syntax → Parse → Graph layout → ASCII/Unicode text output.
Written in **Homun language (.hom)** + Rust helper modules. The .hom files are compiled to .rs by `homunc`.

This project runs with autonomous Claude agents. **Never ask the user for permission or clarification. Just work.**

## Language Reference

- **Homun-Lang spec**: `../Homun-Lang/llm.txt` — READ THIS FIRST. It is your language reference.
- **Legacy Python/Rust reference**: `git show legacy:src/mermaid_ascii/<file>` or `git show legacy:src/rust/<file>`

## Project Structure

```
src/
  *.hom          — Core logic in Homun language (source of truth)
  graph/*.rs     — Rust helper modules (pure Rust, hand-written)
  lib.rs         — API facade + full pipeline (hand-written Rust)
  main.rs        — CLI entry (hand-written Rust)
tests/
  hom/*.hom      — Test files for .hom modules
  hom/*.rs       — Rust integration tests for graph/ modules
examples/
  *.mm.md        — Mermaid input files (test cases)
  *.expect.txt   — Expected ASCII output (golden files)
  *.expect.svg   — Expected SVG output (golden files)
  gen.sh         — Script to regenerate + verify outputs
```

**IMPORTANT**: Generated `.rs` files (from `.hom`) are gitignored. `build.rs` runs `homunc` at build time to produce them. Never commit generated `.rs` in `src/`. Only commit `.hom` source files.

## Work Cycle

### Step 1: Clean Slate
```bash
git status
# If there are uncommitted changes → git reset --hard HEAD
# Start every session with a clean working tree
```

### Step 2: Pick ONE Ticket
- Read `.tmp/llm.plan.status`
- Find the first `[ ]` (unchecked) ticket
- Work on ONLY that ticket — one ticket per session

### Step 3: Implement
- Make the smallest possible change to complete the ticket
- Stay in scope — don't refactor unrelated code
- Don't add features beyond what the ticket asks

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

git add -A
git commit -m "ticket: <short description of what was done>"

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

- **All temp/scratch work MUST go in `./.tmp/`** (project-local), never `/tmp/`.
- `.tmp/` should be in .gitignore — safe for intermediate outputs, downloads, generated files, build artifacts, etc.
- Create `.tmp/` if it doesn't exist before writing to it.

## Autonomous Agent Teams

Use `/claude-bot` to set up autonomous agent teams that work while you're away.

1. **Plan**: Run `/claude-bot` and discuss your project — Claude breaks work into tickets and designs custom runner scripts at `.tmp/claude-bot/`
2. **Launch**: `bash .tmp/claude-bot/start.sh` — workers start solving tickets in tmux
3. **Walk away**: Go eat lunch, take a break — agents work autonomously
4. **Check results**: `tmux attach -t claude-bot` or read `.tmp/llm.working.log`

See `.claude/skills/claude-bot/` for the full skill, example scripts, and planning workflow.

## Changelog

- Maintain `CHANGELOG.md` at the project root.
- Use **vMajor.Minor** format only (e.g., `v1.0`, `v1.1`, `v2.0`) — no patch level.
- Versions may jump (e.g., `v1.1` → `v1.5` or `v1.1` → `v3.0`) — a version jump signals a huge change.
- Each entry: version, date, and bullet list of what changed in short; not all details.

## Rules

- **ONE ticket per session.** Small steps. Do not batch multiple tickets.
- **Never ask questions.** Make reasonable decisions and document them in the commit message.
- **Stay in your assigned scope.** Don't touch files outside your task boundary.
- **If stuck after 3 attempts:** `git stash`, write BLOCKED to the trigger file, stop.
- **All tests must pass** before committing. If tests fail, fix them or stash and report BLOCKED.
- **Don't break existing tests.** If your change breaks unrelated tests, investigate before committing.
- **NEVER commit generated `.rs` files** in `src/` (they belong in `target/`).
- **Commit messages matter.** Use format: `ticket: <verb> <what>` (e.g., `ticket: add SVG renderer`)

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

1. `.hom codegen wraps all Var args in .clone()` — for Vec<T> mutations are lost. Use `Rc<RefCell<...>>` graph/ types.
2. **Nested while loop counter shadowing** — `x := x + 1` generates shadow variable → infinite loop
3. `||`, `&&`, `!` are lex errors — use `or`, `and`, `not`
4. Functions from graph/*.rs are unknown to homunc's semantic checker — appear as warnings
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
| `src/graph/graph.rs` | petgraph DiGraph wrapper (hand-written Rust) |
| `src/graph/layout_state.rs` | Rc<RefCell<...>> mutable state types |
| `src/graph/path_state.rs` | Rc<RefCell<...>> types for A* |
| `src/pathfinder.hom` | A* pathfinding (works — single-level loops + Rc types) |
| `src/canvas.hom` | Canvas/CharSet/BoxChars type definitions + pure functions |
| `src/main.rs` | CLI entry point (clap-based) |

## Build & Run

```bash
cargo build    # homunc compiles .hom → .rs, then rustc builds
cargo test     # All tests pass
cargo run -- input.txt           # Unicode output
cargo run -- --ascii input.txt   # ASCII output
printf 'graph TD\n  A-->B' | cargo run  # stdin
bash examples/gen.sh --check    # Verify against golden files
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
