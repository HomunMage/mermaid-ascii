# CLAUDE.md — mermaid-ascii project instructions

## Project Overview
Mermaid flowchart syntax → Parse → Graph layout → ASCII/Unicode text output.
**Dual-language repo**: Python (prototype) + Rust (production port) in one folder.
Both languages have 1:1 matching module structure and produce identical output.

## Reference Implementation
- Reference repos cloned in `_ref/` (gitignored) — logic references
- Python code at `src/mermaid_ascii/` — the **ground truth** for architecture and module structure
- Rust code at `src/rust/` — must mirror Python's module layout exactly

## Autonomous Mode
This project runs with autonomous Claude agents. **Never ask the user for permission or clarification. Just work.**

## Workflow: Always Check Status Files First

### On every conversation start:
1. **Read `llm.plan.status`** — Understand the overall plan, current phase, and what's been verified
2. **Read `llm.working.status`** — Understand what was last worked on and next steps
3. Work on the current phase as indicated by these files

### While working:
- **Update `llm.working.status`** after completing meaningful work (finished a phase, hit a blocker, made a key decision)
- **Update `llm.plan.status`** when checking off verification items `[ ]` → `[x]` or when plan changes
- Keep both files reflecting the true current state

### Status file conventions:
- `llm.plan.status` — The master plan. Phases, verification checklists, architectural decisions. Update checkboxes as items are verified.
- `llm.working.status` — Current session state. What phase we're in, what's done, what's next, any blockers.

## Development Cycle (CRITICAL — follow every time)

**Small steps → Verify → Lint → Commit → Refactor → Commit**

1. **Implement the smallest possible step** — one function, one struct, one module
2. **Verify it works**:
   - Python: `uv run python -m pytest`
   - Rust: `cargo test`
3. **Lint & format**:
   - Python: `uv run ruff check --fix src/mermaid_ascii/ tests/ && uv run ruff format src/mermaid_ascii/ tests/`
   - Rust: `cargo fmt && cargo clippy`
4. **Git commit** — `git add -A && git commit -m "phase N: description" --no-verify`
5. **Refactor** if code smells — improve names, extract functions, simplify logic
6. **Verify again** (same as step 2)
7. **Lint & format again** (same as step 3)
8. **Git commit the refactor** — `git add -A && git commit -m "refactor: description" --no-verify`

### Error Recovery
- If something breaks and can't be fixed in 3 attempts: `git reset --hard HEAD`
- If a whole approach is wrong: `git log --oneline -10` to find a good checkpoint, then `git reset --hard <hash>`

## Verification Approach
- Python unit tests: `uv run python -m pytest`
- Rust unit + integration tests: `cargo test`
- Visual output verified by generating examples: `bash examples/gen.sh`
- Human reviews `.out.txt` files in `examples/` to confirm rendering correctness
- Do NOT use snapshot tests for rendered output — ASCII art needs human eyes

## Tech Stack

### Python (library — no CLI)
- **Python 3.12+**, package manager `uv` with `pyproject.toml`
- **Parser**: hand-rolled recursive descent
- **Graph**: `networkx` (DiGraph)
- **Testing**: `pytest`
- **Lint**: `ruff`

### Rust (library + CLI binary)
- **Rust 2024 edition**, build with `cargo` (Cargo.toml at project root)
- **Parser**: hand-rolled recursive descent (matching Python, NO pest)
- **Graph**: `petgraph` (DiGraph)
- **CLI**: `clap` (derive)
- **Testing**: `cargo test` (unit tests + integration tests for binary)
- **WASM**: `wasm-bindgen` (optional, for browser playground)
- **No other deps** besides petgraph + clap + wasm-bindgen

## Module Mapping (Python ↔ Rust)

| Python (src/mermaid_ascii/)  | Rust (src/rust/)              | Purpose                              |
|------------------------------|-------------------------------|--------------------------------------|
| `syntax/types.py`            | `syntax/types.rs`             | Enums + AST structs                  |
| `config.py`                  | `config.rs`                   | RenderConfig                         |
| `parsers/registry.py`        | `parsers/mod.rs`              | detect_type() + parse() dispatch     |
| `parsers/base.py`            | `parsers/base.rs`             | Parser protocol/trait                |
| `parsers/flowchart.py`       | `parsers/flowchart.rs`        | Recursive descent parser             |
| `layout/engine.py`           | `layout/mod.rs`               | full_layout() convenience API        |
| `layout/graph.py`            | `layout/graph.rs`             | GraphIR (networkx/petgraph wrapper)  |
| `layout/sugiyama.py`         | `layout/sugiyama.rs`          | Sugiyama algorithm                   |
| `layout/pathfinder.py`       | `layout/pathfinder.rs`        | A* edge routing + occupancy grid     |
| `layout/types.py`            | `layout/types.rs`             | Layout IR: LayoutResult, LayoutNode, RoutedEdge, Point |
| `renderers/base.py`          | `renderers/mod.rs`            | Renderer protocol/trait              |
| `renderers/ascii.py`         | `renderers/ascii.rs`          | ASCII/Unicode renderer               |
| `renderers/canvas.py`        | `renderers/canvas.rs`         | Canvas 2D char grid                  |
| `renderers/charset.py`       | `renderers/charset.rs`        | BoxChars, Arms junction merging      |
| `api.py`                     | `lib.rs`                      | render_dsl() public API              |
| *(no Python CLI)*            | `main.rs`                     | CLI entry point (Rust only)          |
| *(N/A)*                      | `wasm.rs`                     | WASM bindings (wasm-bindgen)         |

## Key Files
- `Cargo.toml` — Rust metadata, deps, build config
- `pyproject.toml` — Python metadata, deps, build config
- `src/mermaid_ascii/` — Python source (library only, no CLI)
- `src/rust/` — Rust source (lib.rs + main.rs)
- `tests/unit/` — Python unit tests (pytest)
- `tests/e2e/test_examples.py` — Python golden file tests (pytest)
- `tests/e2e/test_binary.rs` — Rust integration tests for the compiled binary
- `tests/rust/` — Rust unit test files (included from src via `#[path]`)
- `examples/` — Shared example DSL files + golden .expect files
- `scripts/` — Orchestrator/worker agent scripts
- `_ref/` — Cloned reference repos (gitignored)

## Pipeline (both languages)
```
Mermaid text → parser → AST Graph → GraphIR → Sugiyama layout → Layout IR (LayoutResult) → canvas render → text output
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

## Code Style
- Python 3.12+ with type hints
- dataclasses for AST types (match Rust structs)
- Keep it simple — no over-engineering, no premature abstraction
- Prefer clear names over comments
- Each module should have a single clear responsibility
- Three similar lines > premature abstraction
- Rust follows the Python module structure as closely as possible

## Linting & Formatting (CRITICAL — run every time before commit)

**Always run ruff before committing.** This is mandatory for every commit.

```bash
uv run ruff check --fix src/ tests/    # lint + auto-fix
uv run ruff format src/ tests/          # format
uv run ruff check src/ tests/           # verify clean (must pass with 0 errors)
```

- ruff is configured in `pyproject.toml` (line-length=120, py312, rules: E, F, I, UP, B, SIM)
- Workers MUST run ruff before every git commit
- If ruff reports errors that can't be auto-fixed, fix them manually before committing

## Agent Team Scripts (tmux-based autonomous development)

The `scripts/` directory contains a multi-worker orchestrator system that runs Claude agents in parallel via tmux. This enables infinite autonomous development until all tasks are done.

### Architecture
```
start.sh → tmux session → orchestrator.sh (window 0)
                            ├── plan_tasks() → haiku generates N parallel tasks
                            ├── spawn_worker() → worker.sh in tmux windows 1..N
                            ├── wait_for_workers() → poll _trigger_N files
                            └── collect_results() → loop or exit
```

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/start.sh` | Entry point. Creates tmux session, launches orchestrator. Usage: `bash scripts/start.sh [max_cycles] [num_workers]` |
| `scripts/orchestrator.sh` | Main loop. Plans tasks (via haiku), spawns workers in tmux windows, waits for completion via trigger files, loops until ALL_DONE. |
| `scripts/worker.sh` | One senior programmer agent. Reads CLAUDE.md + status files, executes assigned task, commits with git lock, writes trigger file when done. |
| `scripts/checkpoint.sh` | Quick git checkpoint: `bash scripts/checkpoint.sh "message"` |
| `scripts/rollback.sh` | Reset to last commit or specific hash: `bash scripts/rollback.sh [hash]` |
| `scripts/stop.sh` | Kill tmux session and clean up trigger/lock files. |

### How It Works (Infinite Loop)
1. **Orchestrator** reads `llm.plan.status` + `llm.working.status` via haiku to plan N independent tasks
2. **Workers** (sonnet) each get one task, work in parallel in separate tmux windows
3. Workers use **git lock** (`mkdir _git.lock` / `rmdir _git.lock`) to avoid commit conflicts
4. Workers write to `_trigger_N` files when done (DONE / BLOCKED / ALL_COMPLETE)
5. Orchestrator collects results, loops back to step 1
6. Stops when all phases are complete (ALL_DONE) or max cycles reached

### Usage
```bash
# Start autonomous development (3 workers, up to 50 cycles)
bash scripts/start.sh 50 3

# Monitor
tmux attach -t mermaid-ascii-py

# Stop
bash scripts/stop.sh

# Quick checkpoint
bash scripts/checkpoint.sh "before risky change"

# Rollback if things go wrong
bash scripts/rollback.sh          # reset to HEAD
bash scripts/rollback.sh abc123   # reset to specific commit
```

### Key Conventions
- Workers must NOT edit the same file simultaneously — orchestrator plans non-overlapping tasks
- Workers append to `llm.working.status` with `[W1]`, `[W2]` prefix
- Trigger file values: `DONE` (success), `BLOCKED` (stuck), `ALL_COMPLETE` (project finished)
