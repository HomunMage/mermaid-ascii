# CLAUDE.md — mermaid-ascii-py project instructions

## Project Overview
Mermaid flowchart syntax → Parse → Graph layout → ASCII/Unicode text output (Python).
This is a 1:1 port of mermaid-ascii-rust. Design aligned with mermaid-ascii (Go) and beautiful-mermaid (TS) for easy porting of their updates.

## Reference Implementation
The Rust source lives at `../mermaid-ascii-rust/src/`. Every module in this Python port maps 1:1 to a Rust source file. When in doubt, read the Rust code — it is the ground truth.

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

**Small steps → Verify → Commit → Refactor → Commit**

1. **Implement the smallest possible step** — one function, one class, one module
2. **Verify it works** — `uv run python -m pytest` minimum, `uv run python -m mermaid_ascii` if applicable
3. **Git commit** — `git add -A && git commit -m "phase N: description" --no-verify`
4. **Refactor** if code smells — improve names, extract functions, simplify logic
5. **Verify again** — `uv run python -m pytest`
6. **Git commit the refactor** — `git add -A && git commit -m "refactor: description" --no-verify`

### Error Recovery
- If something breaks and can't be fixed in 3 attempts: `git reset --hard HEAD`
- If a whole approach is wrong: `git log --oneline -10` to find a good checkpoint, then `git reset --hard <hash>`

## Verification Approach
- Unit tests for Python logic (`uv run python -m pytest`)
- Visual output verified by generating examples: `bash examples/gen.sh`
- Human reviews `.out.txt` files in `examples/` to confirm rendering correctness
- Do NOT use snapshot tests for rendered output — ASCII art needs human eyes

## Tech Stack
- **Python 3.11+**
- **Package manager**: `uv` with `pyproject.toml`
- **Parser**: PEG grammar via `parsimonious` (or hand-rolled recursive descent — match Rust `pest` grammar)
- **Graph**: `networkx` (DiGraph — equivalent to Rust `petgraph`)
- **CLI**: `click` (equivalent to Rust `clap`)
- **Testing**: `pytest`
- **No other runtime deps** beyond the above

## Module Mapping (Rust → Python)

| Rust file       | Python module                  | Purpose                              |
|-----------------|--------------------------------|--------------------------------------|
| `ast.rs`        | `mermaid_ascii/ast.py`         | AST types (dataclasses)              |
| `grammar.pest`  | `mermaid_ascii/grammar.py`     | PEG grammar string                   |
| `parser.rs`     | `mermaid_ascii/parser.py`      | Parser → AST                         |
| `graph.rs`      | `mermaid_ascii/graph.py`       | AST → networkx DiGraph (GraphIR)     |
| `layout.rs`     | `mermaid_ascii/layout.py`      | Sugiyama layout algorithm            |
| `render.rs`     | `mermaid_ascii/render.py`      | Canvas renderer                      |
| `lib.rs`        | `mermaid_ascii/__init__.py`    | Library API (render_dsl)             |
| `main.rs`       | `mermaid_ascii/__main__.py`    | CLI entry point (click)              |

## Key Files
- `pyproject.toml` — Project metadata, dependencies, build config (uv)
- `Dockerfile` — Multi-stage build matching Rust Dockerfile pattern
- `src/mermaid_ascii/` — Python source package
- `tests/` — pytest test files
- `examples/` — Example DSL files + gen script (copied from Rust)
- `scripts/` — Orchestrator/worker agent scripts (adapted from Rust)
- `_ref/` — Cloned reference repos (gitignored)

## Pipeline
```
Mermaid text → PEG parser → AST → GraphIR (networkx) → Sugiyama layout → edge routing → canvas render → text output
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
- Python 3.11+ with type hints
- dataclasses for AST types (match Rust structs)
- Keep it simple — no over-engineering, no premature abstraction
- Prefer clear names over comments
- Each module should have a single clear responsibility
- Three similar lines > premature abstraction
- Follow the Rust implementation structure as closely as possible

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
