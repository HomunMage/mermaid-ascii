# CLAUDE.md — text-graph project instructions

## Project Overview
Rust rewrite of graph-easy: DSL text input → Parse → Graph layout → ASCII/Unicode text output.

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

1. **Implement the smallest possible step** — one function, one struct, one module
2. **Verify it works** — `cargo check` minimum, `cargo test` or `cargo run` if applicable
3. **Git commit** — `git add -A && git commit -m "phase N: description" --no-verify`
4. **Refactor** if code smells — improve names, extract functions, simplify logic
5. **Verify again** — `cargo check` / `cargo test`
6. **Git commit the refactor** — `git add -A && git commit -m "refactor: description" --no-verify`
7. **Update status files** and move to next step

### Error Recovery
- If something breaks and can't be fixed in 3 attempts: `git reset --hard HEAD`
- If a whole approach is wrong: `git log --oneline -10` to find a good checkpoint, then `git reset --hard <hash>`
- Document what went wrong in `llm.working.status`

### Trigger File Protocol
When running via orchestrator:
- Write `DONE` to `_trigger` when a work cycle completes successfully
- Write `BLOCKED` to `_trigger` if stuck (orchestrator will retry)
- Write `ALL_COMPLETE` to `_trigger` when all phases are finished

## Verification Approach
- We do NOT write unit tests first — ASCII graph output is hard to verify programmatically
- Instead, generate sample output files in `out/` directory organized by phase
- Human reviews the output files to confirm correctness
- After human approval, lock down with `insta` snapshot tests

## Key Files
- `plan.old.md` — Original reference library research (Chinese)
- `_ref/` — Cloned reference repos (gitignored)
- `out/` — Generated output samples for human review
- `src/` — Rust source code
- `scripts/` — Orchestrator and helper scripts
- `_trigger` — Inter-session communication file

## Tech Stack
- Parser: `pest` (PEG grammar)
- Graph: `petgraph`
- Layout: Sugiyama algorithm (custom implementation)
- CLI: `clap`
- Testing: `insta` snapshots
- Unicode: `unicode-width`

## Reference Repos (in `_ref/`)
- `mermaid-ascii` — Closest competitor, A* edge routing
- `ascii-dag` — Sugiyama in Rust, zero-dep, Layout IR pattern
- `dagre` — Production Sugiyama algorithm (JS)
- `figurehead` — Rust Mermaid→ASCII, plugin architecture
- `d2` — DSL syntax design reference
- `svgbob` — Character rendering techniques
- `beautiful-mermaid` — TS port of mermaid-ascii

## Code Style
- Rust 2021 edition
- Keep it simple — no over-engineering, no premature abstraction
- Prefer clear names over comments
- Each module should have a single clear responsibility
- Three similar lines > premature abstraction
