#!/bin/bash
# worker.sh — One senior programmer agent working on a specific task
# Usage: bash scripts/worker.sh <worker_id> [task_description]
# Called by orchestrator in separate tmux windows

cd "$(dirname "$0")/.." || exit 1
PROJECT_DIR="$(pwd)"
WORKER_ID="${1:-1}"
TASK_DESC="${2:-}"
TRIGGER_FILE="${PROJECT_DIR}/_trigger_${WORKER_ID}"
LOG_FILE="${PROJECT_DIR}/out/worker_${WORKER_ID}.log"
GIT_LOCK="${PROJECT_DIR}/_git.lock"

mkdir -p out

log() {
  echo "$(date '+%H:%M:%S') [W${WORKER_ID}] $1" | tee -a "$LOG_FILE"
}

log "Worker ${WORKER_ID} starting..."
[ -n "$TASK_DESC" ] && log "Task: ${TASK_DESC}"

# Build prompt
if [ -n "$TASK_DESC" ]; then
  TASK_PROMPT="YOUR ASSIGNED TASK: ${TASK_DESC}
Focus ONLY on this specific task. Do not work on other tasks.
Other senior programmers are working on other tasks in parallel — stay in your lane."
else
  TASK_PROMPT="Pick the SMALLEST next task in the current phase that has not been done yet."
fi

# Run Claude in non-interactive mode
CLAUDECODE= claude -p \
  --dangerously-skip-permissions \
  --model sonnet \
  "You are Senior Programmer #${WORKER_ID} on the mermaid-ascii-py project team.
You are one of several senior programmers working IN PARALLEL on this project.
Project dir: ${PROJECT_DIR}

FIRST: Read CLAUDE.md, then llm.plan.status, then llm.working.status.

${TASK_PROMPT}

WORKFLOW (follow strictly):
1. Read status files to understand current phase and progress
2. Implement your assigned task
3. Verify it works:
   - Python tasks: uv run python -m pytest
   - Rust tasks: cd src/rust && cargo test && cd ../..
4. MANDATORY lint before committing:
   - Python: uv run ruff check --fix src/mermaid_ascii/ tests/ && uv run ruff format src/mermaid_ascii/ tests/
   - Rust: cd src/rust && cargo fmt && cargo clippy && cd ../..
5. Git commit with lock (to avoid conflicts with other workers):
   while ! mkdir ${GIT_LOCK} 2>/dev/null; do sleep 2; done
   git add -A && git commit -m \"phase N: description\" --no-verify
   rmdir ${GIT_LOCK}
6. If code smells: refactor, verify again, lint again, commit again (use lock)
7. Update llm.working.status — APPEND your update at the bottom with [W${WORKER_ID}] prefix
8. Write DONE to: ${TRIGGER_FILE}

RULES:
- You are a senior programmer. This is a dual-language repo (Python + Rust).
- For Python: Write clean, idiomatic Python 3.12+ with type hints.
- For Rust: Write clean, idiomatic Rust 2024 edition. Match Python module structure 1:1.
- Rust source is at src/rust/src/. Python source is at src/mermaid_ascii/.
- Reference old Rust code at ../mermaid-ascii-rust/src/ for logic, but restructure to match Python layout.
- Small steps only. One function, one module, one feature at a time.
- Always verify before committing (tests + lint)
- Use the git lock (mkdir/rmdir ${GIT_LOCK}) around ALL git add/commit operations
- If something breaks and can't fix in 3 attempts: git stash, write BLOCKED to ${TRIGGER_FILE}
- Never ask questions. Make reasonable decisions and document them.
- If ALL phases in llm.plan.status are complete, write ALL_COMPLETE to ${TRIGGER_FILE}
- Do NOT edit files that another worker might be editing simultaneously
" 2>&1 | tee -a "$LOG_FILE"

log "Worker ${WORKER_ID} finished."
