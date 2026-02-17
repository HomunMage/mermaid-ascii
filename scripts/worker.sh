#!/bin/bash
# worker.sh — Run one work cycle: implement small step → verify → commit → refactor → commit
# This is called by the orchestrator for each micro-task

cd "$(dirname "$0")/.." || exit 1
PROJECT_DIR="$(pwd)"
TRIGGER_FILE="${PROJECT_DIR}/_trigger"
LOG_FILE="${PROJECT_DIR}/out/worker.log"

# Source Rust environment
. "$HOME/.cargo/env" 2>/dev/null || true

mkdir -p out

echo "$(date '+%H:%M:%S') Worker starting..." | tee -a "$LOG_FILE"

# Run Claude in non-interactive mode for one work cycle
CLAUDECODE= claude -p \
  --dangerously-skip-permissions \
  --model sonnet \
  "You are working on the text-graph project autonomously.
Project dir: ${PROJECT_DIR}

FIRST: Read CLAUDE.md, then llm.plan.status, then llm.working.status.

WORKFLOW (follow strictly):
1. Read status files to understand current phase and progress
2. Pick the SMALLEST next task in the current phase
3. Implement it (write code)
4. Verify it works: cargo check / cargo run / cargo test as appropriate
   (Make sure to source \$HOME/.cargo/env before running cargo commands)
5. If it works: run 'git add -A && git commit -m \"phase N: description\" --no-verify'
6. If it has code smells: refactor, verify again, commit again
7. Update llm.working.status with what you did and what's next
8. If the current phase is complete, check off items in llm.plan.status
9. When done with this cycle, write DONE to file: ${PROJECT_DIR}/_trigger

RULES:
- Small steps only. One function, one module, one feature at a time.
- Always verify before committing (cargo check at minimum)
- If something breaks and you can't fix it in 3 attempts: git reset --hard HEAD then write BLOCKED to _trigger
- Never ask questions. Make reasonable decisions and document them in llm.working.status
- Generate sample output files in out/ for phases that need human verification
- When ALL phases in llm.plan.status are complete, write ALL_COMPLETE to _trigger
" 2>&1 | tee -a "$LOG_FILE"

echo "$(date '+%H:%M:%S') Worker finished." | tee -a "$LOG_FILE"
