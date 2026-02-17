#!/bin/bash
# orchestrator.sh — Multi-worker orchestrator running in tmux
# Spawns parallel workers in separate tmux windows, monitors completion via trigger files
# Usage: bash scripts/orchestrator.sh [max_cycles] [num_workers]

cd "$(dirname "$0")/.." || exit 1
PROJECT_DIR="$(pwd)"
LOG_FILE="${PROJECT_DIR}/out/orchestrator.log"
MAX_CYCLES="${1:-50}"
NUM_WORKERS="${2:-2}"
SESSION="mermaid-ascii-py"
CYCLE=0

mkdir -p out

log() {
  echo "$(date '+%H:%M:%S') [ORCH] $1" | tee -a "$LOG_FILE"
}

# Task planning: read current phase and generate task assignments
plan_tasks() {
  # Read current status and generate N tasks for workers
  # Returns tasks as lines in _task_queue
  CLAUDECODE= claude -p \
    --dangerously-skip-permissions \
    --model haiku \
    "You are the task planner for the mermaid-ascii-py project.
Project dir: ${PROJECT_DIR}

Read these files:
1. ${PROJECT_DIR}/llm.plan.status
2. ${PROJECT_DIR}/llm.working.status

Based on the current phase and progress, generate EXACTLY ${NUM_WORKERS} independent, non-overlapping tasks
that workers can do IN PARALLEL without conflicting with each other.

IMPORTANT RULES:
- Tasks must work on DIFFERENT files/modules (no two workers editing the same file)
- Each task should be small and self-contained
- Tasks should advance the current phase
- If there's only 1 task left, write just 1 task and put SINGLE on the second line

Output format (ONLY output this, nothing else):
TASK1: <one-line description of task 1>
TASK2: <one-line description of task 2>

If ALL phases are complete, output only:
ALL_DONE
" 2>/dev/null | grep -E '^(TASK[0-9]+:|ALL_DONE|SINGLE)' > "${PROJECT_DIR}/_task_queue"
}

# Spawn a worker in a new tmux window
spawn_worker() {
  local WORKER_ID=$1
  local TASK="$2"
  local WINDOW_NAME="worker-${WORKER_ID}"

  log "Spawning worker ${WORKER_ID}: ${TASK}"

  # Clean trigger file
  rm -f "${PROJECT_DIR}/_trigger_${WORKER_ID}"

  # Create new tmux window for this worker
  tmux new-window -t "${SESSION}" -n "${WINDOW_NAME}" \
    "cd ${PROJECT_DIR} && bash scripts/worker.sh ${WORKER_ID} '${TASK}'; echo 'Worker ${WORKER_ID} done. Press enter.'; read"
}

# Wait for all workers to complete (poll trigger files)
wait_for_workers() {
  local TIMEOUT=1800  # 30 min max per cycle
  local ELAPSED=0
  local ALL_DONE=false

  while [ "$ELAPSED" -lt "$TIMEOUT" ] && [ "$ALL_DONE" = "false" ]; do
    ALL_DONE=true
    for i in $(seq 1 $NUM_WORKERS); do
      if [ ! -f "${PROJECT_DIR}/_trigger_${i}" ]; then
        ALL_DONE=false
        break
      fi
    done

    if [ "$ALL_DONE" = "false" ]; then
      sleep 10
      ELAPSED=$((ELAPSED + 10))
      # Log status every 60 seconds
      if [ $((ELAPSED % 60)) -eq 0 ]; then
        local STATUS=""
        for i in $(seq 1 $NUM_WORKERS); do
          if [ -f "${PROJECT_DIR}/_trigger_${i}" ]; then
            STATUS="${STATUS} W${i}:$(cat "${PROJECT_DIR}/_trigger_${i}")"
          else
            STATUS="${STATUS} W${i}:running"
          fi
        done
        log "Status check (${ELAPSED}s):${STATUS}"
      fi
    fi
  done

  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    log "TIMEOUT: Workers didn't finish in ${TIMEOUT}s"
    return 1
  fi
  return 0
}

# Collect results from all workers
collect_results() {
  local HAS_BLOCKED=false
  local HAS_ALL_COMPLETE=false

  for i in $(seq 1 $NUM_WORKERS); do
    local TRIGGER="${PROJECT_DIR}/_trigger_${i}"
    if [ -f "$TRIGGER" ]; then
      local RESULT=$(cat "$TRIGGER")
      log "Worker ${i} result: ${RESULT}"
      case "$RESULT" in
        BLOCKED) HAS_BLOCKED=true ;;
        ALL_COMPLETE) HAS_ALL_COMPLETE=true ;;
      esac
    else
      log "Worker ${i}: no trigger file (may have crashed)"
      HAS_BLOCKED=true
    fi
    # Clean up trigger
    rm -f "$TRIGGER"
    # Close worker tmux window if still open
    tmux kill-window -t "${SESSION}:worker-${i}" 2>/dev/null
  done

  if [ "$HAS_ALL_COMPLETE" = "true" ]; then
    return 2  # Signal all complete
  elif [ "$HAS_BLOCKED" = "true" ]; then
    return 1  # Signal blocked
  fi
  return 0  # Signal done
}

log "========================================"
log "Multi-worker orchestrator started"
log "Max cycles: ${MAX_CYCLES}, Workers: ${NUM_WORKERS}"
log "========================================"

while [ "$CYCLE" -lt "$MAX_CYCLES" ]; do
  CYCLE=$((CYCLE + 1))
  log ""
  log "=== Cycle ${CYCLE}/${MAX_CYCLES} ==="

  # Clean git lock (safety)
  rmdir "${PROJECT_DIR}/_git.lock" 2>/dev/null

  # Plan tasks for this cycle
  log "Planning tasks..."
  plan_tasks

  if grep -q "ALL_DONE" "${PROJECT_DIR}/_task_queue" 2>/dev/null; then
    log "ALL PHASES COMPLETE! Project is done."
    exit 0
  fi

  # Read and spawn workers
  ACTUAL_WORKERS=0
  while IFS= read -r line; do
    TASK_NUM=$(echo "$line" | grep -oP 'TASK\K[0-9]+')
    TASK_DESC=$(echo "$line" | sed 's/^TASK[0-9]*: //')
    if [ -n "$TASK_NUM" ] && [ -n "$TASK_DESC" ]; then
      spawn_worker "$TASK_NUM" "$TASK_DESC"
      ACTUAL_WORKERS=$((ACTUAL_WORKERS + 1))
    fi
  done < "${PROJECT_DIR}/_task_queue"

  if [ "$ACTUAL_WORKERS" -eq 0 ]; then
    log "No tasks generated. Retrying in 15s..."
    sleep 15
    continue
  fi

  # Update NUM_WORKERS for wait
  NUM_WORKERS=$ACTUAL_WORKERS
  log "${ACTUAL_WORKERS} workers spawned. Waiting for completion..."

  # Wait for all workers
  wait_for_workers

  # Collect and evaluate results
  collect_results
  RESULT=$?

  case $RESULT in
    0)
      log "Cycle ${CYCLE} complete. All workers succeeded."
      ;;
    1)
      log "Some workers blocked. Waiting 30s before next cycle..."
      sleep 30
      ;;
    2)
      log "ALL PHASES COMPLETE!"
      log "Project is done. Check out/ for results."
      exit 0
      ;;
  esac

  # Brief pause between cycles
  sleep 5
done

log "Max cycles (${MAX_CYCLES}) reached."
log "Check llm.working.status for current progress."
