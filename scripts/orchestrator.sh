#!/bin/bash
# orchestrator.sh — Continuous agent loop
# Keeps running worker cycles until all phases are complete
# Usage: ./scripts/orchestrator.sh [max_cycles]

cd "$(dirname "$0")/.." || exit 1
PROJECT_DIR="$(pwd)"
TRIGGER_FILE="${PROJECT_DIR}/_trigger"
LOG_FILE="${PROJECT_DIR}/out/orchestrator.log"
MAX_CYCLES="${1:-50}"
CYCLE=0

mkdir -p out

echo "========================================" | tee -a "$LOG_FILE"
echo "Orchestrator started at $(date)" | tee -a "$LOG_FILE"
echo "Max cycles: ${MAX_CYCLES}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while [ "$CYCLE" -lt "$MAX_CYCLES" ]; do
  CYCLE=$((CYCLE + 1))
  echo "" | tee -a "$LOG_FILE"
  echo "--- Cycle ${CYCLE}/${MAX_CYCLES} at $(date '+%H:%M:%S') ---" | tee -a "$LOG_FILE"

  # Clean trigger file
  rm -f "$TRIGGER_FILE"

  # Run worker
  bash "${PROJECT_DIR}/scripts/worker.sh"

  # Check trigger result
  if [ -f "$TRIGGER_FILE" ]; then
    RESULT=$(cat "$TRIGGER_FILE")
    echo "Worker result: ${RESULT}" | tee -a "$LOG_FILE"

    case "$RESULT" in
      DONE)
        echo "Cycle complete. Continuing..." | tee -a "$LOG_FILE"
        ;;
      BLOCKED)
        echo "Worker is BLOCKED. Pausing for 30s then retrying..." | tee -a "$LOG_FILE"
        sleep 30
        ;;
      ALL_COMPLETE)
        echo "ALL PHASES COMPLETE!" | tee -a "$LOG_FILE"
        echo "Finished at $(date)" | tee -a "$LOG_FILE"
        exit 0
        ;;
      *)
        echo "Unknown result: ${RESULT}. Continuing..." | tee -a "$LOG_FILE"
        ;;
    esac
  else
    echo "No trigger file found. Worker may have crashed. Retrying in 10s..." | tee -a "$LOG_FILE"
    sleep 10
  fi

  # Brief pause between cycles
  sleep 5
done

echo "Max cycles (${MAX_CYCLES}) reached. Stopping." | tee -a "$LOG_FILE"
echo "Check llm.working.status for current progress." | tee -a "$LOG_FILE"
