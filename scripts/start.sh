#!/bin/bash
# start.sh — Start the multi-worker orchestrator in a tmux session
# Usage: bash scripts/start.sh [max_cycles] [num_workers]
# Monitor: tmux attach -t mermaid-ascii-py
# Stop: bash scripts/stop.sh

cd "$(dirname "$0")/.." || exit 1
PROJECT_DIR="$(pwd)"
MAX_CYCLES="${1:-50}"
NUM_WORKERS="${2:-3}"
SESSION="mermaid-ascii-py"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Clean up stale files
rm -f _trigger_* _task_queue _git.lock .orchestrator.pid

# Create tmux session with orchestrator in window 0
tmux new-session -d -s "$SESSION" -n "orchestrator" \
  "cd ${PROJECT_DIR} && bash scripts/orchestrator.sh ${MAX_CYCLES} ${NUM_WORKERS}; echo 'Orchestrator ended. Press enter.'; read"

echo "========================================="
echo " mermaid-ascii-py autonomous agent team started"
echo "========================================="
echo ""
echo " tmux session:  ${SESSION}"
echo " Monitor:       tmux attach -t ${SESSION}"
echo " Stop:          bash scripts/stop.sh"
echo " Max cycles:    ${MAX_CYCLES}"
echo " Workers:       ${NUM_WORKERS} (senior programmers)"
echo ""
echo " Window layout:"
echo "   0: orchestrator — task planner + coordinator"
echo "   1-${NUM_WORKERS}: workers — each in its own tmux window"
echo ""
echo " Logs:"
echo "   out/orchestrator.log"
echo "   out/worker_1.log .. out/worker_${NUM_WORKERS}.log"
echo ""
