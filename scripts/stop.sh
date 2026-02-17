#!/bin/bash
# stop.sh — Stop the tmux orchestrator session and all workers
cd "$(dirname "$0")/.." || exit 1
SESSION="mermaid-ascii-py"

echo "Stopping tmux session: ${SESSION}"
tmux kill-session -t "$SESSION" 2>/dev/null && echo "Stopped." || echo "No session found."

# Clean up any orphan claude worker processes
pkill -f "scripts/worker.sh" 2>/dev/null || true
rm -f .orchestrator.pid _trigger_* _task_queue
rmdir _git.lock 2>/dev/null
