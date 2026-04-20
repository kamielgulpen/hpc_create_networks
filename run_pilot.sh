#!/bin/bash
# Run pilot_grid.py for all aggregation levels, one tmux window per task.
#
# Usage:
#   bash run_pilot.sh              # start fresh session
#   tmux attach -t pilot           # reattach to monitor
#   tmux kill-session -t pilot     # kill everything

cd "$(dirname "$0")"
source .venv/bin/activate

SESSION="pilot"
mkdir -p logs/pilot

n_tasks=$(python pilot_grid.py --list_tasks)
echo "Starting ${n_tasks} pilot tasks in tmux session '${SESSION}'"

# Kill existing session if it exists
tmux kill-session -t "${SESSION}" 2>/dev/null

# Create new detached session with first task in window 0
tmux new-session -d -s "${SESSION}" -n "task_0" \
    "source .venv/bin/activate && \
     PYTHONUNBUFFERED=1 python pilot_grid.py --task_id 0 \
     2>&1 | tee logs/pilot/task_0.log; \
     echo '--- task 0 done ---'; read"

# Create one window per remaining task
for task_id in $(seq 1 $((n_tasks - 1))); do
    tmux new-window -t "${SESSION}" -n "task_${task_id}" \
        "source .venv/bin/activate && \
         PYTHONUNBUFFERED=1 python pilot_grid.py --task_id ${task_id} \
         2>&1 | tee logs/pilot/task_${task_id}.log; \
         echo '--- task ${task_id} done ---'; read"
done

# Switch back to window 0
tmux select-window -t "${SESSION}:0"

echo ""
echo "All ${n_tasks} tasks started in session '${SESSION}'."
echo ""
echo "  Attach:          tmux attach -t ${SESSION}"
echo "  Switch windows:  Ctrl-b + n / p  (next/prev)  or  Ctrl-b + 0..9"
echo "  Detach:          Ctrl-b + d"
echo "  Kill session:    tmux kill-session -t ${SESSION}"
echo ""
echo "Logs: logs/pilot/task_N.log"

tmux attach -t "${SESSION}"
