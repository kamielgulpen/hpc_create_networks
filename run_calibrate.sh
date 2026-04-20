#!/bin/bash
# Run calibrate.py for all aggregation levels, one tmux window per task.
#
# Usage:
#   bash run_calibrate.sh              # start fresh session
#   bash run_calibrate.sh --max_iter 150
#   tmux attach -t calibrate           # reattach to monitor
#   tmux kill-session -t calibrate     # kill everything

cd "$(dirname "$0")"
source .venv/bin/activate

SESSION="calibrate"
MAX_ITER=100   # default — override with --max_iter N

# Parse optional --max_iter argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p logs/calibrate

n_tasks=$(python calibrate.py --list_tasks)
echo "Starting ${n_tasks} calibration tasks (max_iter=${MAX_ITER}) in tmux session '${SESSION}'"

# Kill existing session if it exists
tmux kill-session -t "${SESSION}" 2>/dev/null

# Create new detached session with first task in window 0
tmux new-session -d -s "${SESSION}" -n "task_0" \
    "source .venv/bin/activate && \
     PYTHONUNBUFFERED=1 python calibrate.py --task_id 0 --max_iter ${MAX_ITER} \
     2>&1 | tee logs/calibrate/task_0.log; \
     echo '--- task 0 done ---'; read"

# Create one window per remaining task
for task_id in $(seq 1 $((n_tasks - 1))); do
    tmux new-window -t "${SESSION}" -n "task_${task_id}" \
        "source .venv/bin/activate && \
         PYTHONUNBUFFERED=1 python calibrate.py --task_id ${task_id} --max_iter ${MAX_ITER} \
         2>&1 | tee logs/calibrate/task_${task_id}.log; \
         echo '--- task ${task_id} done ---'; read"
done

# Switch back to window 0
tmux select-window -t "${SESSION}:0"

echo ""
echo "All ${n_tasks} tasks started (max_iter=${MAX_ITER}) in session '${SESSION}'."
echo ""
echo "  Attach:          tmux attach -t ${SESSION}"
echo "  Switch windows:  Ctrl-b + n / p  (next/prev)  or  Ctrl-b + 0..9"
echo "  Detach:          Ctrl-b + d"
echo "  Kill session:    tmux kill-session -t ${SESSION}"
echo ""
echo "Logs:    logs/calibrate/task_N.log"
echo "Results: results/calibration/"

tmux attach -t "${SESSION}"
