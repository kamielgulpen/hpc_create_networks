#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

max_parallel=5
running=0
tasks=120
tasks=$((tasks - 1))

mkdir -p logs
echo "Starting $((tasks + 1)) tasks at $(date)"

for task_id in $(seq 0 $tasks); do
    python run_task.py --task_id ${task_id} \
        > logs/task_${task_id}.out \
        2> logs/task_${task_id}.err &
    ((running++))
    if (( running >= max_parallel )); then
        wait -n
        ((running--))
    fi
    if (( task_id % 5 == 0 )); then
        echo "Progress: started task ${task_id}"
    fi
done

wait
echo "All tasks completed at $(date)"
echo "Results: $(ls logs/task_*.out 2>/dev/null | wc -l)/$((tasks + 1)) tasks finished"