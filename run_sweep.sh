#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate

max_parallel=5
running=0
n_tasks=$(python seeding_experiments_memory.py --list_tasks)
mkdir -p logs

echo "Starting ${n_tasks} tasks at $(date)"

for task_id in $(seq 0 $((n_tasks - 1))); do
    PYTHONUNBUFFERED=1 python seeding_experiments_memory.py --task_id ${task_id} \
        > logs/sweep_${task_id}.out \
        2> logs/sweep_${task_id}.err &

    ((running++))

    if (( running >= max_parallel )); then
        wait -n
        ((running--))
    fi

    if (( task_id % 10 == 0 )); then
        echo "Progress: started task ${task_id}"
    fi
done

wait
echo "All tasks completed at $(date)"
echo "Results: $(ls logs/sweep_*.out 2>/dev/null | wc -l)/${n_tasks} tasks finished"
