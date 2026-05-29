#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

max_parallel=8
tasks=750
tasks=$((tasks - 1))

mkdir -p logs
echo "Starting $((tasks + 1)) tasks at $(date)"

for task_id in $(seq 0 $tasks); do
    python PAWN_analysis.py --task_id ${task_id} \
        > logs/task_${task_id}.out \
        2> logs/task_${task_id}.err &
        
    # Check actual number of running background jobs
    while [ $(jobs -r -p | wc -l) -ge $max_parallel ]; do
        sleep 0.5  # Brief pause before checking again
    done

    if (( task_id % 5 == 0 )); then
        echo "Progress: started task ${task_id}"
    fi
done

wait
echo "All tasks completed at $(date)"
echo "Results: $(ls logs/task_*.out 2>/dev/null | wc -l)/$((tasks + 1)) tasks finished"