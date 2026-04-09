#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

<<<<<<< HEAD
max_parallel=3
=======
# Configuration
max_parallel=3
cores_per_task=8  # Adjust based on your total cores

# Set Numba threads to avoid oversubscription
export NUMBA_NUM_THREADS=${cores_per_task}
export OMP_NUM_THREADS=${cores_per_task}

>>>>>>> 6a5d4e1c914d6bd26773d23e4749186aa9fc5443
running=0
n_tasks=$(python seeding_experiments_optimized.py --list_tasks)
mkdir -p logs

echo "Starting ${n_tasks} tasks at $(date)"
echo "Running ${max_parallel} tasks in parallel, ${cores_per_task} cores each"

for task_id in $(seq 0 $((n_tasks - 1))); do
    PYTHONUNBUFFERED=1 python seeding_experiments_optimized.py --task_id ${task_id} \
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