#!/bin/bash
#SBATCH --job-name=create_networks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err

set -e

# Use SLURM's submission directory (where you ran sbatch from)
cd "${SLURM_SUBMIT_DIR}"

echo "Working directory: $(pwd)"
echo "Starting all tasks on $(hostname) at $(date)"
echo "Using ${SLURM_CPUS_PER_TASK} CPUs"

# Activate environment
source .venv/bin/activate

# Logs directory already exists (SLURM created it for main output)
mkdir -p logs

# Function to limit concurrent jobs
max_parallel=40
running=0

for task_id in {0..99}; do
    # Run task in background
    python run_task.py --task_id ${task_id} \
        > logs/task_${task_id}.out \
        2> logs/task_${task_id}.err &
    
    ((running++))
    
    echo "Started task ${task_id} (${running} running)"
    
    # When we hit the limit, wait for one to finish
    if (( running >= max_parallel )); then
        wait -n
        ((running--))
    fi
done

# Wait for all remaining tasks
wait

echo "All 100 tasks completed at $(date)"

# Summary
successful=$(ls logs/task_*.out 2>/dev/null | wc -l)
echo "Completed: ${successful}/100 tasks"