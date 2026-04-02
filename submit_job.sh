#!/bin/bash
#SBATCH --job-name=create_networks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40       # Use all 40 CPUs on the node
#SBATCH --time=04:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err

set -e

# Get to the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Working directory: $(pwd)"
echo "Starting all tasks on $(hostname) at $(date)"
echo "Using ${SLURM_CPUS_PER_TASK} CPUs"

# Activate environment
# source .venv/bin/activate

# Create logs directory
# mkdir -p logs

# Function to limit concurrent jobs
max_parallel=33
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
        wait -n  # Wait for next job to finish
        ((running--))
    fi
done

# Wait for all remaining tasks
wait

echo "All 100 tasks completed at $(date)"

# Summary
successful=$(grep -L "error\|Error\|ERROR" logs/task_*.err 2>/dev/null | wc -l)
echo "Successfully completed: ${successful}/100 tasks"