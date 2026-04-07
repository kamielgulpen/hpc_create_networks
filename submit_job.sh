#!/bin/bash
#SBATCH --job-name=create_networks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err

# Remove set -e temporarily to see all errors
# set -e

echo "=== Script Started ==="
echo "Current dir: $(pwd)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

# Try to cd
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    echo "Changing to SLURM_SUBMIT_DIR..."
    cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cd failed"; exit 1; }
    echo "Success! Now in: $(pwd)"
else
    echo "WARNING: SLURM_SUBMIT_DIR not set, staying in $(pwd)"
fi

echo "Listing directory:"
ls -la

# Check if venv exists
if [ -f ".venv/bin/activate" ]; then
    echo "Found venv, activating..."
    source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }
    echo "Activated! Python: $(which python)"
else
    echo "ERROR: .venv/bin/activate not found"
    echo "Looking for venv:"
    find . -name "activate" -type f 2>/dev/null
    exit 1
fi

# Check if run_task.py exists
if [ -f "run_task.py" ]; then
    echo "Found run_task.py"
else
    echo "ERROR: run_task.py not found"
    exit 1
fi

echo "All checks passed, starting tasks..."

# Rest of the script...
max_parallel=15
running=0

for task_id in {0..199}; do
    python run_task.py --task_id ${task_id} \
        > logs/task_${task_id}.out \
        2> logs/task_${task_id}.err &

    ((running++))

    if (( task_id % 20 == 0 )); then
        echo "Started task ${task_id}"
    fi
    
    if (( running >= max_parallel )); then
        wait -n
        ((running--))
    fi
done

wait
echo "All tasks completed"