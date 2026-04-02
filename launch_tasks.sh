#!/bin/bash
cd $HOME/your_project_directory  # CHANGE THIS to your actual path
source .venv/bin/activate

# Launch all 100 tasks in parallel using srun
for i in {0..99};source .venv/bin/activate
mkdir -p logs

# Launch all 100 tasks in parallel (they'll queue if cluster is busy)
for i in {0..99}; do
  srun --ntasks=1 \
       --cpus-per-task=1 \
       --time=04:00:00 \
       --output=logs/create_${i}.out \
       --error=logs/create_${i}.err \
       --job-name=task_${i} \
       python run_task.py --task_id $i &  # The & makes it parallel!
done

wait  # Wait for ALL to finish
echo "All don do
  srun --ntasks=1 \
       --cpus-per-task=1 \
       --time=04:00:00 \
       --output=logs/create_${i}.out \
       --error=logs/create_${i}.err \
       --job-name=task_${i} \
       python run_task.py --task_id $i &
done

# Wait for all background jobs to complete
wait
echo "All tasks completed"
