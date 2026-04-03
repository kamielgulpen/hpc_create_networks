# Create the script
cat > tmux_submit.sh << 'EOF'
#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate

max_parallel=10
running=0
mkdir -p logs

echo "Starting 100 tasks at $(date)"

for task_id in {0..99}; do
    python run_task.py --task_id ${task_id} \
        > logs/task_${task_id}.out \
        2> logs/task_${task_${task_id}.err &
    
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
echo "Results: $(ls logs/task_*.out 2>/dev/null | wc -l)/100 tasks finished"
EOF

chmod +x run_all_tasks.sh

# Run it in tmux
tmux new -s mytasks
./run_all_tasks.sh
# Ctrl+B, D to detach