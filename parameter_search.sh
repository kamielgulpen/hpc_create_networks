#!/bin/bash
#
# Local parallel task runner for network matching optimization.
# Runs multiple file pairs in parallel without SLURM.
#
# Usage:
#   chmod +x run_locally.sh
#   ./run_locally.sh
#

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate

# Configuration
max_parallel=5              # Number of parallel tasks
data_folder="Data/Data/enriched/aggregated"
log_dir="logs"

# Create log directory
mkdir -p "$log_dir"

# Discover all pop/interactions pairs
declare -a pairs
declare -a pair_names
pair_count=0

for pop_file in $(find "$data_folder" -name 'pop_*.csv' -type f | sort); do
    suffix=$(basename "$pop_file" .csv | sed 's/^pop_//')
    
    # Skip if suffix has too many underscores (5+ means too aggregated)
    underscore_count=$(echo "$suffix" | grep -o '_' | wc -l)
    if (( underscore_count >= 5 )); then
        continue
    fi
    
    link_file="$data_folder/interactions_${suffix}.csv"
    
    if [[ -f "$link_file" ]]; then
        pairs+=("$pop_file|$link_file")
        pair_names+=("$suffix")
        ((pair_count++))
    else
        echo "warning: no matching interactions file for $(basename "$pop_file")"
    fi
done

if (( pair_count == 0 )); then
    echo "ERROR: No pop/interactions pairs found in $data_folder"
    exit 1
fi

echo "Found $pair_count aggregation file pairs"
for name in "${pair_names[@]}"; do
    echo "  - $name"
done

# Task counter for tracking
running=0
task_id=0
total_tasks=$pair_count

echo ""
echo "Starting $total_tasks tasks at $(date)"
echo "Max parallel workers: $max_parallel"
echo ""

for pair in "${pairs[@]}"; do
    IFS='|' read -r pop_file link_file <<< "$pair"
    suffix="${pair_names[$task_id]}"
    
    log_out="$log_dir/task_${task_id}_${suffix}.out"
    log_err="$log_dir/task_${task_id}_${suffix}.err"
    
    python run_optimization.py \
        --pop_file "$pop_file" \
        --link_file "$link_file" \
        --suffix "$suffix" \
        > "$log_out" \
        2> "$log_err" &
    
    ((running++))
    ((task_id++))
    
    # Wait if we've hit max parallel limit
    if (( running >= max_parallel )); then
        wait -n
        ((running--))
    fi
    
    # Progress update
    if (( task_id % 5 == 0 )); then
        echo "Progress: launched $task_id/$total_tasks tasks at $(date)"
    fi
done

# Wait for all remaining tasks
wait

echo ""
echo "All tasks completed at $(date)"

# Count successful completions
successful=0
for log_out in "$log_dir"/task_*.out; do
    if grep -q "Optuna study complete\|all .* trials already complete" "$log_out" 2>/dev/null; then
        ((successful++))
    fi
done

echo "Results: $successful/$total_tasks tasks appear successful"
echo "Check logs in $log_dir for details"
echo ""
echo "Summary available at: results/summary.csv"