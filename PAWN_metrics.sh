#!/bin/bash
#
# Local parallel runner for stage 2: compute network metrics from saved .npz files.
# Runs multiple files in parallel without SLURM.
#
# Usage:
#   chmod +x run_compute_metrics_locally.sh
#   tmux new -s metrics
#   ./run_compute_metrics_locally.sh
#   (Ctrl-b d to detach, tmux attach -t metrics to reattach)
#
cd "$(dirname "$0")" || exit 1

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Configuration
max_parallel=5                           # Number of parallel tasks (tune to RAM)
networks_dir="pawn_results/networks"
metrics_dir="pawn_results/metrics"
log_dir="logs_metrics"

mkdir -p "$log_dir"
mkdir -p "$metrics_dir"

# Discover all edges.npz files
declare -a files
file_count=0
while IFS= read -r f; do
    files+=("$f")
    ((file_count++))
done < <(find "$networks_dir" -name 'edges.npz' -type f | sort)

if (( file_count == 0 )); then
    echo "ERROR: No edges.npz files found in $networks_dir"
    exit 1
fi

echo "Found $file_count edges.npz files"
echo "Max parallel workers: $max_parallel"
echo "Starting at $(date)"
echo ""

running=0
task_id=0
total_tasks=$file_count
skipped=0

for f in "${files[@]}"; do
    # Build a safe identifier from the path: sample_XXXXX__<label>
    rel="${f#$networks_dir/}"            # sample_00001/label/edges.npz
    rel="${rel%/edges.npz}"              # sample_00001/label
    safe_name="${rel//\//__}"            # sample_00001__label

    # Skip if metrics already computed (matches process_one logic)
    if [[ -f "$metrics_dir/${safe_name}.json" ]]; then
        ((skipped++))
        ((task_id++))
        continue
    fi

    log_out="$log_dir/task_${task_id}_${safe_name}.out"
    log_err="$log_dir/task_${task_id}_${safe_name}.err"

    python compute_metrics.py \
        --npz_file "$f" \
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
    if (( task_id % 10 == 0 )); then
        echo "Progress: launched $task_id/$total_tasks at $(date)"
    fi
done

# Wait for all remaining tasks
wait

echo ""
echo "All tasks completed at $(date)"
echo "Skipped (already done): $skipped"

# Count successful completions: a metric file with no "error" key
successful=0
errors=0
for jf in "$metrics_dir"/*.json; do
    [[ -f "$jf" ]] || continue
    if grep -q '"error"' "$jf" 2>/dev/null; then
        ((errors++))
    else
        ((successful++))
    fi
done

echo "Results: $successful ok, $errors errors out of $((successful + errors)) metric files"
echo "Check logs in $log_dir for details"
echo ""
echo "Next: python aggregate_pawn_metrics.py"