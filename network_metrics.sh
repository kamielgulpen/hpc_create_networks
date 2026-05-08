#!/bin/bash
#
# Local parallel metrics computation for network .npz files.
# Run with tmux for long jobs:
#   tmux new -s metrics
#   ./run_metrics_locally.sh
#   (Ctrl-b d to detach, tmux attach -t metrics to reattach)
#
cd "$(dirname "$0")" || exit 1

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Configuration
max_parallel=5
base_dir="my_networks"
output_dir="network_metrics/per_file"
log_dir="logs"

mkdir -p "$log_dir"
mkdir -p "$output_dir"

# Discover all .npz files
declare -a files
file_count=0
while IFS= read -r f; do
    files+=("$f")
    ((file_count++))
done < <(find "$base_dir" -name '*.npz' -type f | sort)

if (( file_count == 0 )); then
    echo "ERROR: No .npz files found in $base_dir"
    exit 1
fi

echo "Found $file_count .npz files"
echo "Max parallel workers: $max_parallel"
echo "Starting at $(date)"
echo ""

running=0
task_id=0
total_tasks=$file_count

for f in "${files[@]}"; do
    # Build a safe log name from the relative path
    rel="${f#$base_dir/}"
    safe_name="${rel//\//__}"
    safe_name="${safe_name%.npz}"

    log_out="$log_dir/task_${task_id}_${safe_name}.out"
    log_err="$log_dir/task_${task_id}_${safe_name}.err"

    python compute_metrics.py \
        --npz_file "$f" \
        --base_dir "$base_dir" \
        --output_dir "$output_dir" \
        > "$log_out" \
        2> "$log_err" &

    ((running++))
    ((task_id++))

    if (( running >= max_parallel )); then
        wait -n
        ((running--))
    fi

    if (( task_id % 5 == 0 )); then
        echo "Progress: launched $task_id/$total_tasks at $(date)"
    fi
done

wait

echo ""
echo "All tasks completed at $(date)"

# Aggregate per-file JSON into jsonl + csv
python - <<'PYEOF'
import json
from pathlib import Path
import pandas as pd

per_file_dir = Path("network_metrics/per_file")
out_dir = Path("network_metrics")
records = []
for jf in sorted(per_file_dir.glob("*.json")):
    with open(jf) as f:
        records.append(json.load(f))

with open(out_dir / "results.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r, default=float) + "\n")

pd.DataFrame(records).to_csv(out_dir / "results.csv", index=False)

errs = sum("error" in r for r in records)
print(f"Aggregated {len(records)} records: {len(records)-errs} ok, {errs} errors")
PYEOF

echo "Results written to network_metrics/results.{jsonl,csv}"