"""
Stage 3: Aggregate per-network metric JSONs into a single results.csv
ready for analyze_pawn.py.

Reads pawn_results/metrics/*.json + pawn_results/samples.csv,
averages metrics across labels per sample, joins with input parameters.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUTPUT_DIR   = Path('pawn_results')
METRICS_DIR  = OUTPUT_DIR / 'metrics'
SAMPLES_FILE = OUTPUT_DIR / 'samples.csv'
RESULTS_FILE = OUTPUT_DIR / 'results.csv'


def main():
    records = []
    for jf in sorted(METRICS_DIR.glob('*.json')):
        with open(jf) as f:
            records.append(json.load(f))
    if not records:
        print(f"No metric files in {METRICS_DIR}")
        return

    df = pd.DataFrame(records)
    df['sample_id'] = df['sample_dir'].str.replace('sample_', '').astype(int)

    # Average numeric metrics across labels per sample
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'sample_id']
    agg = df.groupby('sample_id')[numeric_cols].mean().reset_index()

    samples = pd.read_csv(SAMPLES_FILE)
    samples['sample_id'] = samples.index
    samples = samples.rename(columns={'transitivity': 'transitivity_param'})

    merged = samples.merge(agg, on='sample_id', how='inner')
    merged.to_csv(RESULTS_FILE, index=False)
    print(f"Wrote {len(merged)} rows to {RESULTS_FILE}")

    n_failed = df['error'].notna().sum() if 'error' in df.columns else 0
    print(f"  {len(records)} metric files, {n_failed} with errors")


if __name__ == '__main__':
    main()