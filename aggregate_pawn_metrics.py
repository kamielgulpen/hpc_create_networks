"""
Stage 3: Aggregate per-network metric JSONs into results CSVs
ready for analyze_pawn.py.

Outputs:
  - results_per_label.csv   (one row per sample × label — for per-aggregation-level PAWN)
  - results_aggregated.csv  (one row per sample, averaged across labels)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUTPUT_DIR   = Path('pawn_results')
METRICS_DIR  = OUTPUT_DIR / 'metrics'
SAMPLES_FILE = OUTPUT_DIR / 'samples.csv'
PER_LABEL_FILE  = OUTPUT_DIR / 'results_per_label.csv'
AGGREGATED_FILE = OUTPUT_DIR / 'results_aggregated.csv'


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

    samples = pd.read_csv(SAMPLES_FILE)
    samples['sample_id'] = samples.index
    samples = samples.rename(columns={'transitivity': 'transitivity_param'})

    merged = samples.merge(df, on='sample_id', how='inner')

    # Drop rows with errors
    if 'error' in merged.columns:
        n_err = merged['error'].notna().sum()
        merged = merged[merged['error'].isna()].drop(columns=['error'])
        print(f"  Dropped {n_err} rows with errors")

    n_labels  = merged['label'].nunique()
    n_samples = merged['sample_id'].nunique()

    # --- Per-label (one row per sample × label) ---
    merged.to_csv(PER_LABEL_FILE, index=False)
    print(f"Wrote {len(merged)} rows ({n_labels} labels × {n_samples} samples) to {PER_LABEL_FILE}")

    # --- Aggregated (one row per sample, mean across labels) ---
    input_cols = ['sample_id'] + [c for c in samples.columns if c != 'sample_id']
    numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns
                    if c not in input_cols]
    agg = merged.groupby('sample_id')[numeric_cols].mean().reset_index()
    agg = samples.merge(agg, on='sample_id', how='inner')
    agg.to_csv(AGGREGATED_FILE, index=False)
    print(f"Wrote {len(agg)} rows (averaged across labels) to {AGGREGATED_FILE}")


if __name__ == '__main__':
    main()