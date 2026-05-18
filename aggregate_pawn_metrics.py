"""
Stage 3: Aggregate per-network metric JSONs into results CSVs
ready for analyze_pawn.py.

Input filename convention:
  pawn_results__networks__sample_{id}__{label}.json

Outputs:
  - results_per_label.csv   (one row per sample × label)
  - results_aggregated.csv  (one row per sample, averaged across labels)
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

OUTPUT_DIR      = Path('pawn_results')
NETWORKS_DIR    = OUTPUT_DIR / 'networks'          # folder containing the JSONs
SAMPLES_FILE    = OUTPUT_DIR / 'samples.csv'
PER_LABEL_FILE  = OUTPUT_DIR / 'results_per_label.csv'
AGGREGATED_FILE = OUTPUT_DIR / 'results_aggregated.csv'

# Matches: pawn_results__networks__sample_00248__geslacht_lft_oplniv.json
FILENAME_RE = re.compile(r'^pawn_results__networks__sample_(\d+)__(.+)\.json$')


def parse_filename(path: Path) -> tuple[int, str] | None:
    """Return (sample_id, label) from filename, or None if it doesn't match."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def main():
    # ── locate files ──────────────────────────────────────────────────────────
    # Support both: files sitting directly in OUTPUT_DIR, or in NETWORKS_DIR
    search_dirs = [NETWORKS_DIR, OUTPUT_DIR]
    json_files  = []
    for d in search_dirs:
        json_files = sorted(d.glob('pawn_results__networks__sample_*.json'))
        if json_files:
            print(f"Found {len(json_files)} metric file(s) in {d}")
            break

    if not json_files:
        print(f"No matching metric files found in {search_dirs}")
        return

    # ── build records ─────────────────────────────────────────────────────────
    records, skipped = [], 0
    for jf in json_files:
        parsed = parse_filename(jf)
        if parsed is None:
            print(f"  Skipping unrecognised filename: {jf.name}")
            skipped += 1
            continue

        sample_id, label = parsed
        with open(jf) as f:
            data = json.load(f)

        data['sample_id'] = sample_id
        data['label']     = label
        records.append(data)

    if skipped:
        print(f"  Skipped {skipped} file(s) with unrecognised names")

    if not records:
        print("No valid records to process.")
        return

    df = pd.DataFrame(records)

    # ── merge with sampling parameters ───────────────────────────────────────
    samples = pd.read_csv(SAMPLES_FILE)
    samples['sample_id'] = samples.index
    samples = samples.rename(columns={'transitivity': 'transitivity_param'})

    merged = samples.merge(df, on='sample_id', how='inner')

    # Drop rows flagged with errors
    if 'error' in merged.columns:
        n_err  = merged['error'].notna().sum()
        merged = merged[merged['error'].isna()].drop(columns=['error'])
        if n_err:
            print(f"  Dropped {n_err} row(s) with errors")

    n_labels  = merged['label'].nunique()
    n_samples = merged['sample_id'].nunique()

    # ── per-label output ──────────────────────────────────────────────────────
    merged.to_csv(PER_LABEL_FILE, index=False)
    print(f"Wrote {len(merged)} rows "
          f"({n_labels} labels × {n_samples} samples) → {PER_LABEL_FILE}")

    # ── aggregated output (mean across labels per sample) ─────────────────────
    input_cols   = ['sample_id'] + [c for c in samples.columns if c != 'sample_id']
    numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns
                    if c not in input_cols]

    agg = merged.groupby('sample_id')[numeric_cols].mean().reset_index()
    agg = samples.merge(agg, on='sample_id', how='inner')
    agg.to_csv(AGGREGATED_FILE, index=False)
    print(f"Wrote {len(agg)} rows (averaged across {n_labels} labels) → {AGGREGATED_FILE}")


if __name__ == '__main__':
    main()