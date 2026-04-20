"""
Pilot grid scan for calibration parameter-space exploration.

Runs a 4 x 4 x 2 grid of (n_communities, pref_attachment, transitivity_p)
for one aggregation level to identify the promising region before running
the full SA calibration.

Grid values are derived from LEVEL_BOUNDS in calibrate.py:
  - n_communities : 4 values on a log scale across [comms_min, comms_max]
  - pref_attachment: 4 values linearly across [pa_min, pa_max]
  - transitivity_p : 2 values — [trans_min, trans_max]

Total: 32 evaluations per aggregation level.

Output: results/pilot/{combo_name}_grid.csv

Usage:
    python pilot_grid.py --list_tasks        # print number of tasks
    python pilot_grid.py --task_id N         # run grid for pair N
"""

import argparse
import os
import tempfile
import time
from itertools import product
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy import stats

from asnu import generate, create_communities
from calibrate import (
    LEVEL_BOUNDS, DEFAULT_BOUNDS,
    TARGET_TRANSITIVITY, TARGET_MODULARITY, TARGET_DEGREE_SKEW,
    SCALE, RECIPROCITY_P, BRIDGE_PROB,
    ENRICHED_AGG_DIR,
    discover_enriched_pairs, nx_to_igraph, compute_metrics, loss, evaluate,
)


# =============================================================================
# Grid construction
# =============================================================================

def build_grid(bounds):
    """
    Return list of (n_communities, pref_attachment, transitivity_p) combos.

    n_communities uses geometric spacing so wide ranges (e.g. 1–50000) are
    covered evenly on a log scale. pref_attachment and transitivity_p use
    linear spacing.
    """
    comms_min = bounds['comms_min']
    comms_max = bounds['comms_max']
    pa_min    = bounds['pa_min']
    pa_max    = bounds['pa_max']
    trans_min = bounds['trans_min']
    trans_max = bounds['trans_max']

    # 4 n_communities values: geometric if range > 100, else linear
    if comms_max / max(comms_min, 1) > 10:
        comms_vals = np.geomspace(max(comms_min, 1), comms_max, 4)
    else:
        comms_vals = np.linspace(comms_min, comms_max, 4)
    comms_vals = [int(round(v)) for v in comms_vals]

    # 4 pa values: linear
    pa_vals = [float(pa_min), float(pa_max)]

    # 2 trans values: the two bounds
    trans_vals = [float(trans_min), float(trans_max)]

    return list(product(comms_vals, pa_vals, trans_vals))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4x4x2 pilot grid scan for one enriched aggregation level"
    )
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task index (0-based). Defaults to SLURM_ARRAY_TASK_ID env var.')
    parser.add_argument('--list_tasks', action='store_true',
                        help='Print number of aggregation levels and exit.')
    parser.add_argument('--output_dir', type=str, default='results/pilot')
    args = parser.parse_args()

    pairs = discover_enriched_pairs()

    if args.list_tasks:
        print(len(pairs))
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    if task_id >= len(pairs):
        print(f"Task {task_id} out of range (only {len(pairs)} pairs). Exiting.")
        return

    label, pops, links = pairs[task_id]
    combo_name = Path(label).name
    bounds     = LEVEL_BOUNDS.get(combo_name, DEFAULT_BOUNDS)
    grid       = build_grid(bounds)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{combo_name}_grid.csv'

    # Resume: skip already-evaluated combos
    done = set()
    if out_file.exists():
        df_done = pd.read_csv(out_file)
        for _, row in df_done.iterrows():
            done.add((int(row['n_communities']),
                      round(float(row['pref_attachment']), 4),
                      round(float(row['trans_param']), 4)))
        print(f"  Resuming — {len(done)}/{len(grid)} combos already done.")

    print(f"Task {task_id}/{len(pairs) - 1}: pilot grid [{label}]")
    print(f"Grid: {len(grid)} combinations  "
          f"({len(grid) - len(done)} remaining)")
    print(f"Targets — T={TARGET_TRANSITIVITY}  M={TARGET_MODULARITY}  D={TARGET_DEGREE_SKEW}")

    for i, (n_comms, pa, trans) in enumerate(grid):
        key = (n_comms, round(pa, 4), round(trans, 4))
        if key in done:
            continue

        print(f"\n[{i+1}/{len(grid)}] comms={n_comms}  pa={pa:.4f}  trans={trans:.4f}",
              flush=True)

        met_t, met_m, met_d, l, elapsed = evaluate(n_comms, pa, trans, label, pops, links)

        row = {
            'label':             label,
            'n_communities':     n_comms,
            'pref_attachment':   pa,
            'trans_param':       trans,
            'metric_transit':    round(met_t, 4),
            'modularity':        round(met_m, 4),
            'degree_skew':       round(met_d, 4),
            'loss':              round(l, 6),
            'elapsed_s':         round(elapsed, 2),
        }
        print(f"  loss={l:.4f}  T={met_t:.4f}  M={met_m:.4f}  D={met_d:.4f}  ({elapsed:.1f}s)")

        write_header = not out_file.exists()
        pd.DataFrame([row]).to_csv(out_file, mode='a', header=write_header, index=False)

    # Summary: print sorted by loss
    df = pd.read_csv(out_file)
    df_label = df[df['label'] == label].sort_values('loss')
    print(f"\n{'='*60}")
    print(f"Grid results for [{combo_name}] — sorted by loss:")
    print(df_label[['n_communities', 'pref_attachment', 'trans_param',
                     'metric_transit', 'modularity', 'degree_skew', 'loss']].to_string(index=False))
    print(f"\nBest combo:")
    best = df_label.iloc[0]
    print(f"  n_communities  = {int(best['n_communities'])}")
    print(f"  pref_attachment= {best['pref_attachment']:.4f}")
    print(f"  trans_param    = {best['trans_param']:.4f}")
    print(f"  loss           = {best['loss']:.6f}")
    print(f"\nFull results saved to {out_file}")
    print(f"Task {task_id} complete.")


if __name__ == '__main__':
    main()
