"""
Compute PAWN sensitivity indices from results.

Modes:
  --mode aggregated   → one PAWN analysis on results_aggregated.csv (averaged across labels)
  --mode per_label    → separate PAWN analysis per aggregation level (label)
  --mode both         → runs both (default)

Outputs:
  pawn_results/pawn_indices_aggregated.csv + heatmap
  pawn_results/pawn_indices_<label>.csv    + heatmap per label
  pawn_results/pawn_comparison.png         → side-by-side top sensitivities across labels
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


OUTPUT_DIR = Path('pawn_results')
INPUT_NAMES = ['pref_attachment', 'n_communities', 'transitivity_param']


# =============================================================================
# PAWN core
# =============================================================================

def pawn_index(X, Y, n_slices=10, statistic='median'):
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]
    if len(X) < 20 or np.std(Y) == 0:
        return np.nan

    quantiles = np.quantile(X, np.linspace(0, 1, n_slices + 1))
    ks_stats = []
    for i in range(n_slices):
        lo, hi = quantiles[i], quantiles[i + 1]
        in_slice = (X >= lo) & (X <= hi) if i == n_slices - 1 else (X >= lo) & (X < hi)
        if in_slice.sum() < 10:
            continue
        ks, _ = sp_stats.ks_2samp(Y[in_slice], Y)
        ks_stats.append(ks)

    if not ks_stats:
        return np.nan
    agg = {'median': np.median, 'max': np.max, 'mean': np.mean}
    return float(agg[statistic](ks_stats))


def bootstrap_ci(X, Y, n_slices=10, statistic='median', n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(pawn_index(X[idx], Y[idx], n_slices, statistic))
    boots = np.array([b for b in boots if np.isfinite(b)])
    if len(boots) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def compute_pawn_for_df(df, n_slices, statistic, do_bootstrap):
    excluded = set(INPUT_NAMES + ['sample_id', '_gen_time_s', 'label', 'sample_dir'])
    output_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in excluded]
    rows = []
    for inp in INPUT_NAMES:
        if inp not in df.columns:
            continue
        X = df[inp].to_numpy(dtype=float)
        for out in output_cols:
            Y = df[out].to_numpy(dtype=float)
            idx = pawn_index(X, Y, n_slices, statistic)
            row = {'input': inp, 'output': out, 'pawn_index': idx}
            if do_bootstrap:
                lo, hi = bootstrap_ci(X, Y, n_slices, statistic)
                row['ci_lo'], row['ci_hi'] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmap(indices, title, out_path):
    pivot = indices.pivot(index='input', columns='output', values='pawn_index')
    pivot = pivot.reindex([i for i in INPUT_NAMES if i in pivot.index])
    col_order = pivot.max(axis=0).sort_values(ascending=False).index
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(col_order)), 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='PAWN index (KS)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap: {out_path}")


def plot_comparison(all_indices, out_path):
    """Bar chart comparing top sensitivities across labels."""
    # For each label, get mean PAWN index per input
    rows = []
    for label, df in all_indices.items():
        for inp in INPUT_NAMES:
            sub = df[df['input'] == inp]
            rows.append({
                'label': label,
                'input': inp,
                'mean_pawn': sub['pawn_index'].mean(),
            })
    comp = pd.DataFrame(rows)

    labels = sorted(comp['label'].unique())
    x = np.arange(len(labels))
    width = 0.8 / len(INPUT_NAMES)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    for i, inp in enumerate(INPUT_NAMES):
        vals = comp[comp['input'] == inp].set_index('label').reindex(labels)['mean_pawn']
        ax.bar(x + i * width, vals, width, label=inp)
    ax.set_xticks(x + width * (len(INPUT_NAMES) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean PAWN index')
    ax.set_title('Parameter sensitivity across aggregation levels')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot: {out_path}")


# =============================================================================
# Main
# =============================================================================

def run_aggregated(n_slices, statistic, do_bootstrap):
    path = OUTPUT_DIR / 'results_aggregated.csv'
    if not path.exists():
        print(f"Not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"\n=== Aggregated ({len(df)} samples) ===")
    indices = compute_pawn_for_df(df, n_slices, statistic, do_bootstrap)
    indices.to_csv(OUTPUT_DIR / 'pawn_indices_aggregated.csv', index=False)
    plot_heatmap(indices, f'PAWN — aggregated ({statistic})', OUTPUT_DIR / 'pawn_heatmap_aggregated.png')
    print("\n  Top 10:")
    print(indices.sort_values('pawn_index', ascending=False).head(10).to_string(index=False))
    return {'aggregated': indices}


def run_per_label(n_slices, statistic, do_bootstrap):
    path = OUTPUT_DIR / 'results_per_label.csv'
    if not path.exists():
        print(f"Not found: {path}")
        return None
    df = pd.read_csv(path)
    labels = sorted(df['label'].unique())
    print(f"\n=== Per-label analysis ({len(labels)} labels) ===")

    all_indices = {}
    for label in labels:
        sub = df[df['label'] == label]
        safe_label = label.replace('/', '__')
        print(f"\n  [{label}] ({len(sub)} samples)")
        indices = compute_pawn_for_df(sub, n_slices, statistic, do_bootstrap)
        indices['label'] = label
        indices.to_csv(OUTPUT_DIR / f'pawn_indices_{safe_label}.csv', index=False)
        plot_heatmap(indices, f'PAWN — {label} ({statistic})', OUTPUT_DIR / f'pawn_heatmap_{safe_label}.png')
        all_indices[label] = indices
        top3 = indices.sort_values('pawn_index', ascending=False).head(3)
        for _, r in top3.iterrows():
            print(f"    {r['input']:25s} → {r['output']:30s}  {r['pawn_index']:.3f}")

    # Combined CSV with all labels
    combined = pd.concat(all_indices.values(), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / 'pawn_indices_all_labels.csv', index=False)

    # Comparison chart
    plot_comparison(all_indices, OUTPUT_DIR / 'pawn_comparison.png')

    return all_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['aggregated', 'per_label', 'both'], default='both')
    parser.add_argument('--n_slices', type=int, default=10)
    parser.add_argument('--statistic', choices=['median', 'max', 'mean'], default='median')
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()

    if args.mode in ('aggregated', 'both'):
        run_aggregated(args.n_slices, args.statistic, args.bootstrap)
    if args.mode in ('per_label', 'both'):
        run_per_label(args.n_slices, args.statistic, args.bootstrap)


if __name__ == '__main__':
    main()