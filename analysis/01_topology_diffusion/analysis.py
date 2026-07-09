import pandas as pd
import data_lake as dl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap


df = pd.read_parquet(dl.ROOT / "analysis_tables" / "full_table.parquet")

print(df.head())

df['cc'] = df['network_id'].str.split('__').str[1]

metrics = ['avg_neighbor_degree_min', 'local_clustering_q25', 'pagerank_skew', 'density', 'detected_community_density_by_size_intra_std', 'detected_community_density_by_size_intra_q75', 'gen_time_s', 'local_clustering_skew', 'frac_in_lscc', 'detected_community_density_by_size_inter_q75', 'avg_neighbor_degree_q25', 'frac_sinks', 'detected_community_density_by_size_inter_skew', 'local_clustering_max', 'ignition_probability']

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_teal_purple",
    ["#7fd4c8", "#3fb5ac", "#1f8a8c", "#2f5e9e", "#5b2a86", "#3d1255"],
)


def _kde(v, n_grid=256, pad=0.12):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    spread = v.std(ddof=1)
    if not np.isfinite(spread) or spread <= 0:
        return None
    bw = spread * v.size ** (-1.0 / 5.0)
    lo, hi = v.min(), v.max()
    span = max(hi - lo, bw)
    grid = np.linspace(lo - pad * span, hi + pad * span, n_grid)
    u = (grid[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * u * u).sum(1) / (v.size * bw * np.sqrt(2 * np.pi))
    return grid, dens


def ridgeline(df, cols, hhi_col, agg_col, out_path=None,
              title="Distribution ridgelines by HHI", overlap=0.85):
    """Ridgeline grid: one panel per column in `cols`, one KDE ridge per group in
    `agg_col`, stacked and colored by that group's HHI (`hhi_col`).

    df       : long dataframe, one row per observation
    cols     : list of numeric columns to plot (one panel each)
    hhi_col  : column holding the HHI value (constant within each agg group)
    agg_col  : column identifying the aggregation/group
    """
    hhi = df.groupby(agg_col)[hhi_col].first()
    order = sorted(hhi.index, key=lambda a: hhi[a])          # low HHI at bottom
    vals = hhi.values[np.isfinite(hhi.values) & (hhi.values > 0)]
    if vals.size >= 2 and vals.min() != vals.max():
        norm = LogNorm(vals.min(), vals.max())
    else:
        norm = Normalize(float(vals.min()), float(vals.max()) + 1e-9)
    cmap = SEQUENTIAL

    n_agg = len(order)
    n_cols = min(3, len(cols))
    n_rows = int(np.ceil(len(cols) / n_cols))
    row_h = max(3.8, 0.42 * n_agg)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(5.2 * n_cols, row_h * n_rows),
                             layout="constrained")
    axes_flat = axes.flatten()
    step = 1.0

    for idx, col in enumerate(cols):
        ax = axes_flat[idx]
        curves = {a: _kde(df.loc[df[agg_col] == a, col].dropna()) for a in order}
        peak = max((c[1].max() for c in curves.values() if c is not None), default=0)
        if peak <= 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="0.6")
            continue
        hscale = step * (1.0 + overlap) / peak
        label_col = (idx % n_cols == 0)

        for i in reversed(range(n_agg)):            # top first -> front overpaints
            a = order[i]
            base = i * step
            c = curves[a]
            color = cmap(norm(hhi[a])) if np.isfinite(hhi[a]) else "0.6"
            if c is None:
                continue
            grid, dens = c
            y = base + dens * hscale
            ax.fill_between(grid, base, y, color=color, lw=0, zorder=i)
            ax.plot(grid, y, color="white", lw=1.3, zorder=i + 0.4)
            ax.plot(grid, y, color=color, lw=1.0, zorder=i + 0.5)
            if label_col and np.isfinite(hhi[a]):
                ax.text(-0.012, base, f"{hhi[a]:.2f}",
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=6.5, color="0.5")

        ax.set_title(col, fontsize=12, fontweight="semibold", pad=10)
        if label_col:
            ax.set_ylabel("HHI  \u2191", fontsize=8, color="0.5", labelpad=18)
        ax.set_yticks([])
        ax.set_ylim(-0.4 * step, (n_agg - 1) * step + step * (1 + overlap) + 0.2)
        ax.tick_params(axis="x", labelsize=8, colors="0.4")
        ax.margins(x=0.01)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("0.75")

    for ax in axes_flat[len(cols):]:
        ax.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes_flat), fraction=0.018, pad=0.02)
    cbar.set_label("HHI (group concentration)", fontsize=9, color="0.4")
    cbar.ax.tick_params(labelsize=7, colors="0.4")
    cbar.outline.set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    if out_path:
        fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {out_path}")
    return fig


# ---- quick self-test on the mock data ----
if __name__ == "__main__":

    ridgeline(df, cols=metrics, hhi_col="mix_werkschool_hhi_overall", agg_col="cc",
              out_path="ridge_simple.png")