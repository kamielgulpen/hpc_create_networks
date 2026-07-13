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

metrics = ['detected_community_edges_inter_mean', 'detected_community_density_by_minsize_inter_mean', 'coreness_skew', 'detected_community_edges_inter_skew', 'detected_community_density_by_minsize_inter_skew', 'global_clustering', 'detected_community_density_by_degree_intra_q25', 'detected_community_density_by_minsize_intra_skew', 'num_communities', 'detected_community_density_by_size_intra_mean']


plt.scatter(df["mix_familie_homophily_coleman_mean"], df["mix_familie_hhi_overall"], c= "red")
plt.scatter(df["mix_buren_homophily_coleman_mean"], df["mix_buren_hhi_overall"], c = "blue")
plt.scatter(df["mix_werkschool_homophily_coleman_mean"], df["mix_werkschool_hhi_overall"], c = "green")

plt.show()

for i in df.columns:
    print(i)

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

# =====================================================================
# Q1: How much of the topology is fixed by the mixing-pattern constraints
#     versus filled in by the generator's residual freedom?
#
#   within-setting spread  = spread of a metric across replicate networks
#                            that share a `cc` (same constraints)  -> generator freedom
#   across-setting spread  = spread of the per-`cc` MEANS across `cc` groups
#                            (i.e. how much changing the constraints moves the metric)
#   ratio                  = within / across.  <1 -> constraints dominate (generator
#                            well-behaved);  >1 -> generator's freedom swamps the
#                            constraint effect (metric barely controlled by constraints)
#
# Assumes `df`, `metrics`, and the `cc` column already exist (defined above).
# =====================================================================


def q1_spread_table(df, cols, group_col="cc", spread="std", min_per_group=2):
    """Return a per-metric table of within-setting vs across-setting spread.

    spread : "std"  -> use standard deviation as the spread measure
             "iqr"  -> use inter-quartile range (robust to outliers/skew)

    For each metric:
      within_setting  = mean over groups of the within-group spread
                        (average generator wiggle at fixed constraints)
      across_setting  = spread of the per-group means
                        (movement due to changing constraints)
      ratio           = within_setting / across_setting
    """
    def _spread(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size < min_per_group:
            return np.nan
        if spread == "iqr":
            q75, q25 = np.percentile(v, [75, 25])
            return q75 - q25
        return v.std(ddof=1)

    rows = []
    for col in cols:
        sub = df[[group_col, col]].dropna()
        # size of each group (how many replicate networks per constraint setting)
        sizes = sub.groupby(group_col)[col].size()
        usable = sizes[sizes >= min_per_group].index
        sub = sub[sub[group_col].isin(usable)]
        if sub[group_col].nunique() < 2:
            rows.append(dict(metric=col, within_setting=np.nan,
                             across_setting=np.nan, ratio=np.nan,
                             n_groups=sub[group_col].nunique(),
                             median_group_size=np.nan))
            continue

        grouped = sub.groupby(group_col)[col]
        within_per_group = grouped.apply(_spread)          # generator freedom per setting
        group_means = grouped.mean()                       # mean topology per setting

        within = np.nanmean(within_per_group.values)       # avg generator wiggle
        across = _spread(group_means.values)               # constraint-driven movement
        ratio = within / across if (across and np.isfinite(across) and across > 0) else np.nan

        rows.append(dict(
            metric=col,
            within_setting=within,
            across_setting=across,
            ratio=ratio,
            n_groups=int(sub[group_col].nunique()),
            median_group_size=float(sizes.loc[usable].median()),
        ))

    tbl = pd.DataFrame(rows).set_index("metric")
    tbl = tbl.sort_values("ratio")
    return tbl


def plot_q1_spread(tbl, out_path=None,
                   title="Q1: generator freedom vs constraint effect"):
    """Horizontal bar chart of the within/across ratio, one bar per metric.

    ratio < 1 (left of the line) : constraints dominate the metric -> generator
                                   is well-behaved for that topological feature.
    ratio > 1 (right of the line): the generator's residual freedom swamps the
                                   constraint effect -> constraints barely control
                                   that feature.
    """
    t = tbl.dropna(subset=["ratio"]).copy()
    if t.empty:
        print("q1: no metric had >=2 usable groups; nothing to plot.")
        return None

    t = t.sort_values("ratio")
    y = np.arange(len(t))
    colors = ["#1f8a8c" if r < 1 else "#5b2a86" for r in t["ratio"]]

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.42 * len(t))),
                           layout="constrained")
    ax.barh(y, t["ratio"], color=colors, height=0.68, zorder=3)
    ax.axvline(1.0, color="0.35", lw=1.2, ls="--", zorder=2)
    ax.text(1.0, len(t) - 0.35, "  generator = constraint", fontsize=8,
            color="0.35", va="top", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=9)
    ax.set_xlabel("within-setting spread  /  across-setting spread", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.margins(y=0.02)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("0.75")
    ax.tick_params(axis="x", labelsize=8, colors="0.4")

    # annotate each bar with its ratio
    xmax = t["ratio"].max()
    for yi, r in zip(y, t["ratio"]):
        ax.text(r + xmax * 0.01, yi, f"{r:.2f}", va="center",
                fontsize=7.5, color="0.4")

    if out_path:
        fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {out_path}")
    return fig


def q1_summary(tbl):
    """One-line-per-metric verdict plus an overall headline for Q1."""
    t = tbl.dropna(subset=["ratio"])
    if t.empty:
        print("q1 summary: no usable metrics.")
        return
    constraint_dom = t.index[t["ratio"] < 1].tolist()
    generator_dom = t.index[t["ratio"] >= 1].tolist()
    print("\n--- Q1 verdict ---")
    print(f"constraints dominate (ratio<1, generator well-behaved): "
          f"{len(constraint_dom)}/{len(t)} metrics")
    if constraint_dom:
        print("   " + ", ".join(constraint_dom))
    print(f"generator's residual freedom dominates (ratio>=1): "
          f"{len(generator_dom)}/{len(t)} metrics")
    if generator_dom:
        print("   " + ", ".join(generator_dom))
    med = t["ratio"].median()
    verdict = ("constraints" if med < 1 else "the generator's residual freedom")
    print(f"median ratio = {med:.2f}  ->  overall, {verdict} controls the "
          f"emergent topology.")
    if generator_dom:
        print("   NB: metrics where the generator dominates are the ones whose "
              "within-`cc` spread you must average over in Q2 before attributing "
              "diffusion differences to the constraints.")



# ---- quick self-test on the mock data ----
if __name__ == "__main__":

    ridgeline(df, cols=metrics, hhi_col="mix_werkschool_homophily_raw", agg_col="cc",
              out_path="ridge_simple.png")
    
    # ---- Q1 run ------------------------------------------------------------
    # spread="std" for the standard reading; switch to spread="iqr" for a
    # heavy-tail-robust version (several metrics here are skew statistics).
    q1_tbl = q1_spread_table(df, cols=metrics, group_col="cc", spread="std")
    print("\n=== Q1: within- vs across-setting spread (per topological metric) ===")
    print(q1_tbl.to_string(float_format=lambda x: f"{x:.4g}"))
    q1_summary(q1_tbl)
    plot_q1_spread(q1_tbl, out_path="q1_spread.png")

