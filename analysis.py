"""
Quick health check / summary of everything currently in the data lake.

Walks data_lake/ (as resolved by data_lake.ROOT) and reports, for each
piece of the pipeline that exists so far:

  - samples: how many LHS samples, parameter ranges
  - aggregation_levels: how many levels, population/interaction row counts
    and totals (eyeball that aggregation didn't lose anyone), whether
    mixing features have been computed
  - runs: how many distinct runs exist
  - networks: how many networks, grouped by aggregation level and layer,
    node/edge count stats, whether nodes.parquet exists yet
  - infection_events: how many networks have simulation results
  - disk usage: size of each top-level folder, so you can see where space
    is actually going before it becomes a problem

Safe to run at any point in the pipeline -- every section reports "not
found yet" for pieces that don't exist rather than erroring.

Usage:
    python analyze_data_lake.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import data_lake as dl


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def analyze_samples() -> None:
    _section("Samples")
    if not dl.samples_exist():
        print("  not found yet")
        return
    df = dl.read_samples()
    print(f"  {len(df)} samples")
    for col in df.columns:
        if col == "sample_id":
            continue
        print(f"  {col}: min={df[col].min():.4g} max={df[col].max():.4g} mean={df[col].mean():.4g}")


def analyze_aggregation_levels() -> pd.DataFrame | None:
    _section("Aggregation levels")
    agg_dir = dl.ROOT / "aggregation_levels"
    if not agg_dir.exists():
        print("  not found yet")
        return None

    levels = sorted(p for p in agg_dir.iterdir() if p.is_dir())
    print(f"  {len(levels)} aggregation levels")

    rows = []
    for level_dir in levels:
        agg_level_id = level_dir.name
        pop_path = level_dir / "population.parquet"
        n_groups, total_pop = None, None
        if pop_path.exists():
            pop = pd.read_parquet(pop_path)
            n_groups, total_pop = len(pop), pop["n"].sum()

        layer_paths = sorted(level_dir.glob("interactions_*.parquet"))
        layer_names = [lp.stem[len("interactions_"):] for lp in layer_paths]
        has_features = (level_dir / "mixing_features.parquet").exists()

        rows.append({
            "agg_level_id": agg_level_id,
            "n_groups": n_groups,
            "total_population": total_pop,
            "n_layers": len(layer_paths),
            "has_mixing_features": has_features,
        })

        print(f"  {agg_level_id}: {n_groups} groups, {total_pop} individuals, "
              f"layers={layer_names}, features={'yes' if has_features else 'no'}")

    return pd.DataFrame(rows)


def analyze_runs() -> None:
    _section("Runs")
    runs_dir = dl.ROOT / "runs"
    if not runs_dir.exists():
        print("  not found yet")
        return
    run_ids = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    print(f"  {len(run_ids)} runs")


def analyze_networks() -> pd.DataFrame | None:
    _section("Networks")
    net_dir = dl.ROOT / "networks"
    if not net_dir.exists():
        print("  not found yet")
        return None

    network_ids = sorted(p.name for p in net_dir.iterdir() if p.is_dir())
    print(f"  {len(network_ids)} networks")
    if not network_ids:
        return None

    rows = []
    for network_id in network_ids:
        n_dir = net_dir / network_id
        meta_path = n_dir / "meta.json"
        stats_path = n_dir / "network_stats.parquet"
        nodes_path = n_dir / "nodes.parquet"

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        stats = {}
        if stats_path.exists():
            stats_df = pd.read_parquet(stats_path)
            stats = dict(zip(stats_df["stat_name"], stats_df["stat_value"]))

        rows.append({
            "network_id": network_id,
            "agg_level_id": meta.get("agg_level_id"),
            "layer": meta.get("layer"),
            "n_nodes": stats.get("n_nodes"),
            "n_edges": stats.get("n_edges"),
            "gen_time_s": stats.get("gen_time_s"),
            "has_nodes_file": nodes_path.exists(),
        })

    df = pd.DataFrame(rows)

    print("  by aggregation level:")
    for agg_level_id, group in df.groupby("agg_level_id", dropna=False):
        layers = sorted(l for l in group["layer"].unique() if l is not None)
        print(f"    {agg_level_id}: {len(group)} networks, layers={layers}")

    if df["n_nodes"].notna().any():
        print(f"  n_nodes: min={df['n_nodes'].min():.0f} "
              f"max={df['n_nodes'].max():.0f} mean={df['n_nodes'].mean():.0f}")
    if df["n_edges"].notna().any():
        print(f"  n_edges: min={df['n_edges'].min():.0f} "
              f"max={df['n_edges'].max():.0f} mean={df['n_edges'].mean():.0f}")
    if df["gen_time_s"].notna().any():
        print(f"  gen_time_s: total={df['gen_time_s'].sum():.1f}s "
              f"mean={df['gen_time_s'].mean():.2f}s")

    n_with_nodes = int(df["has_nodes_file"].sum())
    print(f"  {n_with_nodes}/{len(df)} networks have a nodes.parquet written")

    return df


def analyze_infection_events() -> None:
    _section("Infection events")
    ie_dir = dl.ROOT / "infection_events"
    if not ie_dir.exists():
        print("  not found yet")
        return
    network_dirs = sorted(p for p in ie_dir.iterdir() if p.is_dir())
    n_files = sum(len(list(p.glob("*.parquet"))) for p in network_dirs)
    print(f"  {len(network_dirs)} networks with simulation results, {n_files} threshold files total")


def analyze_disk_usage() -> None:
    _section("Disk usage")
    if not dl.ROOT.exists():
        print(f"  {dl.ROOT} does not exist yet")
        return
    print(f"  lake root: {dl.ROOT}")
    total = 0
    for sub in sorted(dl.ROOT.iterdir()):
        size = _dir_size(sub) if sub.is_dir() else sub.stat().st_size
        total += size
        print(f"    {sub.name}: {_fmt_bytes(size)}")
    print(f"  total: {_fmt_bytes(total)}")


def show_table_heads(n: int = 5) -> None:
    """
    Print df.head(n) for one representative file of each table *kind* in
    the lake -- not every single network's files, since with hundreds of
    networks that's hundreds of near-identical dumps. Notes how many
    similar files exist alongside the example shown.
    """
    _section(f"Table previews (head of {n})")

    def _preview(path: Path, label: str, sibling_count: int | None = None) -> None:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"\n--- {label} ---\n  could not read {path}: {e}")
            return
        note = f" (showing {path.parent.name}, {sibling_count} similar files exist)" if sibling_count else ""
        print(f"\n--- {label}{note} ---")
        print(df.head(n).to_string(index=False))

    # samples -- single file, always show it in full
    if dl.samples_exist():
        _preview(dl.ROOT / "samples.parquet", "samples.parquet")
    else:
        print("\n--- samples.parquet --- \n  not found yet")

    # aggregation levels -- population, mixing_features, one interaction layer, nodes
    agg_dir = dl.ROOT / "aggregation_levels"
    if agg_dir.exists():
        levels = sorted(p for p in agg_dir.iterdir() if p.is_dir())
        if levels:
            first = levels[0]
            _preview(first / "population.parquet", "aggregation_levels/*/population.parquet", len(levels))
            _preview(first / "mixing_features.parquet", "aggregation_levels/*/mixing_features.parquet", len(levels))

            layer_files = sorted(first.glob("interactions_*.parquet"))
            if layer_files:
                _preview(layer_files[0], "aggregation_levels/*/interactions_*.parquet", len(levels) * len(layer_files))

            agg_nodes_files = [p for p in (l / "nodes.parquet" for l in levels) if p.exists()]
            if agg_nodes_files:
                _preview(agg_nodes_files[0], "aggregation_levels/*/nodes.parquet (level-keyed)", len(agg_nodes_files))
    else:
        print("\n--- aggregation_levels/* --- \n  not found yet")

    # networks -- network_stats, nodes (if network-keyed)
    net_dir = dl.ROOT / "networks"
    if net_dir.exists():
        networks = sorted(p for p in net_dir.iterdir() if p.is_dir())
        if networks:
            first = networks[0]
            _preview(first / "network_stats.parquet", "networks/*/network_stats.parquet", len(networks))
            if (first / "nodes.parquet").exists():
                net_nodes_files = [p for p in (n / "nodes.parquet" for n in networks) if p.exists()]
                _preview(first / "nodes.parquet", "networks/*/nodes.parquet (network-keyed)", len(net_nodes_files))
    else:
        print("\n--- networks/* --- \n  not found yet")

    # infection events -- one threshold file
    ie_dir = dl.ROOT / "infection_events"
    if ie_dir.exists():
        ie_files = sorted(ie_dir.rglob("*.parquet"))
        if ie_files:
            _preview(ie_files[0], "infection_events/*/*.parquet", len(ie_files))
    else:
        print("\n--- infection_events/* --- \n  not found yet")


def _read_all_network_meta() -> pd.DataFrame:
    net_dir = dl.ROOT / "networks"
    if not net_dir.exists():
        return pd.DataFrame()
    rows = []
    for network_dir in sorted(p for p in net_dir.iterdir() if p.is_dir()):
        meta_path = network_dir / "meta.json"
        if meta_path.exists():
            rows.append(json.loads(meta_path.read_text()))
    return pd.DataFrame(rows)


def _read_all_network_stats() -> pd.DataFrame:
    """Long-format network_stats across every network, pivoted wide: one row per network_id."""
    net_dir = dl.ROOT / "networks"
    if not net_dir.exists():
        return pd.DataFrame()
    frames = []
    for network_dir in sorted(p for p in net_dir.iterdir() if p.is_dir()):
        stats_path = network_dir / "network_stats.parquet"
        if stats_path.exists():
            df = pd.read_parquet(stats_path)
            df["network_id"] = network_dir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    wide = long_df.pivot(index="network_id", columns="stat_name", values="stat_value").reset_index()
    wide.columns.name = None
    return wide


def _read_all_runs() -> pd.DataFrame:
    runs_dir = dl.ROOT / "runs"
    if not runs_dir.exists():
        return pd.DataFrame()
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        row = {"run_id": meta["run_id"], "seed": meta.get("seed")}
        row.update(meta.get("sampled_params", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def _read_all_mixing_features() -> pd.DataFrame:
    """Long-format mixing_features across every aggregation level, pivoted wide: one row per agg_level_id."""
    agg_dir = dl.ROOT / "aggregation_levels"
    if not agg_dir.exists():
        return pd.DataFrame()
    frames = []
    for level_dir in sorted(p for p in agg_dir.iterdir() if p.is_dir()):
        feat_path = level_dir / "mixing_features.parquet"
        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            df["agg_level_id"] = level_dir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    wide = long_df.pivot(index="agg_level_id", columns="feature_name", values="feature_value").reset_index()
    wide.columns.name = None
    return wide


def build_networks_analysis_table(save: bool = True) -> pd.DataFrame:
    """
    Join everything scoped to a network into one flat, wide table:
    network identity (agg_level_id, layer, run_id) + topological features
    (network_stats, pivoted wide) + that network's run's sampled PAWN
    parameters (runs) + that aggregation level's mixing-pattern features
    (mixing_features, pivoted wide, prefixed 'mix_' to avoid clashing with
    topology stat names).

    This is a derived, regenerable view for analysis -- e.g. correlating
    topology against PAWN parameters or mixing-pattern features -- not a
    new source of truth. Safe to call at any point in the pipeline: missing
    pieces (no mixing features computed yet, no runs registered yet) just
    mean fewer columns, not an error.

    Saved to data_lake/analysis_tables/networks_full.parquet if save=True
    -- purely a convenience cache, always safe to delete and regenerate.
    """
    meta = _read_all_network_meta()
    if meta.empty:
        print("No networks found yet -- nothing to merge.")
        return pd.DataFrame()

    stats = _read_all_network_stats()
    runs = _read_all_runs()
    mixing = _read_all_mixing_features()

    table = meta
    if not stats.empty:
        table = table.merge(stats, on="network_id", how="left")
    if not runs.empty:
        table = table.merge(runs, on="run_id", how="left", suffixes=("", "_run"))
    if not mixing.empty:
        mixing = mixing.rename(columns={c: f"mix_{c}" for c in mixing.columns if c != "agg_level_id"})
        table = table.merge(mixing, on="agg_level_id", how="left")

    if save:
        out_path = dl.ROOT / "analysis_tables" / "networks_full.parquet"
        dl._write_parquet(out_path, table)
        print(f"  saved: {out_path}")

    return table


def analyze_merged_table() -> pd.DataFrame:
    _section("Merged analysis table (networks + stats + runs + mixing features)")
    table = build_networks_analysis_table(save=True)
    if table.empty:
        return table
    print(f"  shape: {table.shape[0]} rows x {table.shape[1]} columns")
    print(f"  columns: {list(table.columns)}")
    print(table.head().to_string(index=False))

    print(table["community_density_by_size_inter_mean"])
    return table


def main():
    print(f"Data lake root: {dl.ROOT}")
    analyze_samples()
    analyze_aggregation_levels()
    analyze_runs()
    analyze_networks()
    analyze_infection_events()
    analyze_disk_usage()
    show_table_heads()
    analyze_merged_table()


if __name__ == "__main__":
    main()