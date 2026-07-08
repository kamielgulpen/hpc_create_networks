"""
Build the single combined analysis table: mixing-pattern features +
network topology stats + run parameters + ignition probability from the
diffusion simulations, one row per (network, threshold).

Reads directly from the data lake (data_lake.py) -- no database, no other
setup needed. Saves to data_lake/analysis_tables/full_table.parquet.

Usage:
    python build_full_table.py
"""

from __future__ import annotations

import json

import pandas as pd

import data_lake as dl


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
    return table


def compute_ignition_stats(ignition_threshold_fraction: float = 0.5) -> pd.DataFrame:
    """
    Per (network, threshold), simulation-outcome summary from
    infection_events: what fraction of the n_simulations runs resulted in
    a large-scale cascade ("ignition") vs fizzling out early.

    ignition_threshold_fraction: a simulation counts as "ignited" if its
    final adoption fraction (nodes ever infected / total nodes) is >= this
    value. Default 0.5 -- adjust if "fully ignites" should mean something
    stricter for your model. full_cascade_probability (exactly 100%
    adoption) is included separately as an unambiguous, threshold-free
    alternative -- pick whichever matches what you actually mean.
    """
    ie_dir = dl.ROOT / "infection_events"
    if not ie_dir.exists():
        return pd.DataFrame()

    rows = []
    for network_dir in sorted(p for p in ie_dir.iterdir() if p.is_dir()):
        network_id = network_dir.name
        for threshold_file in sorted(network_dir.glob("threshold_*.parquet")):
            # threshold_idx is already in the filename -- one less column
            # to read off disk. Only threshold_value still needs reading
            # since it's not derivable from the path.
            threshold_idx = int(threshold_file.stem.removeprefix("threshold_"))
            events = pd.read_parquet(threshold_file, columns=["sim", "infection_step", "threshold_value"])
            if events.empty:
                continue

            # every node appears once per sim, so any sim's row count IS
            # total_nodes -- avoids needing the (expensive, string) node_id
            # column at all now that we've pruned it from the read above
            group_sizes = events.groupby("sim").size()
            total_nodes = int(group_sizes.iloc[0])
            final_infected = events.groupby("sim")["infection_step"].count()
            final_fraction = final_infected / total_nodes

            rows.append({
                "network_id":                  network_id,
                "threshold_idx":               threshold_idx,
                "threshold_value":             float(events["threshold_value"].iloc[0]),
                "n_simulations":               int(final_infected.shape[0]),
                "total_nodes":                 int(total_nodes),
                "ignition_probability":        float((final_fraction >= ignition_threshold_fraction).mean()),
                "full_cascade_probability":    float((final_infected == total_nodes).mean()),
                "mean_final_adoption_fraction": float(final_fraction.mean()),
                "std_final_adoption_fraction":  float(final_fraction.std()),
            })

    return pd.DataFrame(rows)


def build_full_analysis_table(ignition_threshold_fraction: float = 0.5, save: bool = True) -> pd.DataFrame:
    """
    The big table: one row per (network, threshold) combining mixing-pattern
    features, network topology stats, run parameters, and ignition
    probability from the diffusion simulations.

    Grain note: topology/mixing/run columns are per-network and get
    repeated across every threshold that network was simulated at (they
    don't change per threshold) -- ignition columns are what actually vary
    per row. If a network has no infection_events yet, it's simply absent
    from this table (inner-join behavior on the ignition side) rather than
    showing up with NaN ignition columns; use build_networks_analysis_table()
    directly if you want every generated network regardless of simulation
    status.

    Derived/regenerable, same as build_networks_analysis_table -- safe to
    delete data_lake/analysis_tables/full_table.parquet and rebuild anytime.
    """
    ignition = compute_ignition_stats(ignition_threshold_fraction)
    networks_table = build_networks_analysis_table(save=False)

    if networks_table.empty:
        print("No networks found yet -- nothing to merge.")
        return pd.DataFrame()

    if ignition.empty:
        print("No infection_events found yet -- run the simulation stage first.")
        return pd.DataFrame()

    table = ignition.merge(networks_table, on="network_id", how="left")

    if save:
        out_path = dl.ROOT / "analysis_tables" / "full_table.parquet"
        dl._write_parquet(out_path, table)
        print(f"  saved: {out_path}")

    return table


if __name__ == "__main__":
    table = build_full_analysis_table()
    if not table.empty:
        print(f"{table.shape[0]} rows x {table.shape[1]} columns")