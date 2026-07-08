"""
Part 2/3: build the "rest of the table" -- everything scoped to a network
except ignition: network identity/meta, topology stats, run parameters,
and mixing-pattern features for that network's aggregation level.

meta.json files are tiny -- read with plain json, no polars benefit there.
network_stats/mixing_features need a long -> wide pivot; that one step
goes through pandas (well-established pivot() API) since these tables are
tiny compared to infection_events -- no performance reason to gamble on
polars' pivot() API (which has shifted shape across versions, unverifiable
in this sandbox) for a step this small. Everything else stays in polars.

Requires: pip install polars
Unverified in this sandbox -- see the caveat in compute_ignition_probability.py.
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl

import data_lake as dl

OUT_FILENAME = "networks_table.parquet"


def _read_all_network_meta() -> pl.DataFrame:
    net_dir = dl.ROOT / "networks"
    if not net_dir.exists():
        return pl.DataFrame()
    rows = []
    for network_dir in sorted(p for p in net_dir.iterdir() if p.is_dir()):
        meta_path = network_dir / "meta.json"
        if meta_path.exists():
            rows.append(json.loads(meta_path.read_text()))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _read_all_runs() -> pl.DataFrame:
    runs_dir = dl.ROOT / "runs"
    if not runs_dir.exists():
        return pl.DataFrame()
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        row = {"run_id": meta["run_id"], "seed": meta.get("seed")}
        row.update(meta.get("sampled_params", {}))
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _read_and_pivot_long_parquet(files: list[Path], id_col: str, name_col: str,
                                  value_col: str, prefix: str = "") -> pl.DataFrame:
    """
    Scan+concat every file with polars (fast), then do the actual long ->
    wide pivot in pandas -- see module docstring for why.
    """
    if not files:
        return pl.DataFrame()

    scans = [
        pl.scan_parquet(f).with_columns(pl.lit(f.parent.name).alias(id_col))
        for f in files
    ]
    long_df = pl.concat(scans).collect().to_pandas()
    if long_df.empty:
        return pl.DataFrame()

    wide = long_df.pivot(index=id_col, columns=name_col, values=value_col).reset_index()
    wide.columns.name = None
    if prefix:
        wide = wide.rename(columns={c: f"{prefix}{c}" for c in wide.columns if c != id_col})

    return pl.from_pandas(wide)


def build_networks_table(save: bool = True) -> pl.DataFrame:
    meta = _read_all_network_meta()
    if meta.is_empty():
        print("No networks found yet -- nothing to build.")
        return pl.DataFrame()

    net_dir = dl.ROOT / "networks"
    stats_files = sorted(net_dir.glob("*/network_stats.parquet"))
    stats = _read_and_pivot_long_parquet(stats_files, "network_id", "stat_name", "stat_value")

    runs = _read_all_runs()

    agg_dir = dl.ROOT / "aggregation_levels"
    mix_files = sorted(agg_dir.glob("*/mixing_features.parquet"))
    mixing = _read_and_pivot_long_parquet(mix_files, "agg_level_id", "feature_name",
                                           "feature_value", prefix="mix_")

    table = meta
    if not stats.is_empty():
        table = table.join(stats, on="network_id", how="left")
    if not runs.is_empty():
        table = table.join(runs, on="run_id", how="left", suffix="_run")
    if not mixing.is_empty():
        table = table.join(mixing, on="agg_level_id", how="left")

    if save:
        out_path = dl.ROOT / "analysis_tables" / OUT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out_path)
        print(f"  saved: {out_path}  ({table.height} rows, {len(table.columns)} columns)")

    return table


if __name__ == "__main__":
    build_networks_table()
