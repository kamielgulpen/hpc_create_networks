"""
Part 1/3: compute ignition probability from infection_events.

This is the expensive part of the pipeline -- infection_events files can
be huge (up to n_nodes x n_simulations rows each). Polars scans all of
them in one lazy, multi-threaded query with column pruning, instead of
looping over files and reading full columns in pandas.

Requires: pip install polars

Could not be executed in this sandbox (no internet access to install
polars) -- the two-stage aggregation math (raw rows -> per-sim counts ->
per-network-threshold stats) was validated against known-correct numbers
using an equivalent pandas implementation, but the polars API calls
themselves are unverified. Run the smoke test at the bottom of this file
on your machine before trusting it on real data.
"""

from pathlib import Path

import polars as pl

import data_lake as dl

OUT_FILENAME = "ignition_probability.parquet"


def compute_ignition_probability(ignition_threshold_fraction: float = 0.5) -> pl.DataFrame:
    """
    Per (network, threshold): fraction of simulations that "ignited" (final
    adoption fraction >= ignition_threshold_fraction), plus
    full_cascade_probability (exactly 100% adoption -- an unambiguous,
    threshold-free alternative) and adoption-fraction summary stats.
    """
    ie_dir = dl.ROOT / "infection_events"
    files = sorted(ie_dir.glob("*/threshold_*.parquet"))
    if not files:
        return pl.DataFrame()

    # Nothing is read yet -- scan_parquet is lazy. threshold_idx comes from
    # the filename (cheap, no need to read it off disk); file_path is
    # attached as a literal column so we can recover network_id after
    # aggregation, without depending on a specific polars version's
    # include_file_paths kwarg.
    scans = [
        pl.scan_parquet(f)
          .select(["sim", "infection_step", "threshold_value"])
          .with_columns([
              pl.lit(str(f)).alias("file_path"),
              pl.lit(int(f.stem.removeprefix("threshold_"))).alias("threshold_idx"),
          ])
        for f in files
    ]

    # Stage 1: per (file, sim) -- the expensive part. Raw node-level rows
    # collapse down to one row per simulation; everything after this
    # operates on that tiny result, not the raw data.
    per_sim = (
        pl.concat(scans)
        .group_by(["file_path", "threshold_idx", "threshold_value", "sim"])
        .agg([
            pl.col("infection_step").is_not_null().sum().alias("n_infected"),
            pl.len().alias("total_nodes"),
        ])
        .collect()
    )

    if per_sim.is_empty():
        return pl.DataFrame()

    per_sim = per_sim.with_columns([
        pl.col("file_path")
          .map_elements(lambda p: Path(p).parent.name, return_dtype=pl.Utf8)
          .alias("network_id"),
        (pl.col("n_infected") / pl.col("total_nodes")).alias("final_fraction"),
    ])

    # Stage 2: per (network, threshold) -- cheap, operates on the reduced
    # per-sim table.
    final = (
        per_sim
        .group_by(["network_id", "threshold_idx", "threshold_value"])
        .agg([
            pl.len().alias("n_simulations"),
            pl.col("total_nodes").first(),
            (pl.col("final_fraction") >= ignition_threshold_fraction)
                .mean().alias("ignition_probability"),
            (pl.col("n_infected") == pl.col("total_nodes"))
                .mean().alias("full_cascade_probability"),
            pl.col("final_fraction").mean().alias("mean_final_adoption_fraction"),
            pl.col("final_fraction").std().alias("std_final_adoption_fraction"),
        ])
    )

    return final


def run(ignition_threshold_fraction: float = 0.5, save: bool = True) -> pl.DataFrame:
    result = compute_ignition_probability(ignition_threshold_fraction)
    if save and not result.is_empty():
        out_path = dl.ROOT / "analysis_tables" / OUT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(out_path)
        print(f"  saved: {out_path}  ({result.height} rows)")
    return result


if __name__ == "__main__":
    # Smoke test: 10 nodes, 4 sims -> [10, 6, 4, 1] infected.
    # Expect: ignition_probability=0.5, full_cascade_probability=0.25,
    #         mean_final_adoption_fraction=0.525
    import tempfile
    import pandas as pd  # only used here, to build the test fixture via data_lake

    with tempfile.TemporaryDirectory() as tmp:
        dl.PROJECT_ROOT = tmp
        dl.ROOT = dl._default_root()

        rows = []
        for sim, count in {0: 10, 1: 6, 2: 4, 3: 1}.items():
            for node in range(10):
                step = 0 if node < count else float("nan")
                rows.append({"node_id": str(node), "sim": sim, "infection_step": step})
        events = pd.DataFrame(rows)
        dl.write_infection_events("net_A", threshold_idx=0, threshold_value=0.15, events=events)

        result = compute_ignition_probability(ignition_threshold_fraction=0.5)
        print(result)
        assert result["ignition_probability"][0] == 0.5
        assert result["full_cascade_probability"][0] == 0.25
        assert result["mean_final_adoption_fraction"][0] == 0.525
        print("\nSmoke test passed.")
