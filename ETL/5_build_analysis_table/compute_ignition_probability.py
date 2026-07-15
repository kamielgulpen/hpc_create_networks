"""
Part 1/3: compute ignition probability from infection_events.

This is the expensive part of the pipeline -- infection_events files can
be huge (up to n_nodes x n_simulations rows each) and the whole directory
can add up to several GB on disk, likely more once decompressed in memory.

Files are processed in BATCHES rather than one query over the whole
dataset, so peak memory is bounded to roughly "one batch's worth of
decompressed data" regardless of total dataset size -- this doesn't
depend on polars' streaming engine actually engaging on your installed
version, which varies enough across versions that it wasn't worth betting
on. Each (file, sim) group is entirely contained within a single file, so
batching never fragments a group -- identical result to a one-shot query,
just with bounded memory. Tune `batch_size` down if this still runs out
of memory.

Requires: pip install polars

Could not be executed in this sandbox (no internet access to install
polars) -- the two-stage aggregation math (raw rows -> per-sim counts ->
per-network-threshold stats), including the batching behavior, was
validated against known-correct numbers using an equivalent pandas
implementation, but the polars API calls themselves are unverified. Run
--smoke-test on your machine before trusting this on real data.
"""

from pathlib import Path

import polars as pl

import data_lake as dl

OUT_FILENAME = "ignition_probability.parquet"


def compute_ignition_probability(ignition_threshold_fraction: float = 0.5,
                                  batch_size: int = 50) -> pl.DataFrame:
    """
    Per (network, threshold): fraction of simulations that "ignited" (final
    adoption fraction >= ignition_threshold_fraction), plus
    full_cascade_probability (exactly 100% adoption -- an unambiguous,
    threshold-free alternative) and adoption-fraction summary stats.

    Processes files in batches of `batch_size` rather than one query over
    the whole dataset -- this bounds peak memory to roughly "one batch's
    worth of decompressed data" regardless of total dataset size, rather
    than depending on whether the streaming engine is actually engaging
    on your installed polars version. Each (file, sim) group is entirely
    contained within a single file, so batching never fragments a group --
    identical result to a one-shot query, just with bounded memory.
    Smaller batch_size = lower peak memory, more overhead; tune down if
    this still gets killed.
    """
    import gc

    ie_dir = dl.ROOT / "infection_events"
    files = sorted(ie_dir.glob("*/threshold_*.parquet"))
    if not files:
        return pl.DataFrame()

    n_batches = -(-len(files) // batch_size)  # ceil division
    per_sim_batches = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        print(f"  batch {i // batch_size + 1}/{n_batches}: "
              f"files {i}-{i + len(batch_files)} of {len(files)}")

        scans = [
            pl.scan_parquet(f)
              .select(["sim", "infection_step", "threshold_idx", "threshold_value"])
              .with_columns(pl.lit(str(f)).alias("file_path"))
            for f in batch_files
        ]

        batch_result = (
            pl.concat(scans)
            .group_by(["file_path", "threshold_idx", "threshold_value", "sim"])
            .agg([
                pl.col("infection_step").is_not_null().sum().alias("n_infected"),
                pl.len().alias("total_nodes"),
            ])
            .collect()
        )
        per_sim_batches.append(batch_result)
        del scans, batch_result
        gc.collect()

    per_sim = pl.concat(per_sim_batches)
    del per_sim_batches
    gc.collect()
    print(f"  scanned down to {per_sim.height} per-simulation rows")

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
            (pl.col("final_fraction") >= pl.col("total_nodes") * 0.5)
                .mean().alias("ignition_probability_50"),
            (pl.col("final_fraction") >= pl.col("total_nodes") * 0.3)
            .mean().alias("ignition_probability_30"),
            (pl.col("final_fraction") >= pl.col("total_nodes") * 0.1)
            .mean().alias("ignition_probability_10")
            (pl.col("n_infected") == pl.col("total_nodes") * 0.9)
                .mean().alias("full_cascade_probability"),
            pl.col("final_fraction").mean().alias("mean_final_adoption_fraction"),
            pl.col("final_fraction").std().alias("std_final_adoption_fraction"),
        ])
    )
    del per_sim
    gc.collect()

    return final


def run(ignition_threshold_fraction: float = 0.5, save: bool = True) -> pl.DataFrame:
    result = compute_ignition_probability(ignition_threshold_fraction)

    # Force any large intermediates (per_sim, the raw scan buffers) to
    # actually be freed before the write allocates anything new. Python's
    # refcounting should already drop them once compute_ignition_probability()
    # returns, but polars' underlying Rust allocator can hold memory in
    # arenas that don't get returned to the OS until GC runs -- if peak
    # memory during the scan was already close to the ceiling, this is
    # what decides whether the write's own allocation has room to succeed.
    import gc
    gc.collect()

    n_rows = result.height if not result.is_empty() else 0
    n_cols = len(result.columns) if not result.is_empty() else 0
    try:
        size_mb = result.estimated_size("mb") if not result.is_empty() else 0.0
        print(f"  result table: {n_rows} rows, {n_cols} columns, ~{size_mb:.1f} MB in memory")
    except AttributeError:
        print(f"  result table: {n_rows} rows, {n_cols} columns "
              f"(estimated_size not available on this polars version)")

    if save and not result.is_empty():
        out_path = dl.ROOT / "analysis_tables" / OUT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  writing to {out_path} ...")
        result.write_parquet(out_path)
        print(f"  saved: {out_path}  ({result.height} rows)")

    return result


def _smoke_test():
    """Quick self-check against synthetic data in a throwaway temp dir --
    does NOT touch your real data lake. Run with --smoke-test."""
    import tempfile
    import pandas as pd  # only used here, to build the test fixture via data_lake

    real_root, real_project_root = dl.ROOT, dl.PROJECT_ROOT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dl.PROJECT_ROOT = tmp
            dl.ROOT = dl._default_root()

            # 10 nodes, 4 sims -> [10, 6, 4, 1] infected.
            # Expect: ignition_probability=0.5, full_cascade_probability=0.25,
            #         mean_final_adoption_fraction=0.525
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
    finally:
        dl.ROOT, dl.PROJECT_ROOT = real_root, real_project_root


if __name__ == "__main__":
    import sys

    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        run()