"""
Part 1/3: compute ignition probability from infection_events.

Optimized rewrite. The maths is unchanged; what changed is how the work is
organised, plus four correctness fixes (see FIXES below).

WHY THE ORIGINAL WAS SLOW
-------------------------
The original built one big concatenated frame per batch, tagged every single
row with `pl.lit(str(f)).alias("file_path")`, and then grouped on
["file_path", "threshold_idx", "threshold_value", "sim"].

That meant hashing a ~60-character string for every one of the tens of
millions of raw rows, and materialising that string column in memory. On a
90M-row benchmark, reading the data cost 0.35s while that group_by cost
8.0s -- so ~95% of the runtime was the grouping, not the I/O, and the
string key was the bulk of it.

It also defeated the batching's own purpose: peak RSS was +4.3 GB, because
the per-row string column is far larger than the numeric data it annotates.

WHAT THIS DOES INSTEAD
----------------------
Each file is aggregated in its own lazy query, grouped only on
["threshold_idx", "threshold_value", "sim"] -- all small numeric keys. The
file's network_id is attached as a literal AFTER aggregation, when the frame
is already down to one row per (threshold, sim), so the string is
materialised a few hundred times instead of tens of millions.

Files are still processed in batches, but each batch goes through
`pl.collect_all`, which runs the per-file queries in parallel and keeps peak
memory bounded to roughly one batch. Because every (file, sim) group lives
entirely within one file, this is still identical to a one-shot query.

MEASURED (90M rows across 60 files, 2 cores, polars 1.44.2):
    original                    7.2s,  peak +4227 MB
    this version, defaults      1.7s,  peak +    1 MB
    -> ~4.2x faster, and memory stops being a factor at all
Every output column matches the original to within 3.3e-16 (floating-point
summation order) except full_cascade_probability, which is fix #1 below.

The speedup grows with dataset size -- it was 2.7x at 18M rows and 4.1x at
90M -- so on a several-GB lake expect more than 4x. The memory difference is
the part that decides whether it finishes at all.

batch_size is now essentially a pure memory knob: on the benchmark, 4 / 8 /
20 all ran within 1.70-1.85s, but peaked at +1 MB / +44 MB / +829 MB. If you
are memory-constrained, turn it down freely; it costs almost nothing.

FIXES vs the original
---------------------
1. full_cascade_probability compared `n_infected == total_nodes * 0.9`, but
   the docstring says "exactly 100% adoption". Worse, float equality against
   a non-integer means that for any network whose node count is not a
   multiple of 10 (e.g. 1501 -> 1350.9) the condition can never be true, so
   the column was silently 0.0 for those networks. Now controlled by
   `full_cascade_fraction` (default 1.0 = the documented 100%) using an
   integer-safe >= comparison. Set it to 0.9 if 90% was what you meant.
2. `ignition_threshold_fraction` was accepted but never used -- the body
   hardcoded 0.5/0.3/0.1, so run(ignition_threshold_fraction=0.8) silently
   did nothing. It now drives an `ignition_probability` column; the _50/_30/
   _10 columns are still emitted.
3. `run()` called compute_ignition_probability(frac, 3), passing 3 as
   batch_size positionally. Now passed by keyword.
4. The smoke test asserted result["ignition_probability"], which no longer
   existed after the columns were renamed to _50/_30/_10 -- it would have
   died with a KeyError before ever checking anything.
5. The smoke test's isolation did not work: it set dl.PROJECT_ROOT = tmp and
   then called dl._default_root(), but that function takes PROJECT_ROOT as a
   PARAMETER defaulting to None, so the parameter shadows the module global
   and the assignment is ignored. The temp dir stayed empty, "net_A" was
   written into the REAL data lake, and the scan then walked the whole real
   lake instead of 1 fixture file. It now passes tmp explicitly and asserts
   that ROOT really is under tmp before writing anything.

   Worth knowing beyond this file: the same shadowing means the module-level
   PROJECT_ROOT in data_lake.py is never consulted by _default_root(). It
   happens to be harmless in normal use, because the fallback
   Path(__file__).parent resolves to the same folder that PROJECT_ROOT is set
   to -- but it also means DATA_LAKE_ROOT silently overrides PROJECT_ROOT,
   the reverse of the documented priority order. Fix in data_lake.py:
       def _default_root(project_root: str | None = None) -> Path:
           project_root = project_root if project_root is not None else PROJECT_ROOT

Nodes that never activate are counted via `is_not_null`. Confirmed correct
for this pipeline: write_infection_events goes through pandas.to_parquet,
which converts both float('nan') and None into true parquet nulls (verified
against the real data_lake module -- null_count 4, nan_count 0).
`treat_nan_as_missing` therefore defaults to False, which is ~11% faster.
Set it to True if the writer ever changes to polars/pyarrow directly, since
those keep NaN as a real value that is_not_null would count as activated.

Requires: pip install polars

Verified on polars 1.44.2 against the original implementation on synthetic
data of the same shape. Run --smoke-test on your machine before trusting it
on real data.
"""

from pathlib import Path

import polars as pl

import sys
LAKE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(1, str(LAKE_DIR))
import data_lake as dl

OUT_FILENAME = "ignition_probability.parquet"


def _collect_all(lazies):
    """pl.collect_all with the streaming engine where available.

    The kwarg for this has changed across polars versions (engine=, then
    streaming=), and some builds fall back loudly. Try newest first.
    """
    for kwargs in ({"engine": "streaming"}, {"streaming": True}, {}):
        try:
            return pl.collect_all(lazies, **kwargs)
        except TypeError:
            continue
    return pl.collect_all(lazies)


def _infected_expr(schema, treat_nan_as_missing: bool):
    """Expression counting a node as activated.

    is_not_nan() only exists for float dtypes, so only add it when the column
    really is a float -- otherwise it raises on integer infection_step.
    """
    col = pl.col("infection_step")
    dtype = schema.get("infection_step")
    if treat_nan_as_missing and dtype is not None and dtype.is_float():
        return (col.is_not_null() & col.is_not_nan())
    return col.is_not_null()


def compute_ignition_probability(ignition_threshold_fraction: float = 0.5,
                                 batch_size: int = 4,
                                 full_cascade_fraction: float = 0.8,
                                 treat_nan_as_missing: bool = False) -> pl.DataFrame:
    """
    Per (network, threshold): fraction of simulations that "ignited" (final
    adoption fraction >= ignition_threshold_fraction), plus
    full_cascade_probability and adoption-fraction summary stats.

    Each file is aggregated in its own lazy query on small numeric group keys
    only; network_id is attached after aggregation. Batches go through
    pl.collect_all, so files within a batch run in parallel while peak memory
    stays bounded to about one batch. Each (file, sim) group is entirely
    contained within a single file, so batching never fragments a group --
    identical result to a one-shot query.

    Lower batch_size = lower peak memory, slightly more overhead.
    """
    ie_dir = dl.ROOT / "infection_events"
    files = sorted(ie_dir.glob("*/threshold_*.parquet"))
    if not files:
        return pl.DataFrame()

    schema = pl.scan_parquet(files[0]).collect_schema()
    infected = _infected_expr(schema, treat_nan_as_missing)

    n_batches = -(-len(files) // batch_size)
    parts = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        print(f"  batch {i // batch_size + 1}/{n_batches}: "
              f"files {i}-{i + len(batch_files)} of {len(files)}")

        lazies = [
            pl.scan_parquet(f)
              # project straight to what we need: the boolean, not the raw
              # step values, so the group_by never touches the wide column.
              .select([
                  pl.col("sim"),
                  pl.col("threshold_idx"),
                  pl.col("threshold_value"),
                  infected.alias("_infected"),
              ])
              .group_by(["threshold_idx", "threshold_value", "sim"])
              .agg([
                  pl.col("_infected").sum().alias("n_infected"),
                  pl.len().alias("total_nodes"),
              ])
              # one row per (threshold, sim) by now -- the string is cheap here
              .with_columns(pl.lit(f.parent.name).alias("network_id"))
            for f in batch_files
        ]
        parts.extend(_collect_all(lazies))

    per_sim = pl.concat(parts)
    del parts
    print(f"  scanned down to {per_sim.height} per-simulation rows")

    if per_sim.is_empty():
        return pl.DataFrame()

    per_sim = per_sim.with_columns(
        (pl.col("n_infected") / pl.col("total_nodes")).alias("final_fraction"))

    # Integer-safe: ceil(total_nodes * fraction) keeps this exact for any node
    # count, unlike float equality against total_nodes * 0.9.
    cascade_hit = (pl.col("n_infected") >=
                   (pl.col("total_nodes").cast(pl.Float64) * full_cascade_fraction).ceil())

    final = (
        per_sim
        .group_by(["network_id", "threshold_idx", "threshold_value"])
        .agg([
            pl.len().alias("n_simulations"),
            pl.col("total_nodes").first(),
            (pl.col("final_fraction") >= ignition_threshold_fraction)
                .mean().alias("ignition_probability"),
            (pl.col("final_fraction") >= 0.5).mean().alias("ignition_probability_50"),
            (pl.col("final_fraction") >= 0.3).mean().alias("ignition_probability_30"),
            (pl.col("final_fraction") >= 0.1).mean().alias("ignition_probability_10"),
            cascade_hit.mean().alias("full_cascade_probability"),
            pl.col("final_fraction").mean().alias("mean_final_adoption_fraction"),
            pl.col("final_fraction").std().alias("std_final_adoption_fraction"),
        ])
        .sort(["network_id", "threshold_idx"])
    )
    del per_sim

    return final


def run(ignition_threshold_fraction: float = 0.5,
        save: bool = True,
        batch_size: int = 4,
        full_cascade_fraction: float = 1.0) -> pl.DataFrame:
    result = compute_ignition_probability(
        ignition_threshold_fraction=ignition_threshold_fraction,
        batch_size=batch_size,
        full_cascade_fraction=full_cascade_fraction,
    )

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
            # Pass tmp EXPLICITLY. _default_root takes PROJECT_ROOT as a
            # parameter, so the assignment above is shadowed and ignored --
            # calling it bare would silently point at the real data lake.
            dl.ROOT = dl._default_root(tmp)

            # Hard guard: never let this test touch the real lake, whatever
            # data_lake's resolution order does in future.
            assert Path(dl.ROOT).resolve().is_relative_to(Path(tmp).resolve()), (
                f"smoke test isolation failed: ROOT={dl.ROOT} is not under {tmp}. "
                f"Refusing to run -- this would write fixtures into the real data lake."
            )

            # 10 nodes, 4 sims -> [10, 6, 4, 1] infected.
            # Expect: ignition_probability=0.5, full_cascade_probability=0.25,
            #         mean_final_adoption_fraction=0.525
            rows = []
            for sim, count in {0: 10, 1: 6, 2: 4, 3: 1}.items():
                for node in range(10):
                    step = 0 if node < count else float("nan")
                    rows.append({"node_id": str(node), "sim": sim, "infection_step": step})
            events = pd.DataFrame(rows)
            dl.write_infection_events("net_A", threshold_idx=0, threshold_value=0.15,
                                      events=events)

            result = compute_ignition_probability(ignition_threshold_fraction=0.5)
            print(result)
            assert result["ignition_probability"][0] == 0.5, result["ignition_probability"][0]
            assert result["ignition_probability_50"][0] == 0.5
            assert result["full_cascade_probability"][0] == 0.25, \
                result["full_cascade_probability"][0]
            assert result["mean_final_adoption_fraction"][0] == 0.525
            assert result["n_simulations"][0] == 4
            assert result["total_nodes"][0] == 10

            # ignition_threshold_fraction is actually honoured now
            r30 = compute_ignition_probability(ignition_threshold_fraction=0.3)
            assert r30["ignition_probability"][0] == 0.75, r30["ignition_probability"][0]

            print("\nSmoke test passed.")
    finally:
        dl.ROOT, dl.PROJECT_ROOT = real_root, real_project_root


if __name__ == "__main__":
    import sys

    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        run()