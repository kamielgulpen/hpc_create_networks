"""
Part 3/3: join the ignition-probability table (part 1) with the rest-of-
table (part 2) into the final combined analysis table.

Reads the OUTPUTS parts 1 and 2 already saved to disk
(ignition_probability.parquet, networks_table.parquet) instead of
recomputing them -- run compute_ignition_probability.py and
build_networks_table.py first. This step should be genuinely cheap: it's
joining two small, already-aggregated tables (one row per network, one row
per network-threshold), not touching the raw infection_events data again.

If this still gets killed after this fix, the actual join is exploding --
most likely because network_id isn't unique in networks_table (a
duplicate-key join multiplies rows combinatorially). The check below
catches that and tells you before it eats all your memory.

Requires: pip install polars
"""

import polars as pl

import data_lake as dl

ANALYSIS_DIR_NAME = "analysis_tables"
IGNITION_FILENAME = "ignition_probability.parquet"
NETWORKS_FILENAME = "networks_table.parquet"
OUT_FILENAME = "full_table.parquet"


def join_full_table(save: bool = True) -> pl.DataFrame:
    analysis_dir = dl.ROOT / ANALYSIS_DIR_NAME
    ignition_path = analysis_dir / IGNITION_FILENAME
    networks_path = analysis_dir / NETWORKS_FILENAME

    if not ignition_path.exists():
        raise FileNotFoundError(
            f"{ignition_path} not found -- run compute_ignition_probability.py first."
        )
    if not networks_path.exists():
        raise FileNotFoundError(
            f"{networks_path} not found -- run build_networks_table.py first."
        )

    # Reading two small, already-aggregated parquet files -- this is the
    # part that should be fast. If it isn't, something upstream is wrong,
    # not this join.
    ignition = pl.read_parquet(ignition_path)
    networks = pl.read_parquet(networks_path)

    if ignition.is_empty() or networks.is_empty():
        print("One of the source tables is empty -- nothing to join.")
        return pl.DataFrame()

    # Guard against the classic join-explosion cause: duplicate keys on the
    # "one" side of what should be a many-to-one join. If network_id isn't
    # actually unique in networks_table, this join multiplies rows
    # combinatorially instead of just attaching columns -- exactly the kind
    # of thing that silently blows up memory.
    dup_count = networks.filter(pl.col("network_id").is_duplicated()).height
    if dup_count > 0:
        raise ValueError(
            f"networks_table has {dup_count} rows with a duplicated network_id -- "
            f"joining against this would multiply rows instead of just attaching "
            f"columns, which is likely why this got heavy. Fix build_networks_table.py's "
            f"upstream joins (network_stats/runs/mixing_features) before joining here."
        )

    table = ignition.join(networks, on="network_id", how="left")

    if save:
        out_path = analysis_dir / OUT_FILENAME
        table.write_parquet(out_path)
        print(f"  saved: {out_path}  ({table.height} rows, {len(table.columns)} columns)")

    return table


if __name__ == "__main__":
    result = join_full_table()
    if not result.is_empty():
        print(result.head())