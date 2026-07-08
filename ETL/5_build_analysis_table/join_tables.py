"""
Part 3/3: join the ignition-probability table (part 1) with the rest-of-
table (part 2: topology + run + mixing features) into the final combined
analysis table.

Grain: one row per (network, threshold) -- network/run/mixing columns
repeat across every threshold that network was simulated at (they don't
change per threshold); the ignition columns are what actually vary per
row. A network with no simulations yet simply doesn't appear (left side
of the join is ignition, so networks without infection_events are absent,
not present-with-nulls).

Requires: pip install polars
Unverified in this sandbox -- see the caveat in compute_ignition_probability.py.
"""

import polars as pl

import data_lake as dl
from compute_ignition_probability import compute_ignition_probability
from build_networks_table import build_networks_table

OUT_FILENAME = "full_table.parquet"


def join_full_table(ignition_threshold_fraction: float = 0.5, save: bool = True) -> pl.DataFrame:
    ignition = compute_ignition_probability(ignition_threshold_fraction)
    networks = build_networks_table(save=False)

    if networks.is_empty():
        print("No networks found yet -- nothing to join.")
        return pl.DataFrame()
    if ignition.is_empty():
        print("No infection_events found yet -- run the simulation stage first.")
        return pl.DataFrame()

    table = ignition.join(networks, on="network_id", how="left")

    if save:
        out_path = dl.ROOT / "analysis_tables" / OUT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out_path)
        print(f"  saved: {out_path}  ({table.height} rows, {len(table.columns)} columns)")

    return table


if __name__ == "__main__":
    result = join_full_table()
    if not result.is_empty():
        print(result.head())
