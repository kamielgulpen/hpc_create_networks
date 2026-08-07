"""
Stage 0: generate every aggregation level for the mixing-pattern data.

Takes the most granular population table (1 row per demographic group) and
the 4 interaction-layer tables (1 row per src-group x dst-group pair, per
layer), and re-aggregates both onto every combination of the 4 demographic
variables -- e.g. ['etngrp'], ['etngrp', 'geslacht'], ['lft', 'oplniv'], etc.

The invariant that matters: aggregating must never change the total number
of individuals (population) or the total number of interactions (per
layer) -- you're only ever regrouping rows, never dropping or duplicating
counts. Every aggregation call is checked against that with an assert.
"""

from __future__ import annotations

import itertools

import pandas as pd

import data_lake

VARS = ["etngrp", "geslacht", "lft", "oplniv"]


def combo_label(keep_cols: list[str]) -> str:
    return "_".join(keep_cols)


def aggregate_population(population: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    agg = population.groupby(keep_cols, as_index=False)["n"].sum()
    before, after = population["n"].sum(), agg["n"].sum()
    assert before == after, f"population count changed: {before} -> {after} for {keep_cols}"
    return agg


def aggregate_interactions(interactions: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    src_cols = [f"{c}_src" for c in keep_cols]
    dst_cols = [f"{c}_dst" for c in keep_cols]
    agg = interactions.groupby(src_cols + dst_cols, as_index=False)["n"].sum()
    before, after = interactions["n"].sum(), agg["n"].sum()
    assert before == after, f"interaction count changed: {before} -> {after} for {keep_cols}"
    return agg


def generate_all_levels(
    population: pd.DataFrame,
    interaction_layers: dict[str, pd.DataFrame],
    min_vars: int = 1,
    max_vars: int | None = None,
):
    """
    Yields (agg_level_id, keep_cols, agg_population, {layer_name: agg_interactions})
    for every combination of VARS of size min_vars..max_vars (inclusive).
    max_vars defaults to len(VARS) -- includes the full-variable combination
    as its own level (a re-aggregation onto all 4 variables at once, which
    is a no-op regroup of your original input but gets stored consistently
    alongside every other level). 2**len(VARS) - 1 = 15 levels total with
    the defaults (every non-empty subset of the 4 variables).
    """
    max_vars = max_vars if max_vars is not None else len(VARS)

    for r in range(min_vars, max_vars + 1):
        for combo in itertools.combinations(VARS, r):
            keep_cols = list(combo)
            agg_level_id = combo_label(keep_cols)

            agg_population = aggregate_population(population, keep_cols)
            agg_layers = {
                layer: aggregate_interactions(df, keep_cols)
                for layer, df in interaction_layers.items()
            }

            yield agg_level_id, keep_cols, agg_population, agg_layers


def run(population: pd.DataFrame, interaction_layers: dict[str, pd.DataFrame]) -> list[str]:
    """Generate every level and write it straight into the data lake."""
    written = []
    for agg_level_id, keep_cols, agg_population, agg_layers in generate_all_levels(
        population, interaction_layers
    ):
        data_lake.write_aggregation_level(
            agg_level_id, label=agg_level_id, description={"grouping": keep_cols}
        )
        data_lake.write_population_counts(agg_level_id, agg_population)
        for layer, df in agg_layers.items():
            data_lake.write_interaction_counts(agg_level_id, layer, df)
        written.append(agg_level_id)
        print(f"  {agg_level_id}: {len(agg_population)} groups, "
              f"{sum(len(d) for d in agg_layers.values())} interaction rows across "
              f"{len(agg_layers)} layers")
    return written


if __name__ == "__main__":
    import sys

    # Point these at your actual files.
    population = pd.read_csv("Data/Data/tab_n_(with oplniv).csv", dtype={"lft": str})
    interaction_layers = {
        "huishouden":pd.read_csv("Data/Data/tab_huishouden.csv", dtype={"lft_src": str, "lft_dst": str}),
        "werkschool":pd.read_csv("Data/Data/tab_werkschool.csv", dtype={"lft_src": str, "lft_dst": str}),
        "buren":     pd.read_csv("Data/Data/tab_buren.csv", dtype={"lft_src": str, "lft_dst": str}),
        "familie":   pd.read_csv("Data/Data/tab_familie.csv", dtype={"lft_src": str, "lft_dst": str}),
    }

    levels = run(population, interaction_layers)
    print(f"Wrote {len(levels)} aggregation levels: {levels}")