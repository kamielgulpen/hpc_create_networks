"""
Stage 1: per-attribute mixing (homophily) features for every aggregation
level in the data lake.

For each aggregation level this reads population.parquet and every
interactions_{layer}.parquet, then -- for each grouping attribute actually
present in the data (1 to 4 of them) -- computes a *marginal* Coleman
homophily index on that single attribute, collapsing over the others.

Features produced
-----------------
population_hhi
    HHI of the population across the full joint grouping. A coarseness /
    aggregation-size signal (higher = fewer, more dominant groups).

per layer ({layer}_...):
    _degree_node_mean       mean interactions per individual (total interactions / total population)

per layer x attribute ({layer}_{attr}_...):
    _population_hhi        marginal population HHI on this attribute
                           (== _homophily_expected)
    _homophily_raw         within-group interaction fraction (attr matches)
    _homophily_expected    same fraction under random mixing
    _homophily_coleman     (raw - expected) / (1 - expected)
    _homophily_coleman_mean / _std   per-group Coleman, summarized
"""

from __future__ import annotations

import json

import pandas as pd

import data_lake


def read_aggregation_level(agg_level_id: str):
    level_dir = data_lake.ROOT / "aggregation_levels" / agg_level_id
    with open(level_dir / "meta.json") as f:
        keep_cols = json.load(f)["description"]["grouping"]

    population = pd.read_parquet(level_dir / "population.parquet")
    layers = {
        path.stem[len("interactions_"):]: pd.read_parquet(path)
        for path in sorted(level_dir.glob("interactions_*.parquet"))
    }
    return keep_cols, population, layers


def present_attributes(interactions: pd.DataFrame, population: pd.DataFrame,
                       candidates: list[str]) -> list[str]:
    """Attributes with both _src/_dst columns and a plain population column."""
    return [
        c for c in candidates
        if {f"{c}_src", f"{c}_dst"} <= set(interactions.columns)
        and c in population.columns
    ]


def hhi(counts: pd.Series) -> float:
    """Herfindahl index: sum of squared shares of a count vector."""
    shares = counts / counts.sum()
    return float((shares ** 2).sum())


def attr_homophily_features(interactions: pd.DataFrame, population: pd.DataFrame,
                            attr: str, prefix: str) -> dict[str, float]:
    """Marginal Coleman homophily on a single attribute, collapsing the rest."""
    src, dst = f"{attr}_src", f"{attr}_dst"

    out = interactions.groupby(src)["n"].sum()
    within = (interactions[interactions[src] == interactions[dst]]
              .groupby(src)["n"].sum()
              .reindex(out.index, fill_value=0))

    pop = population.groupby(attr)["n"].sum()
    p = (pop / pop.sum()).reindex(out.index)      # population share per group
    w = within / out                              # within-group fraction per group
    coleman_group = (w - p) / (1 - p)

    raw = within.sum() / out.sum()
    expected = hhi(pop)
    coleman = (raw - expected) / (1 - expected) if expected < 1 else float("nan")

    key = f"{prefix}_{attr}"
    return {
        f"{key}_population_hhi": expected,
        f"{key}_homophily_raw": float(raw),
        f"{key}_homophily_expected": expected,
        f"{key}_homophily_coleman": float(coleman),
        f"{key}_homophily_coleman_mean": float(coleman_group.mean()),
        f"{key}_homophily_coleman_std": float(coleman_group.std(ddof=0)),
    }


def compute_mixing_features(population: pd.DataFrame,
                             interaction_layers: dict[str, pd.DataFrame],
                             keep_cols: list[str]) -> dict[str, float]:
    total_pop = population["n"].sum()
    features = {"population_hhi": hhi(population["n"])}

    for layer, df in interaction_layers.items():
        # Compute mean degree per node for this layer
        total_interactions = df["n"].sum()
        features[f"{layer}_degree_node_mean"] = float(total_interactions / total_pop) if total_pop > 0 else float("nan")

        for attr in present_attributes(df, population, keep_cols):
            features.update(attr_homophily_features(df, population, attr, layer))
            
    return features


def run(agg_level_ids: list[str] | None = None) -> list[str]:
    agg_dir = data_lake.ROOT / "aggregation_levels"
    if agg_level_ids is None:
        agg_level_ids = sorted(p.name for p in agg_dir.iterdir() if p.is_dir())

    for agg_level_id in agg_level_ids:
        keep_cols, population, layers = read_aggregation_level(agg_level_id)
        features = compute_mixing_features(population, layers, keep_cols)
        data_lake.write_mixing_features(agg_level_id, features)
        print(f"  {agg_level_id}: {len(features)} features")

    return agg_level_ids


if __name__ == "__main__":
    levels = run()
    print(f"Computed mixing features for {len(levels)} aggregation levels")