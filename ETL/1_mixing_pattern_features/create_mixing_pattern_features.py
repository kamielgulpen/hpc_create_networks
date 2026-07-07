"""
Stage 1: compute mixing-pattern features for every aggregation level already
written to the data lake by aggregate_mixing_data.py.

For each aggregation level, reads population.parquet and every
interactions_{layer}.parquet, and computes:

  - population_hhi
      Concentration of the population itself across groups at this level
      (sum of squared population shares). High = a few dominant groups.

  - per layer ({layer} = household, school, work, ...):
      {layer}_degree_mean / _std / _min / _max
          Group-level "degree" = total interaction volume per group (row
          sum of the group x group count matrix). Summarized across groups.
      {layer}_hhi_mean / _std / _max
          Per-group HHI: how concentrated each group's own contacts are
          across its partners (sum of squared partner shares), summarized
          across groups.
      {layer}_hhi_overall
          HHI of the whole layer's interaction volume across all group
          pairs at once (not per-group).
      {layer}_homophily_raw
          Fraction of this layer's interactions that stay within the same
          group (src group == dst group).
      {layer}_homophily_expected
          What that fraction would be under random mixing, given only the
          population shares (sum of squared population shares -- same
          formula as population_hhi, different meaning here: probability
          two randomly chosen individuals land in the same group).
      {layer}_homophily_coleman
          Coleman's homophily index: (raw - expected) / (1 - expected).
          0 = random mixing, 1 = fully segregated, negative = groups
          actively avoid their own kind.
      {layer}_homophily_coleman_mean / _std
          Per-group Coleman index, summarized across groups.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_lake


# ----------------------------------------------------------------------
# reading a level back out of the lake
# ----------------------------------------------------------------------

def read_aggregation_level(agg_level_id: str):
    level_dir = data_lake.ROOT / "aggregation_levels" / agg_level_id
    with open(level_dir / "meta.json") as f:
        meta = json.load(f)
    keep_cols = meta["description"]["grouping"]

    population = pd.read_parquet(level_dir / "population.parquet")
    layers = {}
    for path in sorted(level_dir.glob("interactions_*.parquet")):
        layer_name = path.stem[len("interactions_"):]
        layers[layer_name] = pd.read_parquet(path)

    return keep_cols, population, layers


# ----------------------------------------------------------------------
# core feature computations
# ----------------------------------------------------------------------

def population_hhi(population: pd.DataFrame) -> float:
    shares = population["n"] / population["n"].sum()
    return float((shares ** 2).sum())


def group_degrees(interactions: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    """Total interaction volume (out-strength) per group -- one row per group."""
    src_cols = [f"{c}_src" for c in keep_cols]
    out = interactions.groupby(src_cols, as_index=False)["n"].sum()
    out.columns = keep_cols + ["degree"]
    return out


def group_hhi(interactions: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    """Concentration of each group's own contacts across its partners."""
    src_cols = [f"{c}_src" for c in keep_cols]

    def _hhi(g):
        shares = g["n"] / g["n"].sum()
        return pd.Series({"hhi": (shares ** 2).sum()})

    out = interactions.groupby(src_cols).apply(_hhi, include_groups=False).reset_index()
    out.columns = keep_cols + ["hhi"]
    return out


def overall_hhi(interactions: pd.DataFrame) -> float:
    shares = interactions["n"] / interactions["n"].sum()
    return float((shares ** 2).sum())


def group_homophily(interactions: pd.DataFrame, population: pd.DataFrame,
                     keep_cols: list[str]) -> pd.DataFrame:
    """
    Per-group Coleman homophily index. Returns one row per group with
    out_total, within_total, w_i (within-group fraction), p_i (population
    share), and coleman_h_i.
    """
    src_cols = [f"{c}_src" for c in keep_cols]

    same_group = pd.Series(True, index=interactions.index)
    for c in keep_cols:
        same_group &= interactions[f"{c}_src"] == interactions[f"{c}_dst"]

    out_total = interactions.groupby(src_cols, as_index=False)["n"].sum()
    out_total.columns = keep_cols + ["out_total"]

    within = interactions.loc[same_group].groupby(src_cols, as_index=False)["n"].sum()
    within.columns = keep_cols + ["within_total"]

    merged = out_total.merge(within, on=keep_cols, how="left")
    merged["within_total"] = merged["within_total"].fillna(0)
    merged["w_i"] = merged["within_total"] / merged["out_total"]

    pop_group = population.groupby(keep_cols, as_index=False)["n"].sum()
    pop_group["p_i"] = pop_group["n"] / pop_group["n"].sum()

    merged = merged.merge(pop_group[keep_cols + ["p_i"]], on=keep_cols, how="left")
    merged["coleman_h_i"] = (merged["w_i"] - merged["p_i"]) / (1 - merged["p_i"])
    return merged


def compute_layer_features(interactions: pd.DataFrame, population: pd.DataFrame,
                            keep_cols: list[str], prefix: str) -> dict[str, float]:
    features = {}

    degrees = group_degrees(interactions, keep_cols)["degree"]
    features[f"{prefix}_degree_mean"] = float(degrees.mean())
    features[f"{prefix}_degree_std"] = float(degrees.std(ddof=0))
    features[f"{prefix}_degree_min"] = float(degrees.min())
    features[f"{prefix}_degree_max"] = float(degrees.max())
    features[f"{prefix}_degree_sum"] = float(degrees.sum())
    features[f"{prefix}_degree_node_mean"] = float(degrees.sum())/float(population["n"].sum())

    hhis = group_hhi(interactions, keep_cols)["hhi"]
    features[f"{prefix}_hhi_mean"] = float(hhis.mean())
    features[f"{prefix}_hhi_std"] = float(hhis.std(ddof=0))
    features[f"{prefix}_hhi_max"] = float(hhis.max())
    features[f"{prefix}_hhi_overall"] = overall_hhi(interactions)

    hom = group_homophily(interactions, population, keep_cols)
    raw = hom["within_total"].sum() / hom["out_total"].sum()

    pop_group = population.groupby(keep_cols, as_index=False)["n"].sum()
    pop_shares = pop_group["n"] / pop_group["n"].sum()
    expected = float((pop_shares ** 2).sum())

    coleman_overall = (raw - expected) / (1 - expected) if expected < 1 else float("nan")

    features[f"{prefix}_homophily_raw"] = float(raw)
    features[f"{prefix}_homophily_expected"] = expected
    features[f"{prefix}_homophily_coleman"] = float(coleman_overall)
    features[f"{prefix}_homophily_coleman_mean"] = float(hom["coleman_h_i"].mean())
    features[f"{prefix}_homophily_coleman_std"] = float(hom["coleman_h_i"].std(ddof=0))

    return features


def compute_mixing_features(population: pd.DataFrame, interaction_layers: dict[str, pd.DataFrame],
                             keep_cols: list[str]) -> dict[str, float]:
    features = {"population_hhi": population_hhi(population)}
    for layer, df in interaction_layers.items():
        features.update(compute_layer_features(df, population, keep_cols, prefix=layer))
    return features


# ----------------------------------------------------------------------
# run over every aggregation level in the lake
# ----------------------------------------------------------------------

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