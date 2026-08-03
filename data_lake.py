"""
The data lake: plain files on disk, no database involved.

This is deliberately the ONLY thing this module does — lay out a consistent
directory structure and give every pipeline stage a single, boring way to
write its output. No DuckDB, no SQL, no catalog. That comes later as a
layer on TOP of this (see pipeline_store.py) — it reads these files, it
doesn't replace them.

Layout:

    data_lake/
      aggregation_levels/
        {agg_level_id}/
          meta.json                 # label, description
          mixing_features.parquet   # one row per feature
          nodes.parquet             # node attributes, IF shared across samples
      runs/
        {run_id}/
          meta.json                 # sampled_params, seed
      networks/
        {network_id}/
          graph.npz | graph.graphml # whatever your generator produces
          meta.json                 # run_id, agg_level_id, gen_seed
          network_stats.parquet
          nodes.parquet             # node attributes, IF they vary per network
      simulations/
        {network_id}/
          {simulation_id}.json      # sim_index, seed
      infection_events/
        {network_id}/
          threshold_{i}.parquet

Every write function is idempotent-ish by construction: it always writes to
a path keyed by a deterministic id, so re-running a stage just overwrites
that one file rather than accumulating duplicates. Parallel workers are
safe as long as each one only ever touches its own network_id/run_id/
agg_level_id -- never share a destination file across processes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _default_root(PROJECT_ROOT: str | None = None) -> Path:
    """
    Resolution order, highest priority first:
      1. PROJECT_ROOT below, if set -- the explicit, one-true-location way
         to control this. Set it to your overarching project folder and
         every script that imports data_lake builds under
         {PROJECT_ROOT}/data_lake/, no matter what directory any given
         script is launched from.
      2. DATA_LAKE_ROOT environment variable, if set -- same idea, useful
         when you don't want to hardcode a path into the file itself (e.g.
         different paths on a cluster vs. your laptop).
      3. A "data_lake" folder next to this file, as a fallback default.
    """
    
    # (inside _default_root, replacing lines 63-69)
    if PROJECT_ROOT is not None:
        base = Path(PROJECT_ROOT).resolve()
    else:
        override = os.environ.get("DATA_LAKE_ROOT")
        base = Path(override).resolve() if override else Path(__file__).resolve().parent
    scale = os.environ.get("PIPELINE_SCALE")
    lake_name = f"data_lake_scale{scale}" if scale else "data_lake"
    return base / lake_name


# Set this to the one overarching folder everything should be built under,
# e.g. PROJECT_ROOT = r"C:\Users\you\pawn_project" on Windows, or
# PROJECT_ROOT = "/home/you/pawn_project" on Linux/mac. Leave as None to
# fall back to DATA_LAKE_ROOT / the file-relative default (see above).

from pathlib import Path

# Gets the folder containing the running script
current_folder = Path(__file__).resolve().parent

PROJECT_ROOT: str | None = current_folder

ROOT = _default_root()


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, default=str)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ----------------------------------------------------------------------
# aggregation levels + mixing-pattern features
# ----------------------------------------------------------------------

def write_aggregation_level(agg_level_id: str, label: str,
                             description: dict | None = None) -> Path:
    path = ROOT / "aggregation_levels" / agg_level_id / "meta.json"
    _write_json(path, {"agg_level_id": agg_level_id, "label": label,
                        "description": description or {}})
    return path


def write_mixing_features(agg_level_id: str, features: dict[str, float]) -> Path:
    path = ROOT / "aggregation_levels" / agg_level_id / "mixing_features.parquet"
    df = pd.DataFrame(
        {"feature_name": list(features.keys()), "feature_value": list(features.values())}
    )
    _write_parquet(path, df)
    return path


def write_population_counts(agg_level_id: str, population: pd.DataFrame) -> Path:
    """
    `population` is a demographic-group count table: one row per group
    combination at this aggregation level, with an 'n' column giving the
    number of individuals in that group. This is the input to network
    generation (asnu.generate()'s `pops` argument), NOT the per-individual
    node list that comes out of generation later (see write_nodes below).
    """
    path = ROOT / "aggregation_levels" / agg_level_id / "population.parquet"
    _write_parquet(path, population)
    return path


def write_interaction_counts(agg_level_id: str, layer: str, interactions: pd.DataFrame) -> Path:
    """
    `interactions` is a group x group contact-count matrix for one layer
    (e.g. 'household', 'school', 'work', 'leisure') at this aggregation
    level -- the input to network generation (asnu.generate()'s `links`
    argument).
    """
    path = ROOT / "aggregation_levels" / agg_level_id / f"interactions_{layer}.parquet"
    _write_parquet(path, interactions)
    return path


# ----------------------------------------------------------------------
# runs (sampled parameters)
# ----------------------------------------------------------------------

def write_run(run_id: str, sampled_params: dict, seed: int | None = None) -> Path:
    path = ROOT / "runs" / run_id / "meta.json"
    _write_json(path, {"run_id": run_id, "sampled_params": sampled_params, "seed": seed})
    return path


def write_samples(samples: pd.DataFrame) -> Path:
    """
    The LHS sample design matrix itself -- one row per sample_id, generated
    once upfront by SALib before any individual run/network exists. Lives
    at the top level of the lake (not under runs/) since it's the sampling
    plan as a whole, not any single run's metadata.
    """
    path = ROOT / "samples.parquet"
    _write_parquet(path, samples)
    return path


def read_samples() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "samples.parquet")


def samples_exist() -> bool:
    return (ROOT / "samples.parquet").exists()


# ----------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------

def write_network_meta(network_id: str, run_id: str, agg_level_id: str,
                        gen_seed: int | None = None, **extra) -> Path:
    """
    Call this alongside however you already save the graph file itself
    (nx.write_graphml, np.savez_compressed, etc.) -- point it at
    `network_dir(network_id)` so the graph file and its metadata live
    side by side.
    """
    path = ROOT / "networks" / network_id / "meta.json"
    _write_json(path, {
        "network_id": network_id, "run_id": run_id, "agg_level_id": agg_level_id,
        "gen_seed": gen_seed, **extra,
    })
    return path


def network_dir(network_id: str) -> Path:
    """Directory to save the actual graph file into (graph.npz / graph.graphml / ...)."""
    d = ROOT / "networks" / network_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_network_stats(network_id: str, stats: dict[str, float]) -> Path:
    path = ROOT / "networks" / network_id / "network_stats.parquet"
    df = pd.DataFrame(
        {"stat_name": list(stats.keys()), "stat_value": list(stats.values())}
    )
    _write_parquet(path, df)
    return path


# ----------------------------------------------------------------------
# nodes -- keyed by agg_level_id (shared) or network_id (per-network)
# ----------------------------------------------------------------------

def write_nodes(key_id: str, key_type: str, nodes: pd.DataFrame) -> Path:
    """
    `nodes` needs a 'node_id' column plus typed attribute columns (not JSON --
    let parquet's columnar compression do its job on repetitive fields).

    `key_type` is 'agg_level_id' if these attributes are the same for every
    sample of that aggregation level (write ONCE per level), or 'network_id'
    if they genuinely vary per network. Check this before writing 750 copies
    of the same 1M-row table.
    """
    if key_type == "agg_level_id":
        path = ROOT / "aggregation_levels" / key_id / "nodes.parquet"
    elif key_type == "network_id":
        path = ROOT / "networks" / key_id / "nodes.parquet"
    else:
        raise ValueError("key_type must be 'agg_level_id' or 'network_id'")
    _write_parquet(path, nodes)
    return path


# ----------------------------------------------------------------------
# simulations + infection events
# ----------------------------------------------------------------------

def write_simulation_meta(simulation_id: str, network_id: str, sim_index: int,
                           seed: int | None = None) -> Path:
    path = ROOT / "simulations" / network_id / f"{simulation_id}.json"
    _write_json(path, {
        "simulation_id": simulation_id, "network_id": network_id,
        "sim_index": sim_index, "seed": seed,
    })
    return path


def write_infection_events(network_id: str, threshold_idx: int, threshold_value: float,
                            events: pd.DataFrame) -> Path:
    """
    `events` is the long-format per-node, per-simulation output --
    see infection_events_df() in seeding_experiments.py for how to build it.
    One file per (network, threshold), written once by the worker that ran
    those simulations -- never appended to by another process.
    """
    path = ROOT / "infection_events" / network_id / f"threshold_{threshold_idx}.parquet"
    df = events.copy()
    df["threshold_idx"] = threshold_idx
    df["threshold_value"] = threshold_value
    _write_parquet(path, df)
    return path


# ----------------------------------------------------------------------
# glob patterns -- hand these straight to pipeline_store.PipelineStore later
# ----------------------------------------------------------------------

def nodes_glob(key_type: str = "agg_level_id") -> str:
    subdir = "aggregation_levels" if key_type == "agg_level_id" else "networks"
    return str(ROOT / subdir / "*" / "nodes.parquet")


def infection_events_glob() -> str:
    return str(ROOT / "infection_events" / "*" / "*.parquet")


if __name__ == "__main__":
    # Minimal smoke test of the layout -- no DuckDB, just files.
    write_aggregation_level("etngrp_geslacht", label="etngrp_geslacht",
                             description={"grouping": ["ethnic_group", "gender"]})
    write_mixing_features("etngrp_geslacht", {"assortativity": 0.14, "homophily_index": 0.62})

    write_run("sample_00001", {"n_communities": 12, "transitivity": 0.3}, seed=42)

    network_id = "sample_00001__etngrp_geslacht"
    write_network_meta(network_id, run_id="sample_00001", agg_level_id="etngrp_geslacht", gen_seed=42)
    write_network_stats(network_id, {"n_nodes": 1000, "n_edges": 4200, "density": 0.0084})

    nodes_df = pd.DataFrame({"node_id": range(5), "age_group": ["20-30"] * 5})
    write_nodes("etngrp_geslacht", "agg_level_id", nodes_df)

    write_simulation_meta("sim0", network_id, sim_index=0, seed=0)
    events_df = pd.DataFrame({
        "node_id": range(5), "sim": [0] * 5,
        "infection_step": [0, 1, 1, None, 2],
    })
    write_infection_events(network_id, threshold_idx=0, threshold_value=0.15, events=events_df)

    print("Data lake written under:", ROOT.resolve())
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(ROOT))