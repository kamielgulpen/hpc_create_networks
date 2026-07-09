"""
Stage 2: Generate networks for PAWN sensitivity analysis.

One SLURM task = one LHS sample. For that sample, generates one network per
(aggregation level, interaction layer) pair, reading population + interaction
data from the data lake (data_lake.py) instead of ENRICHED_AGG_DIR CSVs, and
writing results back into the lake. Metrics are computed later in stage 3
(compute_metrics_pawn.py).

Run this from the same directory as data_lake.py, or make sure it's on
your PYTHONPATH -- see the note on namespace-package import errors if
`from data_lake import ...` fails with "(unknown location)".
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.sample import latin

from asnu import generate, create_communities

import data_lake

from run_parallel_generate_network import N_SAMPLES

# =============================================================================
# Configuration
# =============================================================================

POP = 861000
SCALE           = 0.01
RECIPROCITY_P   = 1
N_SAMPLES       = N_SAMPLES
RANDOM_SEED     = 42
PREF_ATTACHMENT = 0  # held fixed
BRIDGE_PROBABILITY = 0.0  # held fixed

PROBLEM = {
    'num_vars': 2,
    'names':    ['n_communities', 'transitivity'],
    'bounds':   [[1/POP,   1.0],
                 [0.0, 1.0]],
}

# Same inclusion/exclusion policy as the original discover_enriched_pairs(),
# now applied to data-lake aggregation levels instead of pop_*.csv filenames.
# See caveat above -- worth revisiting now that all 15 combinations exist.
EXCLUDED_SUBSTRINGS = (
    'inkomensniveau',
    'arbeidsstatus',
    'uitkeringstype',
    'burgerlijke_staat',
    # 'lft',
    # 'etngrp',
    # 'geslacht',
    # 'etngrp',
    # 'oplniv'

)
ALLOWED_EXCEPTIONS = {
    'etngrp_geslacht_lft_oplniv',
    # 'geslacht_lft_oplniv',
    # 'lft_oplniv',
    # 'etngrp_geslacht',
    # 'geslacht',
}


def get_or_create_samples() -> pd.DataFrame:
    if data_lake.samples_exist():
        return data_lake.read_samples()
    samples = latin.sample(PROBLEM, N_SAMPLES, seed=RANDOM_SEED)
    df = pd.DataFrame(samples, columns=PROBLEM['names'])
    df.insert(0, 'sample_id', df.index)
    data_lake.write_samples(df)
    print(f"Wrote {len(df)} samples to the data lake")
    return df


def discover_aggregation_levels() -> list[str]:
    """Aggregation levels available in the lake, after the exclusion policy."""
    agg_dir = data_lake.ROOT / 'aggregation_levels'
    if not agg_dir.exists():
        return []

    levels = []
    for level_dir in sorted(agg_dir.iterdir()):
        if not level_dir.is_dir() or not (level_dir / 'population.parquet').exists():
            continue

        agg_level_id = level_dir.name
        if agg_level_id not in ALLOWED_EXCEPTIONS:
            if any(t in agg_level_id for t in EXCLUDED_SUBSTRINGS):
                continue

        levels.append(agg_level_id)
    return levels


def discover_layers(agg_level_id: str) -> list[str]:
    level_dir = data_lake.ROOT / 'aggregation_levels' / agg_level_id
    return sorted(
        p.stem[len('interactions_'):] for p in level_dir.glob('interactions_*.parquet')
    )


def materialize_csvs(agg_level_id: str, layer: str, tmp_dir: Path) -> tuple[str, str]:
    """
    asnu's generate()/create_communities() take CSV file paths -- write the
    lake's parquet data out as temp CSVs matching the original pop_*.csv /
    interactions_*.csv format. Drop this (and pass DataFrames straight
    through) if asnu turns out to accept them directly.
    """
    level_dir = data_lake.ROOT / 'aggregation_levels' / agg_level_id
    population = pd.read_parquet(level_dir / 'population.parquet')
    interactions = pd.read_parquet(level_dir / f'interactions_{layer}.parquet')

    pops_path = tmp_dir / f'pop_{agg_level_id}.csv'
    links_path = tmp_dir / f'interactions_{agg_level_id}_{layer}.csv'
    population.to_csv(pops_path, index=False)
    interactions.to_csv(links_path, index=False)
    return str(pops_path), str(links_path)


def edges_from_nx(G):
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    edges = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int32)
    return edges


def load_node_communities(communities_path: str) -> dict[str, int]:
    """
    Read {node_id_str: community_id} out of the community file
    create_communities() writes, BEFORE it gets deleted. Confirmed
    structure from real asnu output: a top-level dict with a
    'nodes_to_communities' key mapping node-id strings straight to
    integer community ids -- exactly what we need, nothing else in the
    file (probability_matrix, communities_to_nodes, communities_to_groups,
    node_coordinates) is relevant here.
    """
    with open(communities_path) as f:
        data = json.load(f)
    return {str(k): v for k, v in data['nodes_to_communities'].items()}


def extract_nodes(G, community_map: dict[str, int] | None = None) -> pd.DataFrame:
    """
    Pull per-individual attributes off the generated graph's nodes (whatever
    asnu.generate() attaches -- demographics, etc.) plus community_id from
    community_map, keyed by the same str(node_id) used for node_id here.
    """
    records = []
    matched = 0
    for node_id, attrs in G.nodes(data=True):
        node_id_str = str(node_id)
        rec = {"node_id": node_id_str, **attrs}
        if community_map is not None:
            community_id = community_map.get(node_id_str)
            rec["community_id"] = community_id
            if community_id is not None:
                matched += 1
        records.append(rec)

    if community_map is not None and records:
        print(f"  community_id matched for {matched}/{len(records)} nodes")

    return pd.DataFrame(records)


def generate_one(sample_id: int, params: pd.Series, agg_level_id: str, layer: str,
                  pops_path: str, links_path: str) -> None:

    ncom = float(params['n_communities'])
    tr   = float(params['transitivity'])

    run_id     = f'sample_{sample_id:05d}'
    network_id = f'{run_id}__{agg_level_id}__{layer}'
    net_dir    = data_lake.network_dir(network_id)
    edges_file = net_dir / 'edges.npz'

    if edges_file.exists():
        print(f"  [{network_id}] exists, skipping")
        return

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        communities_path = tmp.name

    try:
        t0 = time.perf_counter()
        create_communities(
            pops_path, links_path,
            scale=SCALE,
            fraction_of_communities=ncom,
            output_path=communities_path,
            isolation_threshold = 0.8,
            refine_swaps=100000
        )

        # Read the community assignment now, while the file still exists --
        # it gets deleted in `finally` below, before extract_nodes() runs.
        community_map = load_node_communities(communities_path)

        graph = generate(
            pops_path, links_path,
            preferential_attachment=PREF_ATTACHMENT,
            scale=SCALE,
            reciprocity=RECIPROCITY_P,
            transitivity=tr,
            community_file=communities_path,
            base_path=str(net_dir / 'gen'),
            bridge_probability=BRIDGE_PROBABILITY,
            fully_connect_communities=False,
            fill_unfulfilled=True,
        )
        elapsed = time.perf_counter() - t0
    finally:
        os.unlink(communities_path)

    edges = edges_from_nx(graph.graph)
    np.savez_compressed(edges_file, edges=edges)

    # Filesystem-only writes -- safe to call from inside a parallel worker.
    # The DuckDB catalog gets populated later by a single-process ingest
    # step, same pattern as everywhere else in this pipeline.
    data_lake.write_run(run_id, sampled_params={
        'n_communities': ncom, 'transitivity': tr,
    }, seed=RANDOM_SEED)

    data_lake.write_network_meta(
        network_id, run_id=run_id, agg_level_id=agg_level_id,
        gen_seed=RANDOM_SEED, layer=layer,
        pref_attachment=PREF_ATTACHMENT, bridge_probability=BRIDGE_PROBABILITY,
    )

    data_lake.write_network_stats(network_id, {
        'n_nodes':    graph.graph.number_of_nodes(),
        'n_edges':    graph.graph.number_of_edges(),
        'gen_time_s': elapsed,
    })

    # NODES: keyed by network_id (per sample x agg_level x layer) -- the
    # conservative default. If the same individuals turn out to be assigned
    # to every layer's network for a given (sample, agg_level) -- i.e. only
    # the edges differ between household/school/work/leisure, not the node
    # set -- switch key_id to f'{run_id}__{agg_level_id}' (drop the layer)
    # so the node table is written ONCE per (sample, agg_level) instead of
    # once per layer. At ~1M nodes per network that's a 4x storage
    # difference for nothing if the individuals are actually shared.
    nodes_df = extract_nodes(graph.graph, community_map)
    if not nodes_df.empty:
        data_lake.write_nodes(network_id, "network_id", nodes_df)

    print(f"  [{network_id}] {graph.graph.number_of_nodes()} nodes, "
          f"{graph.graph.number_of_edges()} edges, {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=None)
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    samples = get_or_create_samples()
    if task_id >= len(samples):
        print(f"task_id {task_id} out of range ({len(samples)}). Exiting.")
        return

    agg_level_ids = discover_aggregation_levels()
    if not agg_level_ids:
        print(f"No aggregation levels found in {data_lake.ROOT / 'aggregation_levels'}. Exiting.")
        return

    params = samples.iloc[task_id]

    print(f"Sample {task_id}: comms={params['n_communities']} trans={params['transitivity']:.4f}")
    print(f"Using {len(agg_level_ids)} aggregation level(s): {agg_level_ids}")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for agg_level_id in agg_level_ids:
            layers = discover_layers(agg_level_id)

            if not layers:
                print(f"  {agg_level_id}: no interaction layers found, skipping")
                continue
            for layer in layers:
                if layer not in ('werkschool'):
                    continue
                pops_path, links_path = materialize_csvs(agg_level_id, layer, tmp_dir)
                generate_one(task_id, params, agg_level_id, layer, pops_path, links_path)


    print(f"Sample {task_id} done.")


if __name__ == '__main__':
    main()