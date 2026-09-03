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

from asnu import generate, create_communities, clone_communities

import data_lake

# =============================================================================
# Configuration
# =============================================================================


SCALE           = float(os.environ.get("PIPELINE_SCALE", "0.10"))
RECIPROCITY_P   = 1
RANDOM_SEED     = 42
BRIDGE_PROBABILITY = 0.0

# Community CLONING switch (opt-in, unset = original behaviour).
# When CLONE_FROM_SCALE is set (e.g. "0.01"), generate_one() reuses each
# network's community structure from that smaller-scale lake -- cloning it up
# to SCALE via asnu.clone_communities -- instead of running the expensive
# create_communities() refinement. If a per-network source is missing it falls
# back to create_communities() for that network, so a partial small-scale run
# never silently drops networks. Unset -> create_communities() as before.
CLONE_FROM_SCALE = os.environ.get("CLONE_FROM_SCALE")  # str like "0.01" or None


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

REFERENCE_AGG_LEVEL = 'etngrp_geslacht_lft_oplniv'


def load_reference_losses() -> dict[tuple[int, str], float]:
    """
    {(sample_id, layer): loss} from the reference sweep (sweep_refine_loss.py).
    That sweep writes ONE parquet per layer, named
    f'{REFERENCE_AGG_LEVEL}__{layer}.parquet', each with sample_id / layer /
    loss columns. We glob them all and key on (sample_id, layer). Returns {}
    if the sweep hasn't been run yet.
    """
    ref_dir = data_lake.ROOT / 'refine_loss_reference'
    if not ref_dir.exists():
        return {}

    losses: dict[tuple[int, str], float] = {}
    for path in sorted(ref_dir.glob(f'{REFERENCE_AGG_LEVEL}__*.parquet')):
        df = pd.read_parquet(path)
        for s, l, v in zip(df['sample_id'], df['layer'], df['loss']):
            losses[(int(s), str(l))] = float(v)
    return losses


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


def clone_source_nodes_path(network_id: str) -> Path | None:
    """
    Locate the small-scale nodes.parquet to clone this network's community
    structure from, when CLONE_FROM_SCALE is set.

    data_lake.ROOT already encodes the CURRENT scale (…/data_lake_scale{SCALE}).
    The source lake is the sibling …/data_lake_scale{CLONE_FROM_SCALE} with the
    SAME network_id -- because network_id (sample__agg__layer) is scale-free, the
    small- and large-scale runs share it one-to-one. Returns the path if it
    exists, else None (caller falls back to create_communities()).
    """
    if not CLONE_FROM_SCALE:
        return None
    root = data_lake.ROOT
    src_root = root.with_name(f"data_lake_scale{CLONE_FROM_SCALE}")
    if src_root == root:
        # Current lake isn't scale-suffixed (PIPELINE_SCALE unset). Be explicit
        # rather than clone from ourselves.
        print(f"  [clone] WARNING: data lake root {root.name!r} is not "
              f"scale-suffixed; cannot locate a distinct source lake. "
              f"Set PIPELINE_SCALE. Falling back to create_communities.")
        return None
    src = src_root / "networks" / network_id / "nodes.parquet"
    return src if src.exists() else None


def generate_one(sample_id: int, params: pd.Series, agg_level_id: str, layer: str,
                  pops_path: str, links_path: str,
                  loss_goal: float | None = None) -> None:

    ncom = float(params['n_communities'])
    tr   = float(params['transitivity'])
    opt  = int(params['optimize'])

    print(ncom, tr, opt)
    refine_swaps = 1000000 if opt else 1

    # create_communities early-stops once loss <= loss_goal. -inf disables the
    # goal entirely (runs the full refine_swaps); it's what we pass when no
    # reference loss exists for this (sample, layer).
    cc_goal = loss_goal if loss_goal is not None else float('-inf')

    run_id     = f'sample_{sample_id:05d}'
    network_id = f'{run_id}__{agg_level_id}__{layer}__ncom_{round(ncom, 2)}__tr_{round(tr,2)}__opt_{round(opt,2)}'
    net_dir    = data_lake.network_dir(network_id)
    edges_file = net_dir / 'edges.npz'

    if edges_file.exists():
        print(f"  [{network_id}] exists, skipping")
        return

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        communities_path = tmp.name

    try:
        t0 = time.perf_counter()

        # Decide: clone the structure from a smaller-scale lake, or derive it
        # fresh with create_communities(). CLONE_FROM_SCALE opts in; a missing
        # per-network source falls back to create_communities().
        src_nodes = clone_source_nodes_path(network_id)
        src_nodes = None
        clone_used = False
        if src_nodes is not None:
            # clone_communities returns (path, cloned_loss). It reuses THIS
            # network's small-scale partition, scaling per-group counts (which
            # are not exact multiples of the small scale) via reconciliation.
            _, loss = clone_communities(
                old_nodes_path=str(src_nodes),
                pops_path=pops_path,
                scale_old=float(CLONE_FROM_SCALE),
                scale_new=SCALE,
                output_path=communities_path,
                seed=RANDOM_SEED,
                verbose=True,
            )
            clone_used = True
        else:
            if CLONE_FROM_SCALE:
                print(f"  [{network_id}] no clone source at scale "
                      f"{CLONE_FROM_SCALE}; using create_communities()")
            # create_communities returns (path, loss); we only need the loss.
            loss = create_communities(
                pops_path, links_path,
                scale=SCALE,
                fraction_of_communities=ncom,
                output_path=communities_path,
                refine_swaps=refine_swaps,
                loss_goal=cc_goal,
            )[1]

        # Read the community assignment now, while the file still exists --
        # it gets deleted in `finally` below, before extract_nodes() runs.
        community_map = load_node_communities(communities_path)

        graph = generate(
            pops_path, links_path,
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
        'n_communities': ncom, 'transitivity': tr, 'optimize': opt,
    }, seed=RANDOM_SEED)

    data_lake.write_network_meta(
        network_id, run_id=run_id, agg_level_id=agg_level_id,
        gen_seed=RANDOM_SEED, layer=layer,
        bridge_probability=BRIDGE_PROBABILITY,
    )

    data_lake.write_network_stats(network_id, {
        'n_nodes':          graph.graph.number_of_nodes(),
        'n_edges':          graph.graph.number_of_edges(),
        'gen_time_s':       elapsed,
        'refine_loss':      loss,        # achieved (CLONED estimate if clone_used)
        'refine_loss_goal': loss_goal,   # None when no goal was applied
        'clone_used':       clone_used,  # True -> loss is a cloned estimate, not optimized
        'clone_src_scale':  float(CLONE_FROM_SCALE) if (clone_used and CLONE_FROM_SCALE) else None,
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

    loss_str = f"{loss:.6g}" if loss is not None else "n/a"
    print(f"  [{network_id}] {graph.graph.number_of_nodes()} nodes, "
          f"{graph.graph.number_of_edges()} edges, {elapsed:.1f}s "
          f"(loss={loss_str}, goal={loss_goal})")


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

    samples = data_lake.read_samples()
    if task_id >= len(samples):
        print(f"task_id {task_id} out of range ({len(samples)}). Exiting.")
        return

    agg_level_ids = discover_aggregation_levels()
    if not agg_level_ids:
        print(f"No aggregation levels found in {data_lake.ROOT / 'aggregation_levels'}. Exiting.")
        return

    params = samples.iloc[task_id]
    # task_id positions into the (optimize=0 | optimize=1) concat, so it is NOT
    # the sample_id. The reference sweep is keyed on the real sample_id, so use
    # that for the loss lookup.
    sample_id = int(params['sample_id'])

    print(f"Task {task_id} -> sample {sample_id}: comms={params['n_communities']} "
          f"trans={params['transitivity']:.4f} optimize={int(params['optimize'])}")
    print(f"Using {len(agg_level_ids)} aggregation level(s): {agg_level_ids}")

    reference_losses = load_reference_losses()

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for agg_level_id in agg_level_ids:
            layers = discover_layers(agg_level_id)

            if not layers:
                print(f"  {agg_level_id}: no interaction layers found, skipping")
                continue

            for layer in layers:
                if layer in ('huishouden'):
                    continue

                loss_goal = reference_losses.get((sample_id, layer))
                if loss_goal is None:
                    print(f"  [{agg_level_id}/{layer}] no reference loss for "
                          f"sample {sample_id}; running without a goal")

                pops_path, links_path = materialize_csvs(agg_level_id, layer, tmp_dir)
                generate_one(task_id, params, agg_level_id, layer,
                             pops_path, links_path, loss_goal=loss_goal)

    print(f"Sample {task_id} done.")


if __name__ == '__main__':
    main()