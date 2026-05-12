"""
Stage 1: Generate networks for PAWN sensitivity analysis.

One SLURM task = one LHS sample. Generates the network and saves edges
as .npz. Metrics are computed later in stage 2 (compute_metrics_pawn.py).
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from SALib.sample import latin

from asnu import generate, create_communities


# =============================================================================
# Configuration
# =============================================================================

ENRICHED_AGG_DIR = Path('Data/Data/enriched/aggregated')
OUTPUT_DIR       = Path('pawn_results')
NETWORKS_DIR     = OUTPUT_DIR / 'networks'
SAMPLES_FILE     = OUTPUT_DIR / 'samples.csv'

SCALE         = 1
RECIPROCITY_P = 1
N_SAMPLES     = 250
RANDOM_SEED   = 42
PREF_ATTACHMENT = 0 # held fixed

PROBLEM = {
    'num_vars': 2,
    'names':    ['n_communities', 'transitivity'],
    'bounds':   [[1,   35000],
                 [0.0, 1.0]],
}
BRIDGE_PROBABILITY = 0.0  # held fixed


def get_or_create_samples():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLES_FILE.exists():
        return pd.read_csv(SAMPLES_FILE)
    samples = latin.sample(PROBLEM, N_SAMPLES, seed=RANDOM_SEED)
    df = pd.DataFrame(samples, columns=PROBLEM['names'])
    df['n_communities'] = df['n_communities'].round().astype(int)
    df.to_csv(SAMPLES_FILE, index=False)
    print(f"Wrote {len(df)} samples to {SAMPLES_FILE}")
    return df


def discover_enriched_pairs():
    excluded = ('inkomensniveau', 'arbeidsstatus', 'uitkeringstype', 'burgerlijke_staat')
    pairs = []
    for pop_file in sorted(ENRICHED_AGG_DIR.glob('pop_*.csv')):
        combo = pop_file.stem[len('pop_'):]
        if any(t in combo for t in excluded):
            continue
        links = ENRICHED_AGG_DIR / f'interactions_{combo}.csv'
        if links.exists():
            pairs.append((combo, str(pop_file), str(links)))
    return pairs


def edges_from_nx(G):
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    edges = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int32)
    return edges


def generate_one(sample_id, params, label, pops, links):

    ncom = int(params['n_communities'])
    tr   = float(params['transitivity'])

    out_dir = NETWORKS_DIR / f'sample_{sample_id:05d}' / label
    edges_file = out_dir / 'edges.npz'

    if edges_file.exists():
        print(f"  [{sample_id}/{label}] exists, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        communities_path = tmp.name

    try:
        t0 = time.perf_counter()
        create_communities(
            pops, links,
            scale=SCALE,
            number_of_communities=ncom,
            output_path=communities_path,
            mode='capacity_fast',
            allow_new_communities=False,
        )
        graph = generate(
            pops, links,
            preferential_attachment=PREF_ATTACHMENT,
            scale=SCALE,
            reciprocity=RECIPROCITY_P,
            transitivity=tr,
            community_file=communities_path,
            base_path=str(out_dir / 'gen'),
            bridge_probability=BRIDGE_PROBABILITY,
            fully_connect_communities=False,
            fill_unfulfilled=True,
        )
        elapsed = time.perf_counter() - t0
    finally:
        os.unlink(communities_path)

    edges = edges_from_nx(graph.graph)
    np.savez_compressed(edges_file, edges=edges)

    meta = {
        'sample_id':          sample_id,
        'label':              label,
        'pref_attachment':    PREF_ATTACHMENT,
        'n_communities':      ncom,
        'transitivity_param': tr,
        'bridge_probability': BRIDGE_PROBABILITY,
        'nodes':              graph.graph.number_of_nodes(),
        'edges':              graph.graph.number_of_edges(),
        'gen_time_s':         elapsed,
    }
    pd.Series(meta).to_json(out_dir / 'meta.json')
    print(f"  [{sample_id}/{label}] {meta['nodes']} nodes, {meta['edges']} edges, {elapsed:.1f}s")


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

    pairs = discover_enriched_pairs()
    if not pairs:
        print(f"No enriched pairs in {ENRICHED_AGG_DIR}. Exiting.")
        return

    params = samples.iloc[task_id]
    print(f"Sample {task_id}:"
          f"comms={int(params['n_communities'])} trans={params['transitivity']:.4f}")
    print(f"Using {len(pairs)} enriched pair(s).")

    for label, pops, links in pairs:
        generate_one(task_id, params, label, pops, links)

    print(f"Sample {task_id} done.")


if __name__ == '__main__':
    main()