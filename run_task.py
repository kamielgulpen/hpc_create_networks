"""
SLURM array task runner for enriched network generation.

Each array task (SLURM_ARRAY_TASK_ID) maps to one (preferential_attachment,
n_communities) combination. For that combination, all enriched (pop, links) pairs
discovered in Data/enriched/aggregated/ are generated and saved as .pkl files.

Usage:
    python run_task.py --task_id $SLURM_ARRAY_TASK_ID
"""

import argparse
import os
import pickle
import time
from itertools import product
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from asnu import generate, create_communities


# =============================================================================
# Parameter space (must match generate_enriched_networks.py)
# =============================================================================

ENRICHED_AGG_DIR  = Path('Data/Data/enriched/aggregated')
SCALE             = 1
RECIPROCITY_P     = 1
TRANSITIVITY_P    = 0
BRIDGE_PROBABILITY = 0.2

PREF_ATTACHMENT_VALUES = np.linspace(0, 0.99, 10)
N_COMMUNITIES_VALUES   = np.linspace(1, 100, 10).astype(int)


def all_combinations():
    """Return all (pref_attachment, n_communities) pairs, ordered by task_id."""
    return list(product(PREF_ATTACHMENT_VALUES, N_COMMUNITIES_VALUES))


def discover_enriched_pairs():
    """Discover all (label, pops_path, links_path) from Data/enriched/aggregated/."""
    pairs = []
    for pop_file in sorted(ENRICHED_AGG_DIR.glob('pop_*.csv')):
        combo_str  = pop_file.stem[len('pop_'):]
        links_file = ENRICHED_AGG_DIR / f'interactions_{combo_str}.csv'
        if links_file.exists():
            pairs.append((f'enriched/{combo_str}', str(pop_file), str(links_file)))
    return pairs


# =============================================================================
# Helpers
# =============================================================================

def nx_to_igraph(nx_graph):
    nodes = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=nx_graph.is_directed())
    if nodes:
        for attr in nx_graph.nodes[nodes[0]].keys():
            ig_graph.vs[attr] = [nx_graph.nodes[n].get(attr) for n in nodes]
    ig_graph.vs["name"] = nodes
    if nx_graph.edges():
        first_edge = next(iter(nx_graph.edges(data=True)))
        for attr in first_edge[2].keys():
            ig_graph.es[attr] = [nx_graph[u][v].get(attr) for u, v in nx_graph.edges()]
    return ig_graph


def params_string(pref_att, n_comms):
    return (f"scale={SCALE}_comms={n_comms}"
            f"_recip={RECIPROCITY_P}_trans={TRANSITIVITY_P}"
            f"_pa={pref_att:.2f}_bridge={BRIDGE_PROBABILITY}")


def generate_one(pref_att, n_comms, label, pops, links, params):
    """Generate and save one enriched network."""
    combo_name = Path(label).name
    output_dir = Path('Data/networks') / Path(label).parent / params
    out_file   = output_dir / f'{combo_name}.pkl'

    if out_file.exists():
        print(f"  Already exists: {out_file}. Skipping.")
        return

    print(f"\n{'='*60}")
    print(f"PA={pref_att:.2f}  comms={n_comms}  [{label}]")
    print(f"{'='*60}")

    start = time.perf_counter()

    create_communities(
        pops,
        links,
        scale=SCALE,
        number_of_communities=n_comms,
        output_path='my_communities.json',
        mode='capacity',
    )

    graph = generate(
        pops,
        links,
        preferential_attachment=pref_att,
        scale=SCALE,
        reciprocity=RECIPROCITY_P,
        transitivity=TRANSITIVITY_P,
        community_file='my_communities.json',
        base_path=f'my_networks/{params}/{label}',
        bridge_probability=BRIDGE_PROBABILITY,
        fully_connect_communities=False,
        fill_unfulfilled=True,
    )

    elapsed = time.perf_counter() - start
    print(f"Generation time: {elapsed:.2f}s")

    G_nx = graph.graph
    G_ig = nx_to_igraph(G_nx)
    degrees = G_ig.degree(mode="in")
    print(f"Nodes: {G_ig.vcount()}, Edges: {G_ig.ecount()}")
    print(f"Reciprocity:   {G_ig.reciprocity():.4f}")
    print(f"Transitivity:  {G_ig.transitivity_undirected():.4f}")
    print(f"Degree — mean: {np.mean(degrees):.1f}, std: {np.std(degrees):.1f}, "
          f"median: {np.median(degrees):.0f}, min: {min(degrees)}, max: {max(degrees)}, "
          f"Q1: {np.quantile(degrees, 0.25):.0f}, Q3: {np.quantile(degrees, 0.75):.0f}, "
          f"skew: {stats.skew(degrees):.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # with open(out_file, 'wb') as f:
    #     pickle.dump(graph, f)
    # print(f"Saved to {out_file}")

    dist_dir = output_dir / 'node_distribution'
    dist_dir.mkdir(exist_ok=True)
    plt.hist(degrees, bins=50)
    plt.title(f"Degree distribution — {label}")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.savefig(dist_dir / f'{combo_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run one enriched network generation task for SLURM array job")
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task index (0-based). Defaults to SLURM_ARRAY_TASK_ID env var.')
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    combos = all_combinations()
    n_total = len(combos)

    if task_id >= n_total:
        print(f"Task {task_id} out of range (only {n_total} combinations). Exiting.")
        return

    pref_att, n_comms = combos[task_id]
    print(f"Task {task_id}/{n_total - 1}: pref_attachment={pref_att:.4f}, n_communities={n_comms}")

    pairs = discover_enriched_pairs()
    if not pairs:
        print(f"No enriched pairs found in {ENRICHED_AGG_DIR}. Exiting.")
        return
    print(f"Found {len(pairs)} enriched pairs to generate.")

    params = params_string(pref_att, n_comms)

    for label, pops, links in pairs:
        generate_one(pref_att, n_comms, label, pops, links, params)

    print(f"\nTask {task_id} complete.")


if __name__ == '__main__':
    main()
