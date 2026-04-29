"""
SLURM array task runner for enriched network generation.

Each array task (SLURM_ARRAY_TASK_ID) maps to one (preferential_attachment,
n_communities, transitivity, bridge_probability) combination.
"""

import argparse
import json
import os
import tempfile
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
# Parameter space
# =============================================================================

ENRICHED_AGG_DIR = Path('Data/Data/enriched/aggregated')
SCALE            = 1
RECIPROCITY_P    = 1

PREF_ATTACHMENT_VALUES    = np.linspace(0, 0.9999, 2)
N_COMMUNITIES_VALUES      = np.linspace(5000, 20000, 10).astype(int)
TRANSITIVITY_VALUES       = np.linspace(0,1,3)
BRIDGE_PROBABILITY_VALUES = np.array([0])


def all_combinations():
    """Return all (pa, n_comms, trans, bridge) pairs, ordered by task_id."""
    return list(product(
        PREF_ATTACHMENT_VALUES,
        N_COMMUNITIES_VALUES,
        TRANSITIVITY_VALUES,
        BRIDGE_PROBABILITY_VALUES,
    ))


def discover_enriched_pairs():
    excluded = ('inkomensniveau', 'arbeidsstatus', 'uitkeringstype', 'burgerlijke_staat')
    pairs = []
    for pop_file in sorted(ENRICHED_AGG_DIR.glob('pop_*.csv')):
        combo_str = pop_file.stem[len('pop_'):]
        if any(term in combo_str for term in excluded):
            continue
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


def params_string(pref_att, n_comms, trans, bridge):
    return (f"scale={SCALE}_comms={n_comms}"
            f"_recip={RECIPROCITY_P}_trans={trans:.2f}"
            f"_pa={pref_att:.2f}_bridge={bridge:.2f}")


def generate_one(pref_att, n_comms, trans, bridge, label, pops, links, params):
    combo_name = Path(label).name
    output_dir = Path('Data/networks') / Path(label).parent / params
    stats_file = output_dir / f'{combo_name}_stats.json'

    if stats_file.exists():
        print(f"  Already exists: {stats_file}. Skipping.")
        return

    print(f"\n{'='*60}")
    print(f"PA={pref_att:.2f}  comms={n_comms}  trans={trans:.2f}  bridge={bridge:.2f}  [{label}]")
    print(f"{'='*60}")

    start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        communities_path = tmp.name

    try:
        create_communities(
            pops,
            links,
            scale=SCALE,
            number_of_communities=n_comms,
            output_path=communities_path,
            mode='capacity_fast',
            allow_new_communities=False
        )

        graph = generate(
            pops,
            links,
            preferential_attachment=pref_att,
            scale=SCALE,
            reciprocity=RECIPROCITY_P,
            transitivity=trans,
            community_file=communities_path,
            base_path=f'my_networks/{params}/{label}',
            bridge_probability=bridge,
            fully_connect_communities=False,
            fill_unfulfilled=True,
        )
    finally:
        os.unlink(communities_path)

    elapsed = time.perf_counter() - start
    print(f"Generation time: {elapsed:.2f}s")

    G_nx = graph.graph
    G_ig = nx_to_igraph(G_nx)
    degrees = G_ig.degree(mode="in")

    community_membership = G_ig.vs['community'] if 'community' in G_ig.vs.attributes() else None
    modularity = G_ig.modularity(community_membership) if community_membership is not None else None

    net_stats = {
        'label':              label,
        'pref_attachment':    round(pref_att, 4),
        'n_communities':      int(n_comms),
        'transitivity_param': round(float(trans), 4),
        'bridge_probability': round(float(bridge), 4),
        'params':             params,
        'generation_time_s':  round(elapsed, 2),
        'nodes':              G_ig.vcount(),
        'edges':              G_ig.ecount(),
        'reciprocity':        round(G_ig.reciprocity(), 4),
        'transitivity':       round(G_ig.transitivity_undirected(), 4),
        'modularity':         round(modularity, 4) if modularity is not None else None,
        'degree_mean':        round(float(np.mean(degrees)), 2),
        'degree_std':         round(float(np.std(degrees)), 2),
        'degree_median':      round(float(np.median(degrees)), 1),
        'degree_min':         int(min(degrees)),
        'degree_max':         int(max(degrees)),
        'degree_q1':          round(float(np.quantile(degrees, 0.25)), 1),
        'degree_q3':          round(float(np.quantile(degrees, 0.75)), 1),
        'degree_skew':        round(float(stats.skew(degrees)), 4),
    }

    print(f"Nodes: {net_stats['nodes']}, Edges: {net_stats['edges']}")
    print(f"Reciprocity:   {net_stats['reciprocity']:.4f}")
    print(f"Transitivity:  {net_stats['transitivity']:.4f}")
    print(f"Modularity:    {net_stats['modularity']}")
    print(f"Degree — mean: {net_stats['degree_mean']}, std: {net_stats['degree_std']}, "
          f"median: {net_stats['degree_median']}, min: {net_stats['degree_min']}, "
          f"max: {net_stats['degree_max']}, Q1: {net_stats['degree_q1']}, "
          f"Q3: {net_stats['degree_q3']}, skew: {net_stats['degree_skew']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(net_stats, f, indent=2)
    print(f"Stats saved to {stats_file}")

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

    pref_att, n_comms, trans, bridge = combos[task_id]
    print(f"Task {task_id}/{n_total - 1}: "
          f"pa={pref_att:.4f}, comms={n_comms}, trans={trans:.4f}, bridge={bridge:.4f}")

    pairs = discover_enriched_pairs()
    if not pairs:
        print(f"No enriched pairs found in {ENRICHED_AGG_DIR}. Exiting.")
        return
    print(f"Found {len(pairs)} enriched pairs to generate.")

    params = params_string(pref_att, n_comms, trans, bridge)

    for label, pops, links in pairs:
        generate_one(pref_att, n_comms, trans, bridge, label, pops, links, params)

    print(f"Task {task_id} complete.")


if __name__ == '__main__':
    main()