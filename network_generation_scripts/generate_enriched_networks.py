import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import pickle
import time
from itertools import combinations
from pathlib import Path
from scipy import stats
from asnu import generate, create_communities


def nx_to_igraph(nx_graph):
    nodes = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]

    directed = nx_graph.is_directed()
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=directed)

    for attr in nx_graph.nodes[nodes[0]].keys() if nodes else []:
        ig_graph.vs[attr] = [nx_graph.nodes[n].get(attr) for n in nodes]

    ig_graph.vs["name"] = nodes

    if nx_graph.edges():
        first_edge = next(iter(nx_graph.edges(data=True)))
        for attr in first_edge[2].keys():
            ig_graph.es[attr] = [
                nx_graph[u][v].get(attr)
                for u, v in nx_graph.edges()
            ]

    return ig_graph


# ── Configuration ─────────────────────────────────────────────────────────────
ENRICHED_AGG_DIR = Path('Data/enriched/aggregated')

BASE_LAYERS          = ["werkschool", "huishouden", "familie", "buren"]
BASE_CHARACTERISTICS = sorted(["geslacht", "lft", "etngrp", "oplniv"])

scale             = 1
reciprocity_p     = 1
transitivity_p    = 0
bridge_probability = 0.2


# ── Discover all (source_label, pops_path, links_path) pairs ─────────────────

# 2. Enriched aggregated: discovered from pop_*.csv files
enriched_pairs = []
for pop_file in sorted(ENRICHED_AGG_DIR.glob('pop_*.csv')):
    combo_str  = pop_file.stem[len('pop_'):]
    links_file = ENRICHED_AGG_DIR / f'interactions_{combo_str}.csv'
    if links_file.exists():
        enriched_pairs.append((f'enriched/{combo_str}', str(pop_file), str(links_file)))

all_pairs = enriched_pairs

# ── Main loop ─────────────────────────────────────────────────────────────────
for preferential_attachment in np.linspace(0, 0.99, 10):
    for number_of_communities in np.linspace(1, 100, 10):
        number_of_communities = int(number_of_communities)

        params = (f"scale={scale}_comms={number_of_communities}"
                  f"_recip={reciprocity_p}_trans={transitivity_p}"
                  f"_pa={preferential_attachment:.2f}_bridge={bridge_probability}")
        count = 0
        for label, pops, links in all_pairs:
            count += 1
            print(f"\n{'='*60}")
            print(f"PA={preferential_attachment:.2f}  comms={number_of_communities}  [{label}]")
            print(f"{'='*60}")

            start = time.perf_counter()

            create_communities(
                pops, 
                links,
                scale=scale, 
                number_of_communities = number_of_communities,
                output_path='my_communities.json',
                mode= "capacity",

            )

            graph = generate(
                pops,
                links,
                preferential_attachment=preferential_attachment,
                scale=scale,
                reciprocity=reciprocity_p,
                transitivity=transitivity_p,
                community_file='my_communities.json',
                base_path=f'my_networks/{params}/{label}',
                bridge_probability=bridge_probability,
                fully_connect_communities=False,
                fill_unfulfilled=True,
            )

            elapsed = time.perf_counter() - start
            print(f"Generation time: {elapsed:.2f}s")

            G_nx = graph.graph
            G_ig = nx_to_igraph(G_nx)

            print(f"Nodes: {G_ig.vcount()}, Edges: {G_ig.ecount()}")
            print(f"Reciprocity:   {G_ig.reciprocity():.4f}")
            print(f"Transitivity:  {G_ig.transitivity_undirected():.4f}")

            degrees = G_ig.degree(mode="in")
            print(f"Degree — mean: {np.mean(degrees):.1f}, std: {np.std(degrees):.1f}, "
                  f"median: {np.median(degrees):.0f}, min: {min(degrees)}, max: {max(degrees)}, "
                  f"Q1: {np.quantile(degrees, 0.25):.0f}, Q3: {np.quantile(degrees, 0.75):.0f}, "
                  f"skew: {stats.skew(degrees):.2f}")

            # Save: Data/networks/<label>/<params>/<combo>.pkl
            # label is e.g. "werkschool/etngrp_lft" or "enriched/etngrp_geslacht_lft_oplniv"
            combo_name = Path(label).name
            output_dir = Path('Data/networks') / Path(label).parent / params
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'node_distribution').mkdir(exist_ok=True)

            plt.hist(degrees, bins=50)
            plt.title(f"Degree distribution — {label}")
            plt.xlabel("Degree")
            plt.ylabel("Count")
            plt.savefig(output_dir / 'node_distribution' / f'{combo_name}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()