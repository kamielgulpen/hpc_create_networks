import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import pickle
import time
import os
from itertools import combinations
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


# Generation parameters
scale = 0.001

number_of_communities = 1
preferential_attachment = 0

reciprocity_p = 1
transitivity_p = 0
bridge_probability = 0.1

for preferential_attachment in np.linspace(0,0.99,10):
    print(preferential_attachment/1)
    for number_of_communities in np.linspace(1,10,10):
        params = (f"scale={scale}_comms={number_of_communities}"
                f"_recip={reciprocity_p}_trans={transitivity_p}"
                f"_pa={preferential_attachment}_bridge={bridge_probability}")

        characteristics = sorted(["geslacht", "lft", "etngrp", "oplniv"])

        layer = "buren"

        for r in range(1, len(characteristics) + 1):
            for combo in combinations(characteristics, r):
                group_cols = list(combo)
                characteristics_string = '_'.join(group_cols)

                links = f'data/enriched/aggregated/interactions_{characteristics_string}.csv'
                pops = f'data/enriched/aggregated/pop_{characteristics_string}.csv'

                print(f"\n{'='*60}")
                print(f"Generating: {characteristics_string}")
                print(f"{'='*60}")

                start = time.perf_counter()

                create_communities(pops, links,
                                scale=scale, number_of_communities=int(number_of_communities),
                                output_path='my_communities.json')

                graph = generate(
                    pops,
                    links,
                    preferential_attachment=preferential_attachment,
                    scale=scale,
                    reciprocity=reciprocity_p,
                    transitivity=transitivity_p,
                    community_file='my_communities.json',
                    base_path="my_network",
                    bridge_probability=bridge_probability,
                )

                elapsed = time.perf_counter() - start
                print(f"Generation time: {elapsed:.2f}s")

                G_nx = graph.graph
                G_ig = nx_to_igraph(G_nx)

                print(f"Nodes: {G_ig.vcount()}, Edges: {G_ig.ecount()}")
                print(f"Reciprocity: {G_ig.reciprocity():.4f}")
                print(f"Transitivity: {G_ig.transitivity_undirected():.4f}")

                degrees = G_ig.degree(mode="in")
                print(f"Degree — mean: {np.mean(degrees):.1f}, std: {np.std(degrees):.1f}, "
                    f"median: {np.median(degrees):.0f}, min: {min(degrees)}, max: {max(degrees)}, "
                    f"Q1: {np.quantile(degrees, 0.25):.0f}, Q3: {np.quantile(degrees, 0.75):.0f}, "
                    f"skew: {stats.skew(degrees):.2f}")

                plt.hist(degrees, bins=50)
                plt.title(f"Degree distribution — {characteristics_string}")
                plt.xlabel("Degree")
                plt.ylabel("Count")


                # Save network: Data/networks/<params>/<characteristics>.pkl
                output_dir = f'networks/{layer}/{params}'
                os.makedirs(output_dir, exist_ok=True)
                filename = f'{output_dir}/{characteristics_string}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(graph, f)
                print(f"Saved to {filename}")
                os.makedirs(f"{output_dir}/node_distribution", exist_ok=True)
                plt.savefig(f'networks/{layer}/{params}/node_distribution/{characteristics_string}.png', dpi=300, bbox_inches='tight')
                plt.close()

