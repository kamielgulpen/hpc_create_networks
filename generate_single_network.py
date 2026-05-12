# Imports
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
import networkx as nx
import matplotlib.pyplot as plt
from asnu import generate, create_communities, check_group_interactions, plot_group_interactions
import time
from scipy import stats
from collections import defaultdict
import igraph as ig

suffix = "geslacht"
layer = "buren"

# Generate network
# links = 'data/enriched/aggregated/interactions_etngrp_lft_inkomensniveau_arbeidsstatus_burgerlijke_staat.csv'
# links = f'Data/enriched/aggregated/interactions_{suffix}.csv'
# links = 'Data/enriched/aggregated/interactions_geslacht.csv'
links = f'Data/Data/enriched/aggregated/interactions_{suffix}.csv'

# as example we use group interaction data on a work / school layer
# pops = 'data/enriched/aggregated/pop_etngrp_lft_inkomensniveau_arbeidsstatus_burgerlijke_staat.csv' 
# pops  = f'Data/enriched/aggregated/pop_{suffix}.csv'
# pops  = 'Data/enriched/aggregated/pop_geslacht.csv'
pops = f'Data/Data/enriched/aggregated/pop_{suffix}.csv'

print(pops, links)
scale = 1
start = time.perf_counter()

# # Step 1: Create communities separately
create_communities(
    pops, links,
    scale=scale,
    number_of_communities=10796,
    output_path='my_communities.json',
    mode='capacity_fast',
    mixing_floor= 0.3,
    allow_new_communities=False,
    isolation_threshold = 0.05,

)

graph = generate(
    pops,                             # The group-level population data
    links,                            # The group-level interaction data
    preferential_attachment=0.0,     # Preferential attachment strength
    scale=scale,                        # Population scaling
    reciprocity = 1,                    # Reciprocal edge probability
    transitivity = 0.609851197462374,                 # Friend of a friend is my friend probability
    community_file='my_communities.json',                  
    base_path="my_network",           # Path for the FileBasedGraph's data
    bridge_probability=0,
    fully_connect_communities = False,
    fill_unfulfilled = True
)

end = time.perf_counter()
results = check_group_interactions(graph)
print(f"Execution time: {end - start:.4f} seconds")

# plot_group_interactions(results, graph)

G_rx = rx.PyDiGraph()
G_nx = graph.graph
# Create node mapping (NetworkX ID -> rustworkx index)
node_map = {}
for node in G_nx.nodes():
    node_attrs = G_nx.nodes[node]
    idx = G_rx.add_node(node_attrs if node_attrs else node)
    node_map[node] = idx

# Add edges
for u, v, edge_attrs in G_nx.edges(data=True):
    G_rx.add_edge(node_map[u], node_map[v], edge_attrs)

print(f"Graph: {len(G_rx)} nodes, {G_rx.num_edges()} edges")
print(f"Transitivity:{rx.transitivity(G_rx)}")
def nx_to_igraph(nx_graph):
    nodes = list(nx_graph.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in nx_graph.edges()]
    G = ig.Graph(n=len(nodes), edges=edges, directed=nx_graph.is_directed())
    return G
G_ig = ig.Graph.from_networkx(G_nx)
print(G_ig.transitivity_undirected(mode="zero"))           # igraph, directed graph)
# Get degree sequence
degrees = [G_rx.in_degree(node) for node in G_rx.node_indices()]

print(degrees.count(0))
print(degrees.count(1))
print(degrees.count(100))
print(f"Mean degree: {np.mean(degrees):.2f}")
print(f"Std degree: {np.std(degrees):.2f}")
print(f"Max degree: {max(degrees)}")
print(f"Min degree: {min(degrees)}")
print(f"Median degree: {np.median(degrees)}")
print(f"first q degree: {np.quantile(degrees, 0.25)}")
print(f"fourth q degree: {np.quantile(degrees, 0.75)}")
print(f"skew: {stats.skew(degrees)}")



plt.hist(degrees, bins = 50)
plt.show()
import pickle
# Create filename from params
# param_str = '_'.join(f'{k}={v}' for k, v in params.items())
# filename = f'a.pkl'
# Result: 'model_lr=0.001_batch_size=32_epochs=100.pkl'

# # Save
# with open(filename, 'wb') as f:
#     pickle.dump(G_nx, f)

print(nx.community.modularity(G_nx.to_undirected(), nx.community.label_propagation_communities(G_nx.to_undirected())))