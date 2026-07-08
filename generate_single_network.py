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

suffix = ""
layer = "buren"

# Generate network
# links = 'data/enriched/aggregated/interactions_etngrp_lft_inkomensniveau_arbeidsstatus_burgerlijke_staat.csv'
# links = f'Data/enriched/aggregated/interactions_{suffix}.csv'
# links = 'Data/enriched/aggregated/interactions_geslacht.csv'
links = f'Data/Data/tab_{layer}.csv'

# as example we use group interaction data on a work / school layer
# pops = 'data/enriched/aggregated/pop_etngrp_lft_inkomensniveau_arbeidsstatus_burgerlijke_staat.csv' 
# pops  = f'Data/enriched/aggregated/pop_{suffix}.csv'
# pops  = 'Data/enriched/aggregated/pop_geslacht.csv'
pops = f'Data/Data/tab_n_(with oplniv).csv'

scale = 0.01
fraction_of_communities = 0.001
transitivity = 0
bridge_probability = 0
start = time.perf_counter()


# # Step 1: Create communities separately
create_communities(
    pops, links,
    scale=scale,
    fraction_of_communities=fraction_of_communities,
    output_path='my_communities.json',
    isolation_threshold = 0.8,
    refine_swaps=1

)

graph = generate(
    pops,                               # The group-level population data
    links, 
    preferential_attachment=  0,     # The group-level interaction data
    scale=scale,                        # Population scaling
    reciprocity = 1,                    # Reciprocal edge probability
    transitivity = 1,        # Friend of a friend is my friend probability
    internal_transitivity = 1,
    external_transitivity = 1,
    community_file='my_communities.json',                  
    base_path="my_network",              # Path for the FileBasedGraph's data
    bridge_probability=bridge_probability,
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

# Get degree sequence
degrees = [G_rx.in_degree(node) for node in G_rx.node_indices()]
top = int(np.argmax(degrees))
theta_top = graph.node_coordinates[top]
# hoeveel knopen delen precies deze coördinaat?
same = sum(1 for v in graph.node_coordinates.values() if v == theta_top)
# hoeveel knopen zitten er op de dichtstbijzijnde coördinaat eronder/erboven?
coords = sorted(set(graph.node_coordinates.values()))
i = coords.index(theta_top)
print("hub coord:", theta_top, "| knopen op die coord:", same)
print("hub degree:", degrees[top])
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
print(f"99th q degree: {np.quantile(degrees, 0.99)}")
print(f"skew: {stats.skew(degrees)}")


print(f"Graph: {len(G_rx)} nodes, {G_rx.num_edges()} edges")
print(f"Transitivity:{rx.transitivity(G_rx)}")

plt.hist(degrees, bins=50, color="#2563eb", alpha=0.7)
plt.show()
def nx_to_igraph(nx_graph):
    nodes = list(nx_graph.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in nx_graph.edges()]
    G = ig.Graph(n=len(nodes), edges=edges, directed=nx_graph.is_directed())
    return G
G_ig = ig.Graph.from_networkx(G_nx)
print(G_ig.transitivity_undirected(mode="zero"))           # igraph, directed graph)
