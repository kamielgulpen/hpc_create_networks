# Imports
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
import pickle
import time
from pathlib import Path
from scipy import stats
from asnu import generate, create_communities, check_group_interactions

# ── Configuration ─────────────────────────────────────────────────────────────
# Pick any matched pop/interactions pair from Data/enriched/aggregated/
# e.g. "etngrp_geslacht_lft_oplniv" or "geslacht_lft_oplniv" etc.
COMBO = "etngrp_geslacht_lft_oplniv"

pops  = f'Data/enriched/aggregated/pop_{COMBO}.csv'
links = f'Data/enriched/aggregated/interactions_{COMBO}.csv'

scale = 0.1

# ── Build communities ──────────────────────────────────────────────────────────
start = time.perf_counter()

create_communities(
    pops,
    links,
    scale=scale,
    number_of_communities=10000,
    output_path='my_communities_enriched.json',
    community_size_distribution='natural',
    new_comm_penalty=100000000000000000000000000000000000,
)

# ── Generate network ───────────────────────────────────────────────────────────
graph = generate(
    pops,
    links,
    preferential_attachment=0.0,
    scale=scale,
    reciprocity=1,
    transitivity=0,
    community_file='my_communities_enriched.json',
    base_path='my_enriched_network',
    bridge_probability=0,
    fully_connect_communities=False,
    fill_unfulfilled=True,
)

end = time.perf_counter()
results = check_group_interactions(graph)
print(f"Execution time: {end - start:.4f} seconds")

# ── Convert to rustworkx for stats ────────────────────────────────────────────
G_nx  = graph.graph
G_rx  = rx.PyDiGraph()
node_map = {}
for node in G_nx.nodes():
    node_attrs = G_nx.nodes[node]
    idx = G_rx.add_node(node_attrs if node_attrs else node)
    node_map[node] = idx

for u, v, edge_attrs in G_nx.edges(data=True):
    G_rx.add_edge(node_map[u], node_map[v], edge_attrs)

print(f"Graph: {len(G_rx)} nodes, {G_rx.num_edges()} edges")
print(f"Transitivity: {rx.transitivity(G_rx)}")

# ── Degree statistics ─────────────────────────────────────────────────────────
degrees = [G_rx.in_degree(node) for node in G_rx.node_indices()]
print(f"Mean degree:   {np.mean(degrees):.2f}")
print(f"Std degree:    {np.std(degrees):.2f}")
print(f"Max degree:    {max(degrees)}")
print(f"Min degree:    {min(degrees)}")
print(f"Median degree: {np.median(degrees)}")
print(f"Q1 degree:     {np.quantile(degrees, 0.25)}")
print(f"Q3 degree:     {np.quantile(degrees, 0.75)}")
print(f"Skew:          {stats.skew(degrees):.4f}")
print(f"Degree 0 count: {degrees.count(0)}")

plt.hist(degrees, bins=50)
plt.xlabel('In-degree')
plt.ylabel('Count')
plt.title(f'Degree distribution — {COMBO}')
plt.tight_layout()
plt.show()

# ── Save ──────────────────────────────────────────────────────────────────────
filename = f'network_enriched_{COMBO}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(G_nx, f)
print(f"Saved -> {filename}")
