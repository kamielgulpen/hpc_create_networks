"""
Multiplex Network Generator
============================

Generates a 4-layer multiplex network where every node exists in all layers.
Household communities are hierarchically nested inside larger super-communities,
and household edges are pre-seeded into dependent layers (counting toward their
edge budget).

Layers:
  1. huishouden  (household)  — generated normally with create_communities()
  2. familie     (family)     — hierarchical communities from household,
                                pre-seeded with a fraction of household edges
  3. buren       (neighbours) — hierarchical communities from household,
                                pre-seeded with ALL household edges
  4. werkschool  (work/school) — generated independently

Pre-seeding ensures household edges count toward the layer's link budget
(no inflation), and nodes in the same household community share the same
super-community in buren/familie layers.
"""

import networkx as nx
import numpy as np
import pickle
import time
import os
from itertools import combinations
import igraph as ig
from asnu import generate, create_communities, create_hierarchical_community_file


# ============================================================================
# Layer definitions and per-layer parameters
# ============================================================================

LAYERS = ['huishouden', 'familie', 'buren', 'werkschool']

LAYER_PARAMS = {
    'huishouden': {
        'number_of_communities': 5000,
        'reciprocity': 1,
        'transitivity': 1,
        'preferential_attachment': 0,
        'bridge_probability': 0,
        'fill_unfulfilled' : False,
        'fully_connect_communities' : True
    },
    'familie': {
        'number_of_communities': 500,
        'reciprocity': 1,
        'transitivity': 1,
        'preferential_attachment': 0,
        'bridge_probability': 0.1,
        'fill_unfulfilled' : False,
        'fully_connect_communities' : False
    },
    'buren': {
        'number_of_communities': 5,
        'reciprocity': 1,
        'transitivity': 1,
        'preferential_attachment': 0,
        'bridge_probability': 0.2,
        'fill_unfulfilled' : True,
        'fully_connect_communities' : False
    },
    'werkschool': {
        'number_of_communities': 50,
        'reciprocity': 1,
        'transitivity': 0.5,
        'preferential_attachment': 0.1,
        'bridge_probability': 0.3,
        'fill_unfulfilled' : True,
        'fully_connect_communities' : False
    },
}

# Shared parameters
scale = 0.01
characteristics = sorted(["geslacht", "lft", "etngrp", "oplniv"])

# Hierarchical pre-seeding settings
family_fraction = 0.3  # fraction of household edges pre-seeded into familie


# ============================================================================
# Core functions
# ============================================================================

def generate_layer(layer_name, pops_path, links_path, layer_params, scale,
                   community_file=None, pre_seed_edges=None):
    """
    Generate a single network layer using ASNU.

    Parameters
    ----------
    layer_name : str
        Name of the layer (for logging and community file naming)
    pops_path : str
        Path to population CSV
    links_path : str
        Path to interaction CSV for this layer
    layer_params : dict
        Per-layer generation parameters
    scale : float
        Population scaling factor
    community_file : str, optional
        Path to pre-built community JSON. If None, creates one with
        create_communities().
    pre_seed_edges : list of (int, int), optional
        Edges to pre-seed into the graph (counted toward link budget).

    Returns
    -------
    nx.DiGraph
        The generated network layer
    """
    # Create community file if not provided (normal layers)
    if community_file is None:
        community_file = f'communities_{layer_name}.json'
        create_communities(
            pops_path, links_path,
            scale=scale,
            number_of_communities=layer_params['number_of_communities'],
            output_path=community_file,
            mode="capacity"
        )

    graph = generate(
        pops_path,
        links_path,
        preferential_attachment=layer_params['preferential_attachment'],
        scale=scale,
        reciprocity=layer_params['reciprocity'],
        transitivity=layer_params['transitivity'],
        fill_unfulfilled=layer_params['fill_unfulfilled'],
        fully_connect_communities=layer_params['fully_connect_communities'],
        community_file=community_file,
        base_path=f'temp_{layer_name}',
        bridge_probability=layer_params['bridge_probability'],
        pre_seed_edges=pre_seed_edges,
    )

    combined = graph.graph
    nodes = list(combined.nodes())
    edges = list(combined.edges())
    n = len(nodes)
    e = len(edges)
    avg_deg = e / n if n > 0 else 0

    recip_edges = sum(1 for u, v in combined.edges() if combined.has_edge(v, u))
    reciprocity = recip_edges / e if e > 0 else 0

    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    igraph_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]
    g = ig.Graph(n=n, edges=igraph_edges, directed=True)
    transitivity_ig = g.transitivity_undirected(mode="nan")

    print(f"  {layer_name:<15} {n:>8} {e:>10} {reciprocity:>13.3f} {avg_deg:>12.1f} {transitivity_ig:>11.3f}")

    return graph.graph


def save_multiplex(layers, output_dir):
    """
    Save multiplex network: individual pkl per layer + combined multiplex pkl.

    Parameters
    ----------
    layers : dict
        {layer_name: nx.DiGraph}
    output_dir : str
        Directory to save files in
    """
    os.makedirs(output_dir, exist_ok=True)

    # Individual layer files
    for layer_name, graph in layers.items():
        filepath = os.path.join(output_dir, f'{layer_name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)

    # Combined multiplex graph: merge all layers into a single nx.DiGraph
    # Each edge gets a 'layers' attribute (set of layer names it belongs to)
    combined = nx.DiGraph()
    for layer_name, graph in layers.items():
        combined.add_nodes_from(graph.nodes(data=True))
        for u, v, data in graph.edges(data=True):
            if combined.has_edge(u, v):
                combined[u][v]['layers'].add(layer_name)
            else:
                combined.add_edge(u, v, **data, layers={layer_name})

    print(f"\n{'='*70}")
    print("MULTIPLEX NETWORK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Layer':<15} {'Nodes':>8} {'Edges':>10} {'Reciprocity':>13} {'Avg Degree':>12} {'Transitivity':>11}")
    print(f"  {'-'*60}")


    n = combined.number_of_nodes()
    e = combined.number_of_edges()
    avg_deg = e / n if n > 0 else 0

    # Reciprocity
    recip_edges = sum(1 for u, v in combined.edges() if combined.has_edge(v, u))
    reciprocity = recip_edges / e if e > 0 else 0

    edges = list(combined.edges())
    nodes = list(combined.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    igraph_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]

    g = ig.Graph(n=len(nodes), edges=igraph_edges, directed=True)

    transitivity_ig = g.transitivity_undirected(mode="nan")

    print(f"  {"multiplex":<15} {n:>8} {e:>10} {reciprocity:>13.3f} {avg_deg:>12.1f} {transitivity_ig:>11.3f} ")

    multiplex_path = os.path.join(output_dir, 'multiplex.pkl')
    with open(multiplex_path, 'wb') as f:
        pickle.dump(combined, f)


def print_layer_stats(layers):
    """Print summary statistics for all layers."""
    print(f"\n{'='*70}")
    print("MULTIPLEX NETWORK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Layer':<15} {'Nodes':>8} {'Edges':>10} {'Reciprocity':>13} {'Avg Degree':>12} {'Transitivity':>11}")
    print(f"  {'-'*60}")

    for name, G in layers.items():
        n = G.number_of_nodes()
        e = G.number_of_edges()
        avg_deg = e / n if n > 0 else 0

        # Reciprocity
        recip_edges = sum(1 for u, v in G.edges() if G.has_edge(v, u))
        reciprocity = recip_edges / e if e > 0 else 0

        edges = list(G.edges())
        nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        igraph_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]

        g = ig.Graph(n=len(nodes), edges=igraph_edges, directed=True)

        transitivity_ig = g.transitivity_undirected(mode="nan")

        print(f"  {name:<15} {n:>8} {e:>10} {reciprocity:>13.3f} {avg_deg:>12.1f} {transitivity_ig:>11.3f} ")

    # Cross-layer overlap
    print(f"\n  Edge Overlap:")
    layer_names = list(layers.keys())
    for i in range(len(layer_names)):
        for j in range(i + 1, len(layer_names)):
            a_name, b_name = layer_names[i], layer_names[j]
            a_edges = set(layers[a_name].edges())
            b_edges = set(layers[b_name].edges())
            overlap = len(a_edges & b_edges)
            if overlap > 0:
                print(f"    {a_name} & {b_name}: {overlap} shared edges")

    print(f"{'='*70}\n")

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("MULTIPLEX NETWORK GENERATION (hierarchical pre-seeding)")
    print("="*70)
    print(f"\nLayers: {', '.join(LAYERS)}")
    print(f"Scale: {scale}")
    print(f"Family fraction: {family_fraction}")
    print()

    params_str = f"multiplex_scale={scale}"

    for r in range(1, len(characteristics) + 1):
        for combo in combinations(characteristics, r):
            group_cols = list(combo)
            characteristics_string = '_'.join(group_cols)

            pops = f'Data/aggregated/tab_n_{characteristics_string}.csv'

            print(f"\n{'='*60}")
            print(f"Generating multiplex: {characteristics_string}")
            print(f"{'='*60}")
            print(f"  {'Layer':<15} {'Nodes':>8} {'Edges':>10} {'Reciprocity':>13} {'Avg Degree':>12} {'Transitivity':>11}")
            print(f"  {'-'*70}")

            layers = {}
            total_start = time.perf_counter()

            # Paths
            hh_community_file = f'communities_huishouden_{characteristics_string}.json'

            # ── 1. huishouden: create communities normally, generate ──
            print(f"\n  --- Layer: huishouden ---")
            start = time.perf_counter()
            hh_links = f'Data/aggregated/tab_huishouden_{characteristics_string}.csv'

            create_communities(
                pops, hh_links,
                scale=scale,
                number_of_communities=LAYER_PARAMS['huishouden']['number_of_communities'],
                output_path=hh_community_file,
            )

            layers['huishouden'] = generate_layer(
                'huishouden', pops, hh_links,
                LAYER_PARAMS['huishouden'], scale,
                community_file=hh_community_file,
            )
            print(f"  huishouden: {layers['huishouden'].number_of_nodes()} nodes, "
                  f"{layers['huishouden'].number_of_edges()} edges ({time.perf_counter() - start:.2f}s)")

            hh_edges = list(layers['huishouden'].edges())

            # ── 2. familie: hierarchical communities from household, ──
            #       pre-seed family_fraction of household edges
            print(f"\n  --- Layer: familie (hierarchical, pre-seed {family_fraction:.0%} of hh edges) ---")
            start = time.perf_counter()
            fam_links = f'Data/aggregated/tab_familie_{characteristics_string}.csv'
            fam_community_file = f'communities_familie_{characteristics_string}.json'

            create_hierarchical_community_file(
                hh_community_file, pops, fam_links,
                scale=scale,
                target_num_communities=LAYER_PARAMS['familie']['number_of_communities'],
                output_path=fam_community_file,
            )

            # Select fraction of household edges to pre-seed
            import random
            fam_pre_edges = random.sample(hh_edges, int(len(hh_edges) * family_fraction))

            layers['familie'] = generate_layer(
                'familie', pops, fam_links,
                LAYER_PARAMS['familie'], scale,
                community_file=fam_community_file,
                pre_seed_edges=fam_pre_edges,
            )
            print(f"  familie: {layers['familie'].number_of_nodes()} nodes, "
                  f"{layers['familie'].number_of_edges()} edges "
                  f"(pre-seeded {len(fam_pre_edges)}, {time.perf_counter() - start:.2f}s)")

            # ── 3. buren: hierarchical communities from household, ──
            #       pre-seed ALL household edges
            print(f"\n  --- Layer: buren (hierarchical, pre-seed ALL hh edges) ---")
            start = time.perf_counter()
            bur_links = f'Data/aggregated/tab_buren_{characteristics_string}.csv'
            bur_community_file = f'communities_buren_{characteristics_string}.json'

            create_hierarchical_community_file(
                hh_community_file, pops, bur_links,
                scale=scale,
                target_num_communities=LAYER_PARAMS['buren']['number_of_communities'],
                output_path=bur_community_file,
            )

            layers['buren'] = generate_layer(
                'buren', pops, bur_links,
                LAYER_PARAMS['buren'], scale,
                community_file=bur_community_file,
                pre_seed_edges=hh_edges,
            )
            print(f"  buren: {layers['buren'].number_of_nodes()} nodes, "
                  f"{layers['buren'].number_of_edges()} edges "
                  f"(pre-seeded {len(hh_edges)}, {time.perf_counter() - start:.2f}s)")

            # ── 4. werkschool: independent, normal generation ──
            print(f"\n  --- Layer: werkschool (independent) ---")
            start = time.perf_counter()
            ws_links = f'Data/aggregated/tab_werkschool_{characteristics_string}.csv'

            layers['werkschool'] = generate_layer(
                'werkschool', pops, ws_links,
                LAYER_PARAMS['werkschool'], scale,
            )
            print(f"  werkschool: {layers['werkschool'].number_of_nodes()} nodes, "
                  f"{layers['werkschool'].number_of_edges()} edges ({time.perf_counter() - start:.2f}s)")

            # Print summary
            print_layer_stats(layers)

            # Save
            output_dir = f'Data/networks/{params_str}/{characteristics_string}'
            save_multiplex(layers, output_dir)

            total_elapsed = time.perf_counter() - total_start
            print(f"Saved to {output_dir}/ ({total_elapsed:.1f}s total)")

            exit()
if __name__ == "__main__":
    main()
