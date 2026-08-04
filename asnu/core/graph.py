"""Minimal wrapper around nx.DiGraph for ASNU population-based network generation.

NetworkXGraph adds metadata tracking (groups, communities, link budgets) on top
of a plain nx.DiGraph. Use G.graph directly for all graph operations.
"""

import os
from collections import deque

import networkx as nx
import numpy as np


class NetworkXGraph:
    """Thin nx.DiGraph wrapper with metadata for population network generation.

    All graph operations use the underlying nx.DiGraph via ``self.graph``.

    Parameters
    ----------
    base_path : str, optional
        Directory for saving metadata. Default is "graph_data".
    """

    def __init__(self, base_path="graph_data"):
        self.base_path = base_path
        self.graph_file = os.path.join(base_path, "graph.gpickle")
        os.makedirs(base_path, exist_ok=True)

        self.graph = nx.DiGraph()

        # Metadata for population-based generation
        self.attrs_to_group = {}
        self.group_to_attrs = {}
        self.group_to_nodes = {}
        self.nodes_to_group = {}
        self.communities_to_nodes = {}
        self.nodes_to_communities = {}
        self.communities_to_groups = {}
        self.existing_num_links = {}
        self.maximum_num_links = {}
        self.number_of_communities = 0

    def get_non_isolates_batch(self, node_list, max_count=None):
        """Return nodes from `node_list` with degree > 0, up to `max_count`."""
        result = []
        for node in node_list:
            if node in self.graph and self.graph.degree(node) > 0:
                result.append(node)
                if max_count and len(result) >= max_count:
                    break
        return result

    def extract_subgraph(self, center_node, max_nodes, output_path, directed=True):
        """Extract a BFS subgraph of up to `max_nodes` around `center_node`.

        Returns a new NetworkXGraph, or None if the center node is isolated.
        """
        if center_node not in self.graph:
            raise ValueError(f"Center node {center_node} not in graph")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.graph.degree(center_node) == 0:
            print(f"Center node {center_node} is an isolate (no edges). Extraction stopped.")
            return None

        # BFS to find the closest nodes
        visited = {center_node}
        queue = deque([center_node])
        extracted_nodes = [center_node]
        while queue and len(extracted_nodes) < max_nodes:
            current = queue.popleft()
            if directed:
                neighbors = set(self.graph.successors(current))
            else:
                neighbors = (set(self.graph.successors(current))
                             | set(self.graph.predecessors(current)))
            for neighbor in neighbors:
                if neighbor not in visited and len(extracted_nodes) < max_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    extracted_nodes.append(neighbor)

        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        extracted_set = set(extracted_nodes)
        subgraph = NetworkXGraph(output_path)
        subgraph.graph = self.graph.subgraph(extracted_nodes).copy()
        subgraph.attrs_to_group = self.attrs_to_group.copy()
        subgraph.group_to_attrs = self.group_to_attrs.copy()
        subgraph.existing_num_links = self.existing_num_links.copy()
        subgraph.maximum_num_links = self.maximum_num_links.copy()
        subgraph.group_to_nodes = {
            gid: [n for n in nodes if n in extracted_set]
            for gid, nodes in self.group_to_nodes.items()
            if any(n in extracted_set for n in nodes)
        }
        subgraph.nodes_to_group = {n: gid for n, gid in self.nodes_to_group.items()
                                   if n in extracted_set}

        subgraph.finalize()
        return subgraph

    def finalize(self):
        """Memory-efficient compressed save of the graph and node attributes."""
        base_path = self.graph_file.replace('.gpickle', '')
        node_attrs = dict(self.graph.nodes(data=True))
        np.savez_compressed(
            f'{base_path}.npz',
            edges=np.array(self.graph.edges(), dtype=np.uint32),
            nodes=np.array(list(self.graph.nodes()), dtype=np.uint32),
            node_attrs=np.array([node_attrs], dtype=object)[0],
            num_nodes=self.graph.number_of_nodes(),
            directed=self.graph.is_directed(),
        )