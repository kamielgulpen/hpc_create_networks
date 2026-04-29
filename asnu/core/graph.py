"""
Minimal wrapper around NetworkX for ASNU population-based network generation.

This module provides a thin NetworkXGraph wrapper that adds metadata tracking
for population-based generation while using NetworkX DiGraph directly for all
graph operations.

Classes
-------
NetworkXGraph : Thin wrapper around nx.DiGraph with metadata for generation

Notes
-----
This is just NetworkX + metadata tracking. For all graph operations, the
underlying nx.DiGraph is used directly. Use G.graph to access NetworkX methods.

Examples
--------
>>> from asnu import NetworkXGraph
>>> G = NetworkXGraph("my_graph")
>>> G.graph.add_node(0, group="A", age=25)  # Use G.graph for NetworkX operations
>>> G.graph.add_edge(0, 1)
>>> G.finalize()  # Save metadata
"""

import networkx as nx
import numpy as np
import json
import os


class NetworkXGraph:
    """
    Minimal wrapper around nx.DiGraph for population network generation.

    Just adds metadata tracking for population groups. All graph operations
    use the underlying nx.DiGraph directly.

    Parameters
    ----------
    base_path : str, optional
        Directory for saving metadata. Default is "graph_data".

    Attributes
    ----------
    graph : nx.DiGraph
        The NetworkX directed graph - use this directly!
    base_path : str
        Directory for metadata storage
    attrs_to_group : dict
        Mapping from attribute tuples to group IDs (for generation)
    group_to_attrs : dict
        Mapping from group IDs to attributes (for generation)
    group_to_nodes : dict
        Mapping from group IDs to node lists (for generation)
    nodes_to_group : dict
        Mapping from node IDs to group IDs (for generation)
    existing_num_links : dict
        Link count tracking between groups (for generation)
    maximum_num_links : dict
        Maximum allowed links between groups (for generation)
    """

    def __init__(self, base_path="graph_data"):
        """Initialize with a NetworkX DiGraph and metadata tracking."""
        self.base_path = base_path
        self.metadata_file = os.path.join(base_path, "metadata.json")
        self.graph_file = os.path.join(base_path, "graph.gpickle")

        os.makedirs(base_path, exist_ok=True)

        # The actual NetworkX graph - use this directly!
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
        self.probability_matrix = np.zeros((2,3))
        self.number_of_communities = 0

        # Group pairs that community structure cannot serve (handled by fill_unfulfilled)
        self.infeasible_pairs = set()

        # Global popularity pool for preferential attachment with memory
        # Keyed by (community_id, group_id) -> list of nodes (with duplicates for weighting)
        self.popularity_pool = {}

        # self._load_metadata()

    def _save_metadata(self):
        """Save generation metadata to JSON."""
        metadata = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'attrs_to_group': {str(k): v for k, v in self.attrs_to_group.items()},
            'group_to_attrs': self.group_to_attrs,
            'group_to_nodes': self.group_to_nodes,
            'nodes_to_group': {str(k): v for k, v in self.nodes_to_group.items()},
            'existing_num_links': {str(k): v for k, v in self.existing_num_links.items()},
            'maximum_num_links': {str(k): v for k, v in self.maximum_num_links.items()}
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_graph_npz(self, path=None):
        """Load graph from compressed .npz file"""
        if path is None:
            path = self.graph_file.replace('.gpickle', '.npz')
        
        # Load compressed archive
        data = np.load(path, allow_pickle=True)
        
        # Create graph
        directed = bool(data['directed'])
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Add edges
        edges = data['edges']
        G.add_edges_from(edges)
        
        # Restore node attributes
        node_attrs = data['node_attrs'].item()  # .item() unpacks the object array
        nx.set_node_attributes(G, node_attrs)
        
        # # Restore metadata if present
        # if 'metadata' in data:
        #     metadata = pickle.loads(data['metadata'].tobytes())
        #     G.graph.update(metadata)
        
        return G

    def to_networkx(self):
        """
        Get the underlying NetworkX DiGraph.

        Returns the actual nx.DiGraph object, which you can modify directly.
        Any changes will be reflected in the wrapper.

        Returns
        -------
        nx.DiGraph
            The underlying NetworkX directed graph

        Examples
        --------
        >>> G = NetworkXGraph("my_graph")
        >>> nx_graph = G.to_networkx()
        >>> # Modify the graph directly
        >>> nx_graph.add_node(999, custom_attr="value")
        >>> # Changes are reflected in G.graph
        >>> assert G.graph.has_node(999)
        """
        return self.graph

    def get_non_isolates_batch(self, node_list, max_count=None):
        """
        Efficiently find non-isolated nodes from a list of candidates.

        Filters a list of nodes to return only those with degree > 0.

        Parameters
        ----------
        node_list : list of int
            List of node IDs to check for isolation
        max_count : int, optional
            Maximum number of non-isolated nodes to return. If None, return all.

        Returns
        -------
        list of int
            List of non-isolated node IDs from the input list

        Examples
        --------
        >>> G = NetworkXGraph("my_graph")
        >>> # Find up to 100 non-isolated nodes from a candidate list
        >>> candidates = list(range(1000))
        >>> non_isolates = G.get_non_isolates_batch(candidates, max_count=100)
        >>> print(f"Found {len(non_isolates)} non-isolated nodes")
        """
        result = []
        for node in node_list:
            if node in self.graph and self.graph.degree(node) > 0:
                result.append(node)
                if max_count and len(result) >= max_count:
                    break
        return result

    def extract_subgraph(self, center_node, max_nodes, output_path, directed=True):
        """
        Extract a subgraph around a center node using BFS.

        Parameters
        ----------
        center_node : int
            The center node for subgraph extraction
        max_nodes : int
            Maximum number of nodes to extract (including center)
        output_path : str
            Path for the new NetworkXGraph directory
        directed : bool, optional
            Whether to treat graph as directed during BFS (default True)

        Returns
        -------
        NetworkXGraph
            New NetworkXGraph containing the extracted subgraph

        Examples
        --------
        >>> G = NetworkXGraph("large_network")
        >>> # Extract 1000 nearest nodes around node 100
        >>> subgraph = G.extract_subgraph(center_node=100, max_nodes=1000,
        ...                                output_path="subgraph_100")
        """
        from collections import deque

        if center_node not in self.graph:
            raise ValueError(f"Center node {center_node} not in graph")

        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

        # Check if center_node is isolated
        if self.graph.degree(center_node) == 0:
            print(f"Center node {center_node} is an isolate (no edges). Extraction stopped.")
            return None

        # BFS to find closest nodes
        visited = set()
        queue = deque([center_node])
        visited.add(center_node)
        extracted_nodes = [center_node]

        while queue and len(extracted_nodes) < max_nodes:
            current = queue.popleft()

            # Get neighbors based on directed flag
            if directed:
                neighbors = set(self.graph.successors(current))
            else:
                neighbors = set(self.graph.successors(current)) | set(self.graph.predecessors(current))

            # Add unvisited neighbors
            for neighbor in neighbors:
                if neighbor not in visited and len(extracted_nodes) < max_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    extracted_nodes.append(neighbor)

        # Create output directory
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Create new NetworkXGraph wrapper
        subgraph = NetworkXGraph(output_path)
        subgraph.graph = self.graph.subgraph(extracted_nodes).copy()

        # Copy metadata
        subgraph.attrs_to_group = self.attrs_to_group.copy()
        subgraph.group_to_attrs = self.group_to_attrs.copy()
        subgraph.existing_num_links = self.existing_num_links.copy()
        subgraph.maximum_num_links = self.maximum_num_links.copy()

        # Filter group_to_nodes for extracted nodes
        extracted_set = set(extracted_nodes)
        filtered_group_to_nodes = {}
        for group_id, node_list in self.group_to_nodes.items():
            filtered_nodes = [n for n in node_list if n in extracted_set]
            if filtered_nodes:
                filtered_group_to_nodes[group_id] = filtered_nodes
        subgraph.group_to_nodes = filtered_group_to_nodes

        # Filter nodes_to_group
        subgraph.nodes_to_group = {n: gid for n, gid in self.nodes_to_group.items() if n in extracted_set}

        subgraph.finalize()
        return subgraph

    def finalize(self):
        """Memory-efficient save for ASNU networks"""
        base_path = self.graph_file.replace('.gpickle', '')
        
        # Edges
        edges = np.array(self.graph.edges(), dtype=np.uint32)
        
        # Node attributes (ALL nodes)
        node_attrs = {
            node: data for node, data in self.graph.nodes(data=True)
        }
        
        # CRITICAL: Save list of ALL node IDs explicitly
        all_nodes = np.array(list(self.graph.nodes()), dtype=np.uint32)
        
        # Save compressed
        np.savez_compressed(
            f'{base_path}.npz',
            edges=edges,
            nodes=all_nodes,  # ADD THIS
            node_attrs=np.array([node_attrs], dtype=object)[0],
            num_nodes=self.graph.number_of_nodes(),
            directed=self.graph.is_directed()
        )

