"""
Memory-Efficient Stochastic Block Model for Large Networks
Handles networks with 10,000+ nodes using sparse matrices and edge lists
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MemoryEfficientSBM:
    """
    Memory-efficient Stochastic Block Model for large networks.
    Uses sparse matrices and edge list representations.
    """
    
    def __init__(self):
        self.population_df = None
        self.connections_df = None
        self.group_to_index = {}
        self.index_to_group = {}
        self.block_sizes = []
        self.prob_matrix = None
        self.adj_matrix = None  # Will be sparse
        self.edge_list = None   # Alternative to adjacency matrix
        self.n_blocks = 0
        self.n_nodes = 0
        self.blocks = None
        self.attribute_names = None
        
    def load_data(self, population_file, connections_file):
        """Load population and connections CSV files."""
        print("Loading data files...")
        
        self.population_df = pd.read_csv(population_file)
        print(f"  Population data: {len(self.population_df)} groups")
        
        self.connections_df = pd.read_csv(connections_file)
        print(f"  Connections data: {len(self.connections_df)} group pairs")
        
        self.attribute_names = [col for col in self.population_df.columns if col != 'n']
        print(f"  Group attributes: {self.attribute_names}")
        
        self._build_group_mappings()
        self._build_probability_matrix()
        
        # Estimate memory requirements
        self._estimate_memory()
        
        print(f"\nModel built: {self.n_blocks} groups, {self.n_nodes:,} total nodes")
        
    def _build_group_mappings(self):
        """Create bidirectional mapping between group attributes and block indices."""
        self.group_to_index = {}
        self.index_to_group = {}
        self.block_sizes = []
        
        pop_sorted = self.population_df.sort_values(by=self.attribute_names)
        
        for idx, row in pop_sorted.iterrows():
            group_key = tuple(row[attr] for attr in self.attribute_names)
            
            if group_key not in self.group_to_index:
                block_idx = len(self.group_to_index)
                self.group_to_index[group_key] = block_idx
                self.index_to_group[block_idx] = group_key
                self.block_sizes.append(int(row['n']))
        
        self.n_blocks = len(self.group_to_index)
        self.n_nodes = sum(self.block_sizes)
        self.blocks = np.repeat(range(self.n_blocks), self.block_sizes)
        
    def _build_probability_matrix(self):
        """Build probability matrix from connections data."""
        self.prob_matrix = np.zeros((self.n_blocks, self.n_blocks))
        
        src_attrs = [col for col in self.connections_df.columns if col.endswith('_src')]
        dst_attrs = [col for col in self.connections_df.columns if col.endswith('_dst')]
        
        for idx, row in self.connections_df.iterrows():
            src_group = tuple(row[col] for col in src_attrs)
            dst_group = tuple(row[col] for col in dst_attrs)
            
            if src_group in self.group_to_index and dst_group in self.group_to_index:
                src_idx = self.group_to_index[src_group]
                dst_idx = self.group_to_index[dst_group]
                
                prob = row['fn']
                self.prob_matrix[src_idx, dst_idx] = prob
                
                if src_idx != dst_idx:
                    self.prob_matrix[dst_idx, src_idx] = prob
        
        print(f"\nProbability matrix statistics:")
        print(f"  Non-zero entries: {np.count_nonzero(self.prob_matrix)} / {self.n_blocks**2}")
        print(f"  Mean probability: {np.mean(self.prob_matrix[self.prob_matrix > 0]):.6f}")
        print(f"  Max probability: {np.max(self.prob_matrix):.6f}")
    
    def _estimate_memory(self):
        """Estimate memory requirements and suggest strategy."""
        n = self.n_nodes
        
        # Dense adjacency matrix
        dense_gb = (n * n * 8) / (1024**3)  # 8 bytes per float64
        
        # Estimate expected edges
        expected_edges = 0
        for i in range(self.n_blocks):
            for j in range(i, self.n_blocks):
                n_i = self.block_sizes[i]
                n_j = self.block_sizes[j]
                prob = self.prob_matrix[i, j]
                
                if i == j:
                    expected_edges += (n_i * (n_i - 1) / 2) * prob
                else:
                    expected_edges += (n_i * n_j) * prob
        
        # Sparse matrix (COO format: 3 arrays of edge count size)
        sparse_gb = (expected_edges * 3 * 8) / (1024**3)
        
        # Edge list (2 columns)
        edgelist_gb = (expected_edges * 2 * 8) / (1024**3)
        
        print(f"\n{'='*70}")
        print("MEMORY ESTIMATES")
        print(f"{'='*70}")
        print(f"Network size: {n:,} nodes")
        print(f"Expected edges: ~{int(expected_edges):,}")
        print(f"Expected density: {expected_edges / (n * (n-1) / 2):.6f}")
        print(f"\nMemory requirements:")
        print(f"  Dense adjacency matrix:  {dense_gb:.2f} GB  {'❌ TOO LARGE' if dense_gb > 4 else '✓ OK'}")
        print(f"  Sparse matrix (CSR):     {sparse_gb:.2f} GB  {'❌ TOO LARGE' if sparse_gb > 4 else '✓ OK'}")
        print(f"  Edge list only:          {edgelist_gb:.2f} GB  ✓ RECOMMENDED")
        
        if dense_gb > 4:
            print(f"\n⚠️  WARNING: Dense matrix would require {dense_gb:.2f} GB")
            print("   Recommended approach: Use generate_edge_list() instead")
        
        return expected_edges, dense_gb, sparse_gb
    
    def generate_edge_list(self, seed=None, output_file=None, batch_size=100000):
        """
        Generate network as edge list (memory-efficient for large networks).
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        output_file : str, optional
            If provided, write edges to CSV file incrementally
        batch_size : int
            Write to file in batches of this size
            
        Returns:
        --------
        edge_list : list of tuples or None
            List of (source, target) edges, or None if writing to file
        """
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\nGenerating edge list for {self.n_nodes:,} nodes...")
        
        edges = []
        n_edges = 0
        
        # Calculate total pairs for progress tracking
        total_pairs = sum(
            self.block_sizes[i] * self.block_sizes[j] if i != j 
            else self.block_sizes[i] * (self.block_sizes[i] - 1) // 2
            for i in range(self.n_blocks) 
            for j in range(i, self.n_blocks)
        )
        
        pairs_processed = 0
        checkpoint = max(1, total_pairs // 20)
        
        # Process block by block
        node_offset = 0
        for block_i in range(self.n_blocks):
            size_i = self.block_sizes[block_i]
            
            # Get node indices for block i
            nodes_i = list(range(node_offset, node_offset + size_i))
            
            # Process connections to this block and subsequent blocks
            node_offset_j = node_offset
            for block_j in range(block_i, self.n_blocks):
                size_j = self.block_sizes[block_j]
                prob = self.prob_matrix[block_i, block_j]
                
                if prob == 0:
                    node_offset_j += size_j
                    continue
                
                # Get node indices for block j
                nodes_j = list(range(node_offset_j, node_offset_j + size_j))
                
                # Generate edges
                if block_i == block_j:
                    # Within-block edges
                    for i_idx, i in enumerate(nodes_i):
                        for j in nodes_i[i_idx + 1:]:
                            if np.random.random() < prob:
                                edges.append((i, j))
                                n_edges += 1
                            
                            pairs_processed += 1
                            if pairs_processed % checkpoint == 0:
                                progress = 100 * pairs_processed / total_pairs
                                print(f"  Progress: {progress:.1f}% ({n_edges:,} edges so far)")
                                
                                # Write batch to file if specified
                                if output_file and len(edges) >= batch_size:
                                    self._write_edge_batch(edges, output_file, 
                                                          n_edges == len(edges))
                                    edges = []
                else:
                    # Between-block edges
                    for i in nodes_i:
                        for j in nodes_j:
                            if np.random.random() < prob:
                                edges.append((i, j))
                                n_edges += 1
                            
                            pairs_processed += 1
                            if pairs_processed % checkpoint == 0:
                                progress = 100 * pairs_processed / total_pairs
                                print(f"  Progress: {progress:.1f}% ({n_edges:,} edges so far)")
                                
                                # Write batch to file if specified
                                if output_file and len(edges) >= batch_size:
                                    self._write_edge_batch(edges, output_file, 
                                                          n_edges == len(edges))
                                    edges = []
                
                node_offset_j += size_j
            
            node_offset += size_i
        
        # Write remaining edges
        if output_file and edges:
            self._write_edge_batch(edges, output_file, n_edges == len(edges))
            print(f"\nEdge list written to {output_file}")
            print(f"Total edges: {n_edges:,}")
            return None
        
        print(f"\nGenerated {n_edges:,} edges")
        self.edge_list = edges
        return edges
    
    def _write_edge_batch(self, edges, filename, first_batch):
        """Write a batch of edges to file."""
        df = pd.DataFrame(edges, columns=['source', 'target'])
        mode = 'w' if first_batch else 'a'
        header = first_batch
        df.to_csv(filename, mode=mode, header=header, index=False)
    
    def generate_sparse_matrix(self, seed=None):
        """
        Generate network as sparse matrix (more memory-efficient than dense).
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
            
        Returns:
        --------
        adj_matrix : scipy.sparse.csr_matrix
            Sparse adjacency matrix
        """
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\nGenerating sparse adjacency matrix for {self.n_nodes:,} nodes...")
        
        # Use LIL format for efficient construction
        adj = sparse.lil_matrix((self.n_nodes, self.n_nodes), dtype=np.int8)
        
        n_edges = 0
        total_pairs = self.n_nodes * (self.n_nodes - 1) // 2
        checkpoint = max(1, total_pairs // 20)
        pairs_processed = 0
        
        for i in range(self.n_nodes):
            block_i = self.blocks[i]
            
            for j in range(i + 1, self.n_nodes):
                block_j = self.blocks[j]
                prob = self.prob_matrix[block_i, block_j]
                
                if prob > 0 and np.random.random() < prob:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    n_edges += 1
                
                pairs_processed += 1
                if pairs_processed % checkpoint == 0:
                    progress = 100 * pairs_processed / total_pairs
                    print(f"  Progress: {progress:.1f}% ({n_edges:,} edges so far)")
        
        # Convert to CSR format (more efficient for operations)
        self.adj_matrix = adj.tocsr()
        
        print(f"\nGenerated sparse matrix: {n_edges:,} edges")
        print(f"Memory usage: ~{self.adj_matrix.data.nbytes / (1024**2):.2f} MB")
        
        return self.adj_matrix
    
    def compute_statistics_from_edgelist(self, edge_list=None):
        """Compute statistics from edge list without creating adjacency matrix."""
        if edge_list is None:
            if self.edge_list is None:
                raise ValueError("No edge list available. Generate one first.")
            edge_list = self.edge_list
        
        n_edges = len(edge_list)
        density = n_edges / (self.n_nodes * (self.n_nodes - 1) / 2)
        
        # Compute degree distribution
        degrees = np.zeros(self.n_nodes)
        for src, dst in edge_list:
            degrees[src] += 1
            degrees[dst] += 1
        
        # Count within-group edges
        same_block_edges = sum(1 for src, dst in edge_list 
                              if self.blocks[src] == self.blocks[dst])
        homophily = same_block_edges / n_edges if n_edges > 0 else 0
        
        print("\n" + "="*70)
        print("NETWORK STATISTICS (from edge list)")
        print("="*70)
        print(f"Nodes: {self.n_nodes:,}")
        print(f"Edges: {n_edges:,}")
        print(f"Density: {density:.6f}")
        print(f"Average degree: {np.mean(degrees):.2f} ± {np.std(degrees):.2f}")
        print(f"Degree range: [{int(np.min(degrees))}, {int(np.max(degrees))}]")
        print(f"Within-group edges: {homophily:.1%}")
        
        return {
            'n_nodes': self.n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': np.mean(degrees),
            'std_degree': np.std(degrees),
            'homophily': homophily
        }
    
    def export_edge_list(self, filename, add_attributes=True):
        """
        Export edge list to CSV with optional node attributes.
        
        Parameters:
        -----------
        filename : str
            Output filename
        add_attributes : bool
            If True, create separate nodes file with attributes
        """
        if self.edge_list is None:
            raise ValueError("No edge list. Generate one first using generate_edge_list()")
        
        # Export edges
        edges_df = pd.DataFrame(self.edge_list, columns=['source', 'target'])
        edges_df.to_csv(filename, index=False)
        print(f"Edge list exported to {filename}")
        
        # Export nodes with attributes
        if add_attributes:
            nodes_file = filename.replace('.csv', '_nodes.csv')
            nodes_data = []
            
            for node_id in range(self.n_nodes):
                block_idx = self.blocks[node_id]
                group_attrs = self.index_to_group[block_idx]
                
                node_dict = {'id': node_id, 'block': block_idx}
                for attr_name, attr_val in zip(self.attribute_names, group_attrs):
                    node_dict[attr_name] = attr_val
                
                nodes_data.append(node_dict)
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_df.to_csv(nodes_file, index=False)
            print(f"Node attributes exported to {nodes_file}")
    
    def sample_network(self, sample_size=5000, seed=None):
        """
        Generate a sampled subnetwork (for visualization/analysis).
        
        Parameters:
        -----------
        sample_size : int
            Number of nodes to sample
        seed : int, optional
            Random seed
            
        Returns:
        --------
        sampled_edges : list
            Edge list for sampled network
        sampled_nodes : array
            Indices of sampled nodes
        """
        if seed is not None:
            np.random.seed(seed)
        
        sample_size = min(sample_size, self.n_nodes)
        sampled_nodes = np.sort(np.random.choice(self.n_nodes, sample_size, replace=False))
        node_to_idx = {node: idx for idx, node in enumerate(sampled_nodes)}
        
        print(f"\nSampling {sample_size:,} / {self.n_nodes:,} nodes...")
        
        sampled_edges = []
        for i_idx, i in enumerate(sampled_nodes):
            block_i = self.blocks[i]
            
            for j in sampled_nodes[i_idx + 1:]:
                block_j = self.blocks[j]
                prob = self.prob_matrix[block_i, block_j]
                
                if np.random.random() < prob:
                    sampled_edges.append((node_to_idx[i], node_to_idx[j]))
        
        print(f"Sampled network: {len(sampled_edges):,} edges")
        
        return sampled_edges, sampled_nodes
    
    def get_group_summary(self):
        """Print summary of groups."""
        print("\n" + "="*70)
        print("GROUP SUMMARY")
        print("="*70)
        
        for attr in self.attribute_names:
            unique_vals = self.population_df[attr].unique()
            print(f"\n{attr}: {len(unique_vals)} categories")
            
            counts = self.population_df.groupby(attr)['n'].sum().sort_values(ascending=False)
            for val, count in counts.head(10).items():
                pct = 100 * count / self.n_nodes
                print(f"  {val}: {count:,} ({pct:.1f}%)")


def main_example():
    """Example usage with memory-efficient methods."""
    print("="*70)
    print("MEMORY-EFFICIENT NETWORK GENERATION")
    print("="*70)
    
    # Load your data
    sbm = MemoryEfficientSBM()
    sbm.load_data(
        'data/tab_buren.csv',  # Replace with your file
        'data/tab_n_(with oplniv).csv'   # Replace with your file
    )
    
    # Get group summary
    sbm.get_group_summary()
    
    # Method 1: Generate edge list and save directly to file (most memory-efficient)
    print("\n--- METHOD 1: Direct edge list generation ---")
    sbm.generate_edge_list(seed=42, output_file='network_edges.csv')
    
    # Method 2: Generate edge list in memory (for smaller networks)
    # print("\n--- METHOD 2: Edge list in memory ---")
    # edges = sbm.generate_edge_list(seed=42)
    # stats = sbm.compute_statistics_from_edgelist(edges)
    # sbm.export_edge_list('network_edgelist.csv', add_attributes=True)
    
    # Method 3: Sample a smaller network for analysis
    print("\n--- METHOD 3: Sampled network ---")
    sampled_edges, sampled_nodes = sbm.sample_network(sample_size=5000, seed=42)
    
    print(f"\nSampled network: {len(sampled_nodes)} nodes, {len(sampled_edges)} edges")
    
    # Export sampled network
    sample_df = pd.DataFrame(sampled_edges, columns=['source', 'target'])
    sample_df.to_csv('network_sample_edges.csv', index=False)
    print("Sampled network exported to network_sample_edges.csv")


if __name__ == "__main__":
    print("\nUpdate file paths and run!")
    print("Use generate_edge_list() for networks with >10,000 nodes")
    print("Use sample_network() for quick analysis of large networks")
    main_example()