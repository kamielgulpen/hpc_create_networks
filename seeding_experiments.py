"""
Parameter sweep for contagion analysis on enriched networks.

Each task runs one (n_communities, pref_attachment) combination across all
threshold values and saves results to results/parameter_sweep/task_{id}.csv.

Usage:
    python seeding_experiments.py --task_id N

To parse final values from results:
    final_values = np.array(row['final_values'].split(','), dtype=int)
"""

import argparse
import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numba
import networkx as nx
import numpy as np
import pandas as pd


# =============================================================================
# ContagionSimulator
# =============================================================================

@numba.njit(parallel=True, cache=True)
def _complex_contagion_kernel(data, indices, indptr, degree, state, threshold,
                               is_fractional, max_steps, verbose=0):
    n, n_sims = state.shape
    infected_counts = np.empty((n, n_sims), dtype=np.float32)
    time_series     = np.empty((max_steps + 1, n_sims), dtype=np.int64)

    # ── NEW: track infection time per node per sim ────────────────────────────
    infection_time = np.full((n, n_sims), -1, dtype=np.int16)  # -1 = never
    # Record seeds (already infected at step 0)
    for i in numba.prange(n):
        for s in range(n_sims):
            if state[i, s] == 1:
                infection_time[i, s] = 0
    # ─────────────────────────────────────────────────────────────────────────

    current_counts = np.sum(state > 0, axis=0).astype(np.int64)
    time_series[0, :] = current_counts
    active = np.ones(n_sims, dtype=numba.boolean)
    actual_steps = 0

    for step in range(max_steps):
        for i in numba.prange(n):
            row_start = indptr[i]
            row_end   = indptr[i + 1]
            for s in range(n_sims):
                if active[s] and state[i, s] == 0:
                    val = 0.0
                    for j_ptr in range(row_start, row_end):
                        val += data[j_ptr] * state[indices[j_ptr], s]
                    infected_counts[i, s] = val

        delta = np.zeros((n, n_sims), dtype=np.int8)
        for i in numba.prange(n):
            d = degree[i]
            for s in range(n_sims):
                if active[s] and state[i, s] == 0:
                    ic = infected_counts[i, s]
                    meets = ((d > 0.0) and (ic / d >= threshold)) if is_fractional else (ic >= threshold)
                    if meets:
                        state[i, s]          = 1
                        delta[i, s]          = 1
                        infection_time[i, s] = step + 1   # ← record step

        prev_counts    = current_counts.copy()
        current_counts = current_counts + np.sum(delta, axis=0).astype(np.int64)
        time_series[step + 1, :] = current_counts
        actual_steps += 1

        any_active = False
        for s in range(n_sims):
            if active[s]:
                if current_counts[s] == n or current_counts[s] == prev_counts[s]:
                    active[s] = False
                else:
                    any_active = True
        if not any_active:
            break

    return time_series[:actual_steps + 1], infection_time 

class ContagionSimulator:
    """Simulates contagion spreading on networks using vectorized sparse operations."""

    def __init__(self, network, name="Network"):
        if isinstance(network, dict):
            self.name   = network.get('name', name)
            self.n      = network['n']
            self.adj    = network['adj']
            self.degree = network['degree']  # kept for kernel compatibility
        else:
            self.name   = name
            self.n      = len(network)
            self.adj    = nx.to_scipy_sparse_array(network, format='csr', dtype=np.float32)
            self.degree = np.array(self.adj.sum(axis=1)).flatten()

        # Explicit directional degrees — works for both directed and undirected
        self.out_degree = np.array(self.adj.sum(axis=1)).flatten().astype(np.int32)  # row sums
        self.in_degree  = np.array(self.adj.sum(axis=0)).flatten().astype(np.int32)  # col sums


    def _seed_state(self, state, n_simulations, seeding, initial_infected,
                base_seed=0, neighbor_k=None):
        """
        Seed the initial state.

        Seeding modes:
            'random'          — N random nodes
            'focal_neighbors' — 1 focal node + ALL its neighbours (original)
            'neighbor_k'      — 1 focal node + exactly K of its neighbours
                                (falls back to all neighbours if degree < K)
        """
        if isinstance(seeding, np.ndarray):
            for sim in range(n_simulations):
                rng = np.random.RandomState(base_seed + sim)
                nodes = rng.choice(seeding, initial_infected, replace=False)
                state[nodes, sim] = 1

        elif seeding == 'focal_neighbors':
            for sim in range(n_simulations):
                rng = np.random.RandomState(base_seed + sim)
                focal = rng.randint(self.n)
                state[focal, sim] = 1
                neighbours = self.adj.indices[
                    self.adj.indptr[focal]:self.adj.indptr[focal + 1]
                ]
                state[neighbours, sim] = 1

        elif seeding == 'neighbor_k':
            if neighbor_k is None:
                raise ValueError("neighbor_k seeding requires neighbor_k parameter")
            
            # Pre-compute eligible focal nodes (degree >= neighbor_k) — do once outside sim loop
            degree_arr = np.array(self.adj.sum(axis=1)).flatten().astype(np.int32)
            eligible = np.where(degree_arr >= neighbor_k)[0]
            
            if len(eligible) == 0:
                raise ValueError(
                    f"No nodes with degree >= {neighbor_k}. "
                    f"Max degree in network: {degree_arr.max()}"
                )
            
            for sim in range(n_simulations):
                rng = np.random.RandomState(base_seed + sim)
                
                # Sample focal node only from nodes with enough neighbours
                focal = rng.choice(eligible)
                state[focal, sim] = 1
                
                neighbours = self.adj.indices[
                    self.adj.indptr[focal]:self.adj.indptr[focal + 1]
                ]
                # Sample exactly neighbor_k from the guaranteed >= neighbor_k pool
                chosen = rng.choice(neighbours, neighbor_k, replace=False)
                state[chosen, sim] = 1

        else:  # 'random'
            for sim in range(n_simulations):
                rng = np.random.RandomState(base_seed + sim)
                nodes = rng.choice(self.n, initial_infected, replace=False)
                state[nodes, sim] = 1

    def complex_contagion_kernel(self, threshold=2, threshold_type='absolute',
                             initial_infected=1, max_steps=1000, n_simulations=1,
                             seeding='random', base_seed=0, neighbor_k=None,  # ← new
                             verbose=0):
        """
        Args:
            ...
            neighbor_k: int, number of neighbours to seed when seeding='neighbor_k'.
                        If the focal node has fewer neighbours, all are seeded.
        """
        state = np.zeros((self.n, n_simulations), dtype=np.int8)
        self._seed_state(state, n_simulations, seeding, initial_infected,
                        base_seed, neighbor_k=neighbor_k)           # ← pass through

        is_fractional = (threshold_type != 'absolute')
        ts = _complex_contagion_kernel(
            self.adj.data, self.adj.indices, self.adj.indptr,
            self.degree, state, float(threshold), is_fractional, max_steps, verbose
        )
        return ts

# =============================================================================
# Network loader
# =============================================================================

def iter_networks(folder):
    """
    Yield (name, network_data) one at a time from subfolders of {folder}/enriched/.
    
    
    Returns:
        name: str, network identifier
        network_data: dict with keys:
            - 'n': int, number of nodes
            - 'adj': scipy.sparse.csr_matrix, adjacency matrix
            - 'degree': np.ndarray, degree of each node
    """
    from scipy import sparse
    
    folder = Path(folder)
    enriched_dir = folder / 'enriched'
    if not enriched_dir.exists():
        print(f"  No enriched/ dir found in {folder}")
        return
    
    for npz_file in sorted(enriched_dir.glob('*/graph.npz')):
        name = npz_file.parent.name
        data = np.load(npz_file, allow_pickle=True)
        
        nodes = data['nodes']
        edges = data['edges']
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        # Build sparse adjacency matrix directly (MUCH faster than NetworkX)
        # Check if nodes are already 0-indexed integers
        nodes_array = np.array(nodes) if not isinstance(nodes, np.ndarray) else nodes
        
        if np.all(nodes_array == np.arange(n_nodes)):
            # Nodes are already 0-indexed - no mapping needed!
            edges_array = np.array(edges) if not isinstance(edges, np.ndarray) else edges
            if edges_array.ndim == 2:
                row = edges_array[:, 0].astype(np.int32)
                col = edges_array[:, 1].astype(np.int32)
            else:
                row = np.array([e[0] for e in edges], dtype=np.int32)
                col = np.array([e[1] for e in edges], dtype=np.int32)
        else:
            # Need to create mapping
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            edges_array = np.array(edges) if not isinstance(edges, np.ndarray) else edges
            
            if edges_array.ndim == 2:
                row = np.array([node_to_idx[e] for e in edges_array[:, 0]], dtype=np.int32)
                col = np.array([node_to_idx[e] for e in edges_array[:, 1]], dtype=np.int32)
            else:
                row = np.array([node_to_idx[e[0]] for e in edges], dtype=np.int32)
                col = np.array([node_to_idx[e[1]] for e in edges], dtype=np.int32)
        
        
        edge_data = np.ones(n_edges, dtype=np.float32)
        
        # Create sparse matrix
        adj = sparse.csr_matrix(
            (edge_data, (row, col)), 
            shape=(n_nodes, n_nodes),
            dtype=np.float32  
        )
        
        # For undirected graphs, make symmetric
        directed = bool(data['directed'])
        if not directed:
            adj = adj + adj.T
            # Remove duplicate entries (self-loops counted twice)
            adj = adj.tocsr()
            adj.data[:] = 1.0  
        
        degree = np.array(adj.sum(axis=1)).flatten()
        
        network_data = {
            'n': n_nodes,
            'adj': adj,
            'degree': degree,
            'name': name
        }
        
        print(f"  Loaded {name} ({n_nodes:,} nodes, {n_edges:,} edges)")
        yield name, network_data


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    n_simulations: int   = 20
    max_steps: int       = 10000
    threshold_type: str  = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 0.30
    n_thresholds: int    = 3
    base_seed: int       = 0
    verbose: int         = 0
    seeding: str         = 'neighbor_k'   # ← default to new mode
    neighbor_k: int      = 20              # ← seed 1 focal + 5 neighbours
    
    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)

@dataclass
class NetworkConfig:
    base_folder: str = "pawn_results/networks"


def discover_network_folders(base_folder: str = "pawn_results/networks") -> List[Path]:
    """Return all parameter folders under base_folder that contain an enriched/ subfolder."""
    base = Path(base_folder)
    return sorted(f for f in base.iterdir() if f.is_dir() and (f / 'enriched').exists())


def parse_folder_params(folder: Path) -> dict:
    """Extract n_communities and pref_attachment from folder name."""
    params = {}
    for key, pattern in [('n_communities', r'comms=([^_]+)'), ('pref_attachment', r'pa=([^_]+)')]:
        m = re.search(pattern, folder.name)
        if m:
            params[key] = float(m.group(1))
    return params


# =============================================================================
# Parameter sweep
# =============================================================================

class ContagionAnalyzer:
    def __init__(self, sim_config: SimulationConfig = None, net_config: NetworkConfig = None):
        self.sim = sim_config or SimulationConfig()
        self.net = net_config or NetworkConfig()

    def run_task(self, task_id: int, output_dir: str = "results/parameter_sweep") -> Optional[pd.DataFrame]:
        folders = discover_network_folders(self.net.base_folder)
        if task_id >= len(folders):
            print(f"Task {task_id} out of range (only {len(folders)} folders). Exiting.")
            return None

        folder = folders[task_id]
        print(f"Task {task_id}/{len(folders) - 1}: {folder.name}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"task_{task_id}.csv"

        self._run_config(folder, out_path, task_id, out_file)

        if out_file.exists():
            return pd.read_csv(out_file)
        return None

    def _run_config(self, folder: Path, out_path: Path, task_id: int, out_file: Path) -> None:
        params = parse_folder_params(folder)

        # Resume: find which networks are already written
        done_networks: set = set()
        if out_file.exists():

            done_networks = set(pd.read_csv(
                out_file, 
                usecols=['network'],
                dtype={'network': 'str'}
            )['network'].unique())
            print(f"  Resuming — already done: {done_networks}")

        try:
            self._sweep_contested(folder, out_path, task_id, out_file, params, done_networks)
        except FileNotFoundError as e:
            print(f"  Network files not found in {folder.name}: {e}")
        except (ValueError, KeyError) as e:
            print(f"  Invalid data in {folder.name}: {e}")
        except Exception as e:
            print(f"  Unexpected error in {folder.name}: {e}")
            raise

    def _sweep_contested(self, folder: Path, out_path: Path, task_id: int,
                         out_file: Path, params: dict, done_networks: set) -> None:
        for name, network_data in iter_networks(str(folder)):
            if name in done_networks:
                print(f"  Skipping {name} (already saved).")
                continue

            ratio = None
            try:
                df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")
                ratio = df_n.n.max() / df_n.n.sum()
                del df_n
            except FileNotFoundError:
                pass
            
            # network_data is a dict with pre-computed sparse matrix - no conversion needed!
            sim = ContagionSimulator(network_data)
            del network_data  # Free memory
            initial = int(sim.n * self.sim.initial_infected_fraction)
            print(f"  Simulating {name} ({sim.n:,} nodes, {self.sim.n_simulations} sims)...", flush=True)

            rows = []
            for i, tau in enumerate(self.sim.thresholds):
                print(f"    threshold {i+1}/{len(self.sim.thresholds)} (tau={tau:.3f})", flush=True)
                ts_array = sim.complex_contagion_kernel(
                    threshold=tau,
                    threshold_type=self.sim.threshold_type,
                    seeding=self.sim.seeding,          # ← was hardcoded 'focal_neighbors'
                    neighbor_k=self.sim.neighbor_k,    # ← new
                    max_steps=self.sim.max_steps,
                    n_simulations=self.sim.n_simulations,
                    initial_infected=initial,
                    base_seed=self.sim.base_seed,
                    verbose=self.sim.verbose,
                )
                
                # Extract final values directly from numpy array
                final_values = ts_array[-1, :]  # Shape: (n_sims,) - last row = final counts
                
                rows.append({
                    **params,
                    'folder': folder.name,
                    'network': name,
                    'threshold_idx': i,
                    'threshold_value': tau,
                    'mean_final_adoption': final_values.mean(),  # NumPy method (faster)
                    'variance_final_adoption': final_values.var(),  # NumPy method (faster)
                    'final_values': ','.join(final_values.astype(np.int32).astype(str)),
                    'ratio': ratio,
                })
                del ts_array, final_values

            del sim

            # Append this network's rows to CSV immediately
            write_header = not out_file.exists()
            pd.DataFrame(rows).to_csv(out_file, mode='a', header=write_header, index=False)
            print(f"  Saved {len(rows)} rows for {name} to {out_file}")
            del rows

            gc.collect()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run one parameter sweep task")
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task index (0-based). Defaults to SLURM_ARRAY_TASK_ID env var.')
    parser.add_argument('--output_dir', type=str, default='results/parameter_sweep')
    parser.add_argument('--list_tasks', action='store_true',
                        help='Print the number of discovered network folders and exit.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity: 0=silent (fastest), 1=summary, 2=detailed')
    args = parser.parse_args()

    net_config = NetworkConfig()

    if args.list_tasks:
        folders = discover_network_folders(net_config.base_folder)
        print(len(folders))
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    sim_config = SimulationConfig(verbose=args.verbose)
    analyzer = ContagionAnalyzer(sim_config=sim_config, net_config=net_config)
    analyzer.run_task(task_id, output_dir=args.output_dir)


if __name__ == '__main__':
    main()