"""
Parameter sweep for contagion analysis on enriched networks.

Each task runs one sample folder across all threshold values and saves
results to results/parameter_sweep/task_{id}.csv.

Usage:
    python seeding_experiments.py --task_id N
    python seeding_experiments.py --list_tasks

To parse final values from results:
    final_values = np.array(row['final_values'].split(','), dtype=int)

Directory structure expected:
    {base_folder}/sample_00025/etngrp_geslacht/params.json
    {base_folder}/sample_00025/etngrp_geslacht/gen/graph.npz
"""

import argparse
import gc
import json
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
# Numba kernel
# =============================================================================

@numba.njit(parallel=True, cache=True)
def _complex_contagion_kernel(data, indices, indptr, degree, state, threshold,
                               is_fractional, max_steps, verbose=0):
    """
    JIT-compiled contagion kernel.

    Optimizations:
      - Skip already-infected nodes in the matmul
      - Per-simulation early exit once converged
      - Incremental count updates (no full O(n*n_sims) scan each step)
      - int8 state (4x less memory bandwidth than float32)

    Args:
        data, indices, indptr : CSR sparse adjacency matrix components
        degree                : (n,) out-degree array
        state                 : (n, n_sims) int8 — modified in-place
        threshold             : adoption threshold value
        is_fractional         : True = fractional threshold, False = absolute
        max_steps             : maximum simulation steps
        verbose               : 0=silent, 1=summary, 2=detailed

    Returns:
        time_series : (actual_steps+1, n_sims) int64 array of infected counts
    """
    n, n_sims = state.shape
    infected_counts = np.empty((n, n_sims), dtype=np.float32)
    time_series     = np.empty((max_steps + 1, n_sims), dtype=np.int64)

    current_counts = np.sum(state > 0, axis=0).astype(np.int64)
    time_series[0, :] = current_counts

    active = np.ones(n_sims, dtype=numba.boolean)
    actual_steps = 0

    for step in range(max_steps):

        # ── Pass 1: sparse matmul — skip already-infected nodes ───────────────
        for i in numba.prange(n):
            row_start = indptr[i]
            row_end   = indptr[i + 1]
            for s in range(n_sims):
                if active[s] and state[i, s] == 0:
                    val = 0.0
                    for j_ptr in range(row_start, row_end):
                        val += data[j_ptr] * state[indices[j_ptr], s]
                    infected_counts[i, s] = val

        # ── Pass 2: threshold check + state update ────────────────────────────
        delta = np.zeros((n, n_sims), dtype=np.int8)
        for i in numba.prange(n):
            d = degree[i]
            for s in range(n_sims):
                if active[s] and state[i, s] == 0:
                    ic = infected_counts[i, s]
                    if is_fractional:
                        meets = (d > 0.0) and (ic / d >= threshold)
                    else:
                        meets = ic >= threshold
                    if meets:
                        state[i, s] = 1
                        delta[i, s] = 1

        # ── Incremental count update ──────────────────────────────────────────
        prev_counts    = current_counts.copy()
        current_counts = current_counts + np.sum(delta, axis=0).astype(np.int64)
        time_series[step + 1, :] = current_counts
        actual_steps += 1

        # ── Per-sim convergence check ─────────────────────────────────────────
        any_active = False
        for s in range(n_sims):
            if active[s]:
                if current_counts[s] == n or current_counts[s] == prev_counts[s]:
                    active[s] = False
                else:
                    any_active = True
        if not any_active:
            break

    # ── Verbose logging ───────────────────────────────────────────────────────
    if verbose >= 1:
        final_counts  = time_series[actual_steps, :]
        converged     = np.sum((final_counts == n) |
                               (final_counts == time_series[actual_steps - 1, :]))
        full_cascades = np.sum(final_counts == n)
        stalled       = converged - full_cascades
        print("    → Steps:", actual_steps, "/", max_steps,
              "| Converged:", converged, "/", n_sims,
              "| Full:", full_cascades, "| Stalled:", stalled)
        if verbose >= 2:
            print("    → Adoption: mean=", round(final_counts.mean(), 1),
                  "std=", round(final_counts.std(), 1),
                  "range=[", int(final_counts.min()), ",", int(final_counts.max()), "]")

    return time_series[:actual_steps + 1]


# =============================================================================
# ContagionSimulator
# =============================================================================

class ContagionSimulator:
    """Simulates complex contagion spreading on networks using a Numba kernel."""

    def __init__(self, network, name="Network"):
        """
        Args:
            network : NetworkX graph OR dict with keys {n, adj, degree, name, params}
            name    : fallback name if network is a NetworkX graph
        """
        if isinstance(network, dict):
            self.name   = network.get('name', name)
            self.n      = network['n']
            self.adj    = network['adj']
            self.degree = network['degree']
            self.params = network.get('params', {})
        else:
            self.name   = name
            self.n      = len(network)
            self.adj    = nx.to_scipy_sparse_array(network, format='csr', dtype=np.float32)
            self.degree = np.array(self.adj.sum(axis=1)).flatten()
            self.params = {}

        # Explicit out/in degree (row sums = out, col sums = in)
        self.out_degree = np.array(self.adj.sum(axis=1)).flatten().astype(np.int32)
        self.in_degree  = np.array(self.adj.sum(axis=0)).flatten().astype(np.int32)

        # Populated by _seed_state for downstream analysis
        self.last_focal_nodes: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------

    def _seed_state(self, state, n_simulations, seeding, initial_infected,
                    base_seed=0, neighbor_k=None):
        """
        Populate initial infection state matrix.

        Modes:
            'random'          — initial_infected random nodes per sim
            'focal_neighbors' — 1 focal node + ALL its out-neighbours
            'neighbor_k'      — 1 focal node (out-degree >= k) + exactly k
                                randomly chosen out-neighbours
        """
        focal_nodes = np.full(n_simulations, -1, dtype=np.int32)

        if isinstance(seeding, np.ndarray):
            for sim in range(n_simulations):
                rng   = np.random.RandomState(base_seed + sim)
                nodes = rng.choice(seeding, initial_infected, replace=False)
                state[nodes, sim] = 1

        elif seeding == 'focal_neighbors':
            for sim in range(n_simulations):
                rng   = np.random.RandomState(base_seed + sim)
                focal = rng.randint(self.n)
                focal_nodes[sim]  = focal
                state[focal, sim] = 1
                neighbours = self.adj.indices[
                    self.adj.indptr[focal]:self.adj.indptr[focal + 1]
                ]
                state[neighbours, sim] = 1

        elif seeding == 'neighbor_k':
            if neighbor_k is None:
                raise ValueError("seeding='neighbor_k' requires neighbor_k parameter")

            # Only nodes with out-degree >= k are eligible as focal node
            eligible = np.where(self.out_degree >= neighbor_k)[0]
            if len(eligible) == 0:
                raise ValueError(
                    f"No nodes with out-degree >= {neighbor_k}. "
                    f"Max out-degree: {self.out_degree.max()}"
                )

            for sim in range(n_simulations):
                rng   = np.random.RandomState(base_seed + sim)
                focal = rng.choice(eligible)
                focal_nodes[sim]  = focal
                state[focal, sim] = 1

                # Out-neighbours of focal (CSR row = out-edges)
                out_neighbours = self.adj.indices[
                    self.adj.indptr[focal]:self.adj.indptr[focal + 1]
                ]
                chosen = rng.choice(out_neighbours, neighbor_k, replace=False)
                state[chosen, sim] = 1

        else:  # 'random'
            for sim in range(n_simulations):
                rng   = np.random.RandomState(base_seed + sim)
                nodes = rng.choice(self.n, initial_infected, replace=False)
                state[nodes, sim] = 1

        self.last_focal_nodes = focal_nodes

    # -------------------------------------------------------------------------
    # Simulation entry point
    # -------------------------------------------------------------------------

    def run(self, threshold=2, threshold_type='absolute', initial_infected=1,
            max_steps=1000, n_simulations=1, seeding='random',
            base_seed=0, neighbor_k=None, verbose=0):
        """
        Run the complex contagion model.

        Args:
            threshold       : adoption threshold value
            threshold_type  : 'absolute' or 'fractional'
            initial_infected: seed size for seeding='random'
            max_steps       : maximum steps before stopping
            n_simulations   : number of parallel simulations
            seeding         : 'random', 'focal_neighbors', or 'neighbor_k'
            base_seed       : RNG base seed for reproducibility
            neighbor_k      : number of out-neighbours to seed (seeding='neighbor_k')
            verbose         : 0=silent, 1=summary, 2=detailed

        Returns:
            np.ndarray of shape (actual_steps+1, n_sims) — infected count per step
        """
        state = np.zeros((self.n, n_simulations), dtype=np.int8)
        self._seed_state(state, n_simulations, seeding, initial_infected,
                         base_seed, neighbor_k=neighbor_k)

        is_fractional = (threshold_type != 'absolute')
        return _complex_contagion_kernel(
            self.adj.data, self.adj.indices, self.adj.indptr,
            self.degree, state, float(threshold),
            is_fractional, max_steps, verbose
        )


# =============================================================================
# Network loader
# =============================================================================

def iter_networks(sample_folder):
    """
    Yield (name, network_data) for every network inside a sample folder.

    Expected layout:
        sample_NNNNN/
        └── {network_label}/          e.g. etngrp_geslacht
            ├── params.json
            └── gen/
                └── graph.npz

    Yields:
        name         : str  — network label (e.g. 'etngrp_geslacht')
        network_data : dict — {n, adj, degree, name, params}
    """
    from scipy import sparse

    sample_folder = Path(sample_folder)

    for npz_file in sorted(sample_folder.glob('*/gen/graph.npz')):
        network_dir = npz_file.parent.parent   # .../etngrp_geslacht/
        name        = network_dir.name         # 'etngrp_geslacht'

        # ── Load params.json ──────────────────────────────────────────────────
        params_file = network_dir / 'params.json'
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
        else:
            params = {'label': name}
            print(f"  Warning: no params.json in {network_dir}")

        # ── Load graph ────────────────────────────────────────────────────────
        data    = np.load(npz_file, allow_pickle=True)
        nodes   = data['nodes']
        edges   = data['edges']
        n_nodes = len(nodes)
        n_edges = len(edges)

        nodes_array = np.array(nodes) if not isinstance(nodes, np.ndarray) else nodes

        if np.all(nodes_array == np.arange(n_nodes)):
            edges_array = np.array(edges) if not isinstance(edges, np.ndarray) else edges
            if edges_array.ndim == 2:
                row = edges_array[:, 0].astype(np.int32)
                col = edges_array[:, 1].astype(np.int32)
            else:
                row = np.array([e[0] for e in edges], dtype=np.int32)
                col = np.array([e[1] for e in edges], dtype=np.int32)
        else:
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            edges_array = np.array(edges) if not isinstance(edges, np.ndarray) else edges
            if edges_array.ndim == 2:
                row = np.array([node_to_idx[e] for e in edges_array[:, 0]], dtype=np.int32)
                col = np.array([node_to_idx[e] for e in edges_array[:, 1]], dtype=np.int32)
            else:
                row = np.array([node_to_idx[e[0]] for e in edges], dtype=np.int32)
                col = np.array([node_to_idx[e[1]] for e in edges], dtype=np.int32)

        edge_data = np.ones(n_edges, dtype=np.float32)
        adj = sparse.csr_matrix(
            (edge_data, (row, col)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32
        )

        # Make undirected graphs symmetric
        directed = bool(data['directed'])
        if not directed:
            adj      = (adj + adj.T).tocsr()
            adj.data[:] = 1.0

        degree = np.array(adj.sum(axis=1)).flatten()

        network_data = {
            'n':      n_nodes,
            'adj':    adj,
            'degree': degree,
            'name':   name,
            'params': params,
        }

        print(f"  Loaded {name} ({n_nodes:,} nodes, {n_edges:,} edges)")
        yield name, network_data


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    n_simulations:            int   = 20
    max_steps:                int   = 10000
    threshold_type:           str   = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold:            float = 0.05
    max_threshold:            float = 0.30
    n_thresholds:             int   = 3
    base_seed:                int   = 0
    verbose:                  int   = 0
    seeding:                  str   = 'neighbor_k'
    neighbor_k:               int   = 20     # focal node must have >= 20 out-neighbours

    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)


@dataclass
class NetworkConfig:
    base_folder: str = "/home/kamiel/practice/hpc_create_networks/results/networks"


def discover_sample_folders(base_folder: str) -> List[Path]:
    """Return all sample_NNNNN folders sorted numerically (sample_00000 … sample_00248)."""
    base = Path(base_folder)
    return sorted(
        f for f in base.iterdir()
        if f.is_dir() and f.name.startswith('sample_')
    )


# =============================================================================
# Analyzer
# =============================================================================

class ContagionAnalyzer:

    def __init__(self, sim_config: SimulationConfig = None,
                 net_config: NetworkConfig = None):
        self.sim = sim_config or SimulationConfig()
        self.net = net_config or NetworkConfig()

    # -------------------------------------------------------------------------

    def run_task(self, task_id: int,
                 output_dir: str = "results/parameter_sweep") -> Optional[pd.DataFrame]:
        """Run one task (= one sample folder)."""
        folders = discover_sample_folders(self.net.base_folder)

        if task_id >= len(folders):
            print(f"Task {task_id} out of range (only {len(folders)} folders). Exiting.")
            return None

        folder = folders[task_id]
        print(f"Task {task_id}/{len(folders) - 1}: {folder.name}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"task_{task_id}.csv"

        self._run_sample(folder, out_file)

        return pd.read_csv(out_file) if out_file.exists() else None

    # -------------------------------------------------------------------------

    def _run_sample(self, folder: Path, out_file: Path) -> None:
        """Sweep all networks inside one sample folder."""

        # Resume: skip networks already written to CSV
        done_networks: set = set()
        if out_file.exists():
            done_networks = set(
                pd.read_csv(out_file, usecols=['network'],
                            dtype={'network': str})['network'].unique()
            )
            print(f"  Resuming — already done: {done_networks}")

        try:
            self._sweep(folder, out_file, done_networks)
        except FileNotFoundError as e:
            print(f"  Network files not found in {folder.name}: {e}")
        except (ValueError, KeyError) as e:
            print(f"  Invalid data in {folder.name}: {e}")
        except Exception as e:
            print(f"  Unexpected error in {folder.name}: {e}")
            raise

    # -------------------------------------------------------------------------

    def _sweep(self, folder: Path, out_file: Path, done_networks: set) -> None:
        """Run the threshold sweep over every network in the sample folder."""

        for name, network_data in iter_networks(str(folder)):
            if name in done_networks:
                print(f"  Skipping {name} (already saved).")
                continue

            params = network_data.get('params', {})
            sim    = ContagionSimulator(network_data)
            del network_data

            initial = max(1, int(sim.n * self.sim.initial_infected_fraction))
            print(f"  Simulating {name} ({sim.n:,} nodes, "
                  f"{self.sim.n_simulations} sims, "
                  f"seeding={self.sim.seeding}, k={self.sim.neighbor_k})...",
                  flush=True)

            rows = []
            for i, tau in enumerate(self.sim.thresholds):
                print(f"    threshold {i+1}/{len(self.sim.thresholds)} "
                      f"(tau={tau:.3f})", flush=True)

                ts_array = sim.run(
                    threshold         = tau,
                    threshold_type    = self.sim.threshold_type,
                    initial_infected  = initial,
                    max_steps         = self.sim.max_steps,
                    n_simulations     = self.sim.n_simulations,
                    seeding           = self.sim.seeding,
                    neighbor_k        = self.sim.neighbor_k,
                    base_seed         = self.sim.base_seed,
                    verbose           = self.sim.verbose,
                )

                final_values = ts_array[-1, :]   # (n_sims,)

                rows.append({
                    # ── network identity ──────────────────────────────────────
                    'sample_id':          params.get('sample_id'),
                    'label':              params.get('label', name),
                    'pref_attachment':    params.get('pref_attachment'),
                    'n_communities':      params.get('n_communities'),
                    'transitivity_param': params.get('transitivity_param'),
                    'bridge_probability': params.get('bridge_probability'),
                    'n_nodes':            params.get('nodes', sim.n),
                    # ── simulation config ─────────────────────────────────────
                    'network':            name,
                    'threshold_idx':      i,
                    'threshold_value':    tau,
                    'seeding':            self.sim.seeding,
                    'neighbor_k':         self.sim.neighbor_k,
                    # ── results ───────────────────────────────────────────────
                    'mean_final_adoption':     float(final_values.mean()),
                    'variance_final_adoption': float(final_values.var()),
                    'final_values': ','.join(final_values.astype(np.int32).astype(str)),
                })
                del ts_array, final_values

            del sim

            write_header = not out_file.exists()
            pd.DataFrame(rows).to_csv(out_file, mode='a',
                                       header=write_header, index=False)
            print(f"  Saved {len(rows)} rows for {name} → {out_file}")
            del rows
            gc.collect()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Contagion parameter sweep — one task per sample folder"
    )
    parser.add_argument(
        '--task_id', type=int, default=None,
        help='Task index (0-based). Falls back to $SLURM_ARRAY_TASK_ID.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/parameter_sweep'
    )
    parser.add_argument(
        '--list_tasks', action='store_true',
        help='Print number of discovered sample folders and exit.'
    )
    parser.add_argument(
        '--verbose', type=int, default=0,
        help='0=silent, 1=summary, 2=detailed'
    )
    parser.add_argument(
        '--seeding', type=str, default='neighbor_k',
        choices=['random', 'focal_neighbors', 'neighbor_k'],
        help='Seeding strategy'
    )
    parser.add_argument(
        '--neighbor_k', type=int, default=30,
        help='Out-neighbours to seed when --seeding=neighbor_k'
    )
    parser.add_argument(
        '--n_sims', type=int, default=200,
        help='Number of simulations per threshold'
    )
    args = parser.parse_args()

    net_config = NetworkConfig()

    # seeding_experiments.py — change this line in main()
    if args.list_tasks:
        folders = discover_sample_folders(net_config.base_folder)
        print(len(folders))   # ← just the number, no extra text
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set $SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    sim_config = SimulationConfig(
        verbose      = args.verbose,
        seeding      = args.seeding,
        neighbor_k   = args.neighbor_k,
        n_simulations= args.n_sims,
    )

    analyzer = ContagionAnalyzer(sim_config=sim_config, net_config=net_config)
    analyzer.run_task(task_id, output_dir=args.output_dir)


if __name__ == '__main__':
    main()