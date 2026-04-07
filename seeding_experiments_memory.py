"""
Parameter sweep for contagion analysis on enriched networks.

Each task runs one (n_communities, pref_attachment) combination across all
threshold values and saves results to results/parameter_sweep/task_{id}.csv.

Usage:
    python seeding_experiments_memory.py --task_id N
"""

import argparse
import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numba
import numpy as np
import pandas as pd


# =============================================================================
# Contagion kernel (JIT-compiled)
# =============================================================================

@numba.njit(parallel=True, cache=True)
def _complex_contagion_kernel(data, indices, indptr, degree, state, threshold, is_fractional, max_steps):
    """
    JIT-compiled contagion kernel with per-node infection time tracking.

    Returns:
        time_series:     (actual_steps+1, n_sims) int64 — infected counts per step
        infection_times: (n, n_sims) int64 — step at which each node was infected
                         (-1 = never infected, 0 = seeded before step 1)
    """
    n, n_sims = state.shape
    infected_counts = np.empty((n, n_sims), dtype=np.float64)
    time_series = np.empty((max_steps + 1, n_sims), dtype=np.int64)
    infection_times = np.full((n, n_sims), -1, dtype=np.int64)

    # Record seeds at time 0
    for s in range(n_sims):
        t = np.int64(0)
        for i in range(n):
            if state[i, s] > 0.0:
                t += np.int64(1)
                infection_times[i, s] = 0
        time_series[0, s] = t

    actual_steps = 0

    for step in range(max_steps):
        for i in numba.prange(n):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            for s in range(n_sims):
                val = 0.0
                for j_ptr in range(row_start, row_end):
                    val += data[j_ptr] * state[indices[j_ptr], s]
                infected_counts[i, s] = val

        for i in numba.prange(n):
            d = degree[i]
            for s in range(n_sims):
                if state[i, s] == 0.0:
                    ic = infected_counts[i, s]
                    if is_fractional:
                        meets = (d > 0.0) and (ic / d >= threshold)
                    else:
                        meets = ic >= threshold
                    if meets:
                        state[i, s] = 1.0
                        infection_times[i, s] = step + 1

        all_done = True
        converged = 0
        for s in range(n_sims):
            t = np.int64(0)
            for i in range(n):
                t += np.int64(state[i, s] > 0.0)
            time_series[step + 1, s] = t
            if t != np.int64(n) and t != time_series[step, s]:
                all_done = False
            if t == time_series[step, s] or t == np.int64(n):
                converged += 1
        actual_steps += 1
        if all_done:
            break

    print(f"converged = {converged}, out of{n_sims}, {actual_steps}")
    return time_series[:actual_steps + 1], infection_times


# =============================================================================
# ContagionSimulator
# =============================================================================

class ContagionSimulator:
    """Simulates contagion spreading on networks using vectorized sparse operations."""

    def __init__(self, network, name="Network"):
        self.G = network
        self.name = name
        self.n = len(network)
        adj = nx.to_scipy_sparse_array(network, format='csr', dtype=np.float64)
        self.adj = adj
        self.degree = np.array(self.adj.sum(axis=1)).flatten()

    def _seed_state(self, state, n_simulations, seeding, initial_infected):
        if isinstance(seeding, np.ndarray):
            for sim in range(n_simulations):
                nodes = np.random.choice(seeding, initial_infected, replace=False)
                state[nodes, sim] = 1.0
        elif seeding == 'focal_neighbors':
            for sim in range(n_simulations):
                focal = np.random.randint(self.n)
                state[focal, sim] = 1.0
                neighbors = self.adj.indices[self.adj.indptr[focal]:self.adj.indptr[focal + 1]]
                state[neighbors, sim] = 1.0
        else:  # 'random'
            for sim in range(n_simulations):
                nodes = np.random.choice(self.n, initial_infected, replace=False)
                state[nodes, sim] = 1.0

    def complex_contagion(self, threshold=2, threshold_type='absolute',
                          initial_infected=1, max_steps=1000, n_simulations=1,
                          seeding='random'):
        """
        Returns:
            time_series:     list of lists — infected counts per step per simulation
            infection_times: np.ndarray (n_nodes, n_sims), int64
                             step at which each node was infected (-1 = never, 0 = seeded)
        """
        state = np.zeros((self.n, n_simulations), dtype=np.float64)
        self._seed_state(state, n_simulations, seeding, initial_infected)
        is_fractional = (threshold_type != 'absolute')
        ts, infection_times = _complex_contagion_kernel(
            self.adj.data, self.adj.indices, self.adj.indptr,
            self.degree, state, float(threshold), is_fractional, max_steps
        )
        time_series = [[int(ts[t, sim]) for t in range(ts.shape[0])]
                       for sim in range(n_simulations)]
        return time_series, infection_times


# =============================================================================
# Network loader
# =============================================================================

def load_networks(folder) -> Dict[str, nx.Graph]:
    """Load all graph.npz networks from subfolders of {folder}/enriched/."""
    folder = Path(folder)
    networks = {}
    enriched_dir = folder / 'enriched'
    if not enriched_dir.exists():
        print(f"  No enriched/ dir found in {folder}")
        return networks
    for npz_file in sorted(enriched_dir.glob('*/graph.npz')):
        name = npz_file.parent.name
        data = np.load(npz_file, allow_pickle=True)
        directed = bool(data['directed'])
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(data['nodes'].tolist())
        G.add_edges_from(data['edges'].tolist())
        networks[name] = G
        print(f"  Loaded {name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return networks


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    n_simulations: int = 200
    max_steps: int = 10000
    threshold_type: str = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 0.30
    n_thresholds: int = 4

    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)


@dataclass
class NetworkConfig:
    base_folder: str = "my_networks"


def discover_network_folders(base_folder: str = "my_networks") -> List[Path]:
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

        if out_file.exists():
            print(f"  Already done: {out_file}. Skipping.")
            return pd.read_csv(out_file)

        results = self._run_config(folder, out_path, task_id)
        if not results:
            print(f"  No results for task {task_id}.")
            return None

        df = pd.DataFrame(results)
        df.to_csv(out_file, index=False)
        print(f"  Saved {len(df)} rows to {out_file}")
        return df

    def _run_config(self, folder: Path, out_path: Path, task_id: int) -> Optional[List[Dict]]:
        params = parse_folder_params(folder)

        try:
            networks = load_networks(str(folder))
            contested_results, ratios = self._sweep_contested(networks, out_path, task_id)

            results = []
            for net_name, thresh_results in contested_results.items():
                for thresh_idx, stat_val in thresh_results.items():
                    results.append({
                        **params,
                        'folder': folder.name,
                        'network': net_name,
                        'threshold_idx': thresh_idx,
                        'threshold_value': self.sim.thresholds[thresh_idx],
                        'mean_final_adoption': stat_val["mean"],
                        'variance_final_adoption': stat_val["variance"],
                        'ratio': ratios.get(net_name),
                    })

            del networks
            gc.collect()
            return results

        except Exception as e:
            print(f"  Error with {folder.name}: {e}")
            return None

    def _sweep_contested(self, networks: Dict, out_path: Path, task_id: int) -> Tuple[Dict, Dict]:
        results, ratios = {}, {}
        infection_times_all = {}

        for name, G in networks.items():
            try:
                df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")
                ratios[name] = df_n.n.max() / df_n.n.sum()
                del df_n
            except FileNotFoundError:
                pass

            sim = ContagionSimulator(G, name)
            initial = int(len(G) * self.sim.initial_infected_fraction)

            finals = {}
            for i, tau in enumerate(self.sim.thresholds):
                ts_list, inf_times = sim.complex_contagion(
                    threshold=tau,
                    threshold_type=self.sim.threshold_type,
                    seeding='focal_neighbors',
                    max_steps=self.sim.max_steps,
                    n_simulations=self.sim.n_simulations,
                    initial_infected=initial,
                )
                finals[i] = {
                    "mean": np.mean([ts[-1] for ts in ts_list]),
                    "variance": np.var([ts[-1] for ts in ts_list]),
                }
                infection_times_all[f"{name}__thresh{i}"] = inf_times
                del ts_list, inf_times

            results[name] = finals
            del sim
            gc.collect()

        np.savez_compressed(out_path / f"task_{task_id}_infection_times.npz", **infection_times_all)
        return results, ratios


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

    analyzer = ContagionAnalyzer(net_config=net_config)
    analyzer.run_task(task_id, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
