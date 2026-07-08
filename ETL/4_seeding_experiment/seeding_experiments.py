"""
Stage: run complex-contagion diffusion simulations on generated networks.

One invocation = one network_id from the data lake. Loads
networks/{network_id}/edges.npz (+ nodes.parquet for original node ids),
runs n_simulations per threshold value, and writes:

  - infection_events/{network_id}/threshold_{i}.parquet
      Long-format, per-node, per-simulation infection timestamps -- the
      atomic fact table for all diffusion analysis. Columns: node_id
      (original id from nodes.parquet), sim, infection_step (NaN if never
      infected), plus threshold_idx / threshold_value added by data_lake.

  - simulations/{network_id}/{network_id}__t{i}.json
      One config record per (network, threshold) describing all
      n_simulations runs (per-sim seeds are derivable as base_seed + sim).

Aggregates like mean final adoption are NOT stored -- they're one groupby
away from infection_events, and storing them separately just risks drift.

The Numba kernel additionally tracks infection_step per node (the step at
which each node first crossed the threshold; seeds recorded as step 0,
never-infected as -1), on top of the original aggregate time series.

Usage:
    python seeding_experiments.py --network_id sample_00000__etngrp__household
    python seeding_experiments.py --task_id N
    python seeding_experiments.py --list_tasks
"""

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numba
import numpy as np
import pandas as pd
from scipy import sparse

import data_lake


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
        state                 : (n, n_sims) int8 -- modified in-place
        threshold             : adoption threshold value
        is_fractional         : True = fractional threshold, False = absolute
        max_steps             : maximum simulation steps
        verbose               : 0=silent, 1=summary, 2=detailed

    Returns:
        time_series    : (actual_steps+1, n_sims) int64 array of infected counts
        infection_step : (n, n_sims) int32 -- step each node first became
                          infected (seeds = 0), or -1 if never infected
    """
    n, n_sims = state.shape
    infected_counts = np.empty((n, n_sims), dtype=np.float32)
    time_series     = np.empty((max_steps + 1, n_sims), dtype=np.int64)
    infection_step  = np.full((n, n_sims), -1, dtype=np.int32)

    # Seeds are already infected in `state` when the kernel starts --
    # record them at step 0 so infection_step covers every infected node.
    for i in numba.prange(n):
        for s in range(n_sims):
            if state[i, s] > 0:
                infection_step[i, s] = 0

    current_counts = np.sum(state > 0, axis=0).astype(np.int64)
    time_series[0, :] = current_counts

    active = np.ones(n_sims, dtype=numba.boolean)
    actual_steps = 0

    for step in range(max_steps):

        # -- Pass 1: sparse matmul -- skip already-infected nodes ------------
        for i in numba.prange(n):
            row_start = indptr[i]
            row_end   = indptr[i + 1]
            for s in range(n_sims):
                if active[s] and state[i, s] == 0:
                    val = 0.0
                    for j_ptr in range(row_start, row_end):
                        val += data[j_ptr] * state[indices[j_ptr], s]
                    infected_counts[i, s] = val

        # -- Pass 2: threshold check + state update ---------------------------
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
                        infection_step[i, s] = step + 1

        # -- Incremental count update -----------------------------------------
        prev_counts    = current_counts.copy()
        current_counts = current_counts + np.sum(delta, axis=0).astype(np.int64)
        time_series[step + 1, :] = current_counts
        actual_steps += 1

        # -- Per-sim convergence check ----------------------------------------
        any_active = False
        for s in range(n_sims):
            if active[s]:
                if current_counts[s] == n or current_counts[s] == prev_counts[s]:
                    active[s] = False
                else:
                    any_active = True
        if not any_active:
            break

    # -- Verbose logging --------------------------------------------------------
    if verbose >= 1:
        final_counts  = time_series[actual_steps, :]
        converged     = np.sum((final_counts == n) |
                               (final_counts == time_series[actual_steps - 1, :]))
        full_cascades = np.sum(final_counts == n)
        stalled       = converged - full_cascades
        print("    -> Steps:", actual_steps, "/", max_steps,
              "| Converged:", converged, "/", n_sims,
              "| Full:", full_cascades, "| Stalled:", stalled)
        if verbose >= 2:
            print("    -> Adoption: mean=", round(final_counts.mean(), 1),
                  "std=", round(final_counts.std(), 1),
                  "range=[", int(final_counts.min()), ",", int(final_counts.max()), "]")

    return time_series[:actual_steps + 1], infection_step


# =============================================================================
# ContagionSimulator
# =============================================================================

class ContagionSimulator:
    """Simulates complex contagion spreading on networks using a Numba kernel."""

    def __init__(self, network, name="Network"):
        """
        Args:
            network : dict with keys {n, adj, degree, name, node_ids}
            name    : fallback name
        """
        self.name     = network.get('name', name)
        self.n        = network['n']
        self.adj      = network['adj']
        self.degree   = network['degree']
        self.node_ids = network.get('node_ids', np.arange(self.n))

        # Explicit out/in degree (row sums = out, col sums = in)
        self.out_degree = np.array(self.adj.sum(axis=1)).flatten().astype(np.int32)
        self.in_degree  = np.array(self.adj.sum(axis=0)).flatten().astype(np.int32)

        # Populated by _seed_state for downstream analysis (positional indices)
        self.last_focal_nodes: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------

    def _seed_state(self, state, n_simulations, seeding, initial_infected,
                    base_seed=0, neighbor_k=None):
        """
        Populate initial infection state matrix.

        Modes:
            'random'          -- initial_infected random nodes per sim
            'focal_neighbors' -- 1 focal node + ALL its out-neighbours
            'neighbor_k'      -- 1 focal node (out-degree >= k) + exactly k
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

        Returns:
            time_series    : (actual_steps+1, n_sims) -- infected count per step
            infection_step : (n, n_sims) int32 -- step each node (by positional
                              index) first got infected, -1 if never. Map row i
                              back to original ids via self.node_ids[i].
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

    # -------------------------------------------------------------------------

    def infection_events_df(self, infection_step: np.ndarray) -> pd.DataFrame:
        """
        Convert a raw (n, n_sims) infection_step array into the long-format
        DataFrame data_lake.write_infection_events() expects, keyed by
        ORIGINAL node id. Never-infected rows are kept with NaN, so 'never
        infected' is explicit rather than silently dropped.
        """
        n, n_sims = infection_step.shape
        node_id_col = np.repeat(self.node_ids, n_sims)
        sim_col     = np.tile(np.arange(n_sims), n)
        step_col    = infection_step.reshape(-1).astype(np.float64)
        step_col[step_col < 0] = np.nan

        return pd.DataFrame({
            'node_id':        node_id_col,
            'sim':            sim_col,
            'infection_step': step_col,
        })


# =============================================================================
# Network loader (data lake)
# =============================================================================

def load_network(network_id: str) -> dict:
    """
    Build {n, adj, degree, name, node_ids} from the lake's
    networks/{network_id}/edges.npz (+ nodes.parquet for original ids).

    edges.npz stores positional int32 (src, dst) pairs, already remapped to
    0..n-1 by PAWN_analysis.edges_from_nx() in the SAME node order that
    extract_nodes() wrote nodes.parquet in -- so row i of nodes.parquet is
    positional index i, and node_ids comes straight from its node_id column.

    Edges are treated as DIRECTED, exactly as stored (generation used
    reciprocity=1, so mutual edges are already present as two rows).
    """
    net_dir = data_lake.ROOT / 'networks' / network_id

    with np.load(net_dir / 'edges.npz', allow_pickle=True) as data:
        edges = np.asarray(data['edges'])

    nodes_path = net_dir / 'nodes.parquet'
    if nodes_path.exists():
        node_ids = pd.read_parquet(nodes_path)['node_id'].to_numpy()
        n = len(node_ids)
    else:
        # Fall back to edge-implied count -- isolated trailing nodes would be
        # missed this way, which is why nodes.parquet is preferred.
        n = int(edges.max()) + 1 if len(edges) else 0
        node_ids = np.arange(n)
        print(f"  Warning: no nodes.parquet for {network_id}; "
              f"using edge-implied n={n} and positional ids")

    adj = sparse.csr_matrix(
        (np.ones(len(edges), dtype=np.float32),
         (edges[:, 0].astype(np.int32), edges[:, 1].astype(np.int32))),
        shape=(n, n), dtype=np.float32,
    )
    degree = np.array(adj.sum(axis=1)).flatten()

    return {'n': n, 'adj': adj, 'degree': degree,
            'name': network_id, 'node_ids': node_ids}


def discover_network_ids() -> list[str]:
    net_dir = data_lake.ROOT / 'networks'
    if not net_dir.exists():
        return []
    return sorted(p.name for p in net_dir.iterdir()
                  if p.is_dir() and (p / 'edges.npz').exists())


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    n_simulations:             int   = 100
    max_steps:                 int   = 10000
    threshold_type:            str   = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold:             float = 0.3
    max_threshold:             float = 0.3
    n_thresholds:              int   = 1
    base_seed:                 int   = 0
    verbose:                   int   = 0
    seeding:                   str   = 'neighbor_k'
    neighbor_k:                int   = 20  # focal node must have >= 20 out-neighbours

    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)


# =============================================================================
# Runner
# =============================================================================

def threshold_done(network_id: str, threshold_idx: int) -> bool:
    return (data_lake.ROOT / 'infection_events' / network_id /
            f'threshold_{threshold_idx}.parquet').exists()


def run_one(network_id: str, cfg: SimulationConfig) -> None:
    todo = [i for i in range(cfg.n_thresholds) if not threshold_done(network_id, i)]
    if not todo:
        print(f"  [{network_id}] all thresholds already done, skipping")
        return

    network = load_network(network_id)
    sim = ContagionSimulator(network)
    del network

    initial = max(1, int(sim.n * cfg.initial_infected_fraction))
    print(f"  Simulating {network_id} ({sim.n:,} nodes, {cfg.n_simulations} sims, "
          f"seeding={cfg.seeding}, k={cfg.neighbor_k})...", flush=True)

    for i in todo:
        tau = float(cfg.thresholds[i])
        print(f"    threshold {i+1}/{cfg.n_thresholds} (tau={tau:.3f})", flush=True)

        ts_array, infection_step = sim.run(
            threshold        = tau,
            threshold_type   = cfg.threshold_type,
            initial_infected = initial,
            max_steps        = cfg.max_steps,
            n_simulations    = cfg.n_simulations,
            seeding          = cfg.seeding,
            neighbor_k       = cfg.neighbor_k,
            base_seed        = cfg.base_seed,
            verbose          = cfg.verbose,
        )

        events = sim.infection_events_df(infection_step)
        data_lake.write_infection_events(network_id, i, tau, events)

        # One config record describing all n_simulations runs at this
        # threshold; per-sim seeds are base_seed + sim by construction.
        
        data_lake.write_simulation_meta(
            f'{network_id}__t{i}', network_id, sim_index=-1,
            seed=cfg.base_seed
        )

        del ts_array, infection_step, events
        gc.collect()

    print(f"  [{network_id}] DONE", flush=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Contagion sweep -- one task per network in the data lake"
    )
    parser.add_argument('--network_id', type=str, default=None,
                        help='Specific network_id to simulate. If omitted, uses --task_id.')
    parser.add_argument('--task_id', type=int, default=None,
                        help='Index into sorted list of networks with edges.npz.')
    parser.add_argument('--list_tasks', action='store_true',
                        help='Print number of discovered networks and exit.')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--seeding', type=str, default='neighbor_k',
                        choices=['random', 'focal_neighbors', 'neighbor_k'])
    parser.add_argument('--neighbor_k', type=int, default=20)
    parser.add_argument('--n_sims', type=int, default=100)
    args = parser.parse_args()

    if args.list_tasks:
        print(len(discover_network_ids()))
        return

    cfg = SimulationConfig(
        verbose       = args.verbose,
        seeding       = args.seeding,
        neighbor_k    = args.neighbor_k,
        n_simulations = args.n_sims,
    )

    if args.network_id:
        run_one(args.network_id, cfg)
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --network_id, --task_id, or set $SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    network_ids = discover_network_ids()
    if task_id >= len(network_ids):
        print(f"task_id {task_id} out of range ({len(network_ids)} networks). Exiting.")
        return

    run_one(network_ids[task_id], cfg)


if __name__ == '__main__':
    main()