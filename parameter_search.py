"""
Bayesian optimization for a single aggregation file pair.

This script is designed to be called per-file (either from run_locally.sh
or from a SLURM array job). It handles one (pop, interactions) pair at a time.

Usage:
    python run_optimization.py --pop_file <path> --link_file <path> --suffix <name>

The script will:
1. Load the empirical network
2. Run Bayesian optimization over generator parameters
3. Save results to results/<suffix>/
"""

import argparse
import json
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import rustworkx as rx
import networkx as nx
from scipy import stats as scipy_stats

from asnu import generate, create_communities

# Quiet Optuna's per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# CONFIGURATION (shared across calls)
# ============================================================================

EMPIRICAL_NETWORK_PATH = Path("Data/empirical/full_network.pkl")
OUTPUT_DIR = Path("results")
N_TRIALS = 25
N_RUNS_PER_TRIAL = 1
SCALE = 1

# Community detection: when True, runs Louvain on undirected projection
USE_DETECTED_COMMUNITIES = True
COMMUNITY_DETECTION_BACKEND = "networkx"  # "networkx" | "networkit"
EMPIRICAL_PARTITION_CACHE = OUTPUT_DIR / "empirical_partition.pkl"

# Fixed parameters — not tuned
FIXED_PARAMS = {
    "scale": SCALE,
    "mixing_floor": 0.3,
    "isolation_threshold": 0.15,
    "bridge_probability": 0.0,
    "fully_connect_communities": False,
    "fill_unfulfilled": True,
    "reciprocity": 1,
    "preferential_attachment": 0,
}

# Search space for tuned parameters
SEARCH_SPACE = {
    "transitivity": ("float", 0.0, 1.0),
    "number_of_communities": ("int", 0, 35000),
}

# Loss weights
LOSS_WEIGHTS = {
    "mean_degree": 1.0,
    "std_degree": 1.0,
    "skew_degree": 0.5,
    "frac_isolates": 0.5,
    "frac_degree_1": 0.5,
    "transitivity": 5.0,
    "modularity": 5.0,
    "frac_within_community": 1.5,
    "mixing_matrix_frobenius": 1.5,
    "degree_distribution_ks": 1.0,
}


# ============================================================================
# NETWORK CONVERSION
# ============================================================================

def nx_to_rx(G_nx: nx.DiGraph) -> tuple[rx.PyDiGraph, dict]:
    """Convert NetworkX DiGraph to rustworkx PyDiGraph."""
    G_rx = rx.PyDiGraph()
    node_map: dict = {}
    for node in G_nx.nodes():
        attrs = G_nx.nodes[node]
        payload = {"_orig_id": node, **(attrs if attrs else {})}
        idx = G_rx.add_node(payload)
        node_map[node] = idx
    for u, v, edge_attrs in G_nx.edges(data=True):
        G_rx.add_edge(node_map[u], node_map[v], edge_attrs if edge_attrs else {})
    return G_rx, node_map


# ============================================================================
# COMMUNITY DETECTION
# ============================================================================

def detect_communities_louvain(G_rx: rx.PyDiGraph,
                               backend: str = "networkx",
                               seed: int = 42) -> dict:
    """Run Louvain on undirected projection of G_rx.

    Returns dict mapping rustworkx node index -> community id.
    """
    n = len(G_rx)
    if n == 0:
        return {}

    if backend == "networkit":
        try:
            import networkit as nk
        except ImportError:
            print("  warning: networkit not installed, falling back to networkx")
            backend = "networkx"

    if backend == "networkit":
        # Build undirected NetworKit graph
        nk_g = nk.Graph(n, weighted=True, directed=False)
        und_weights: dict = defaultdict(float)
        for u, v in G_rx.edge_list():
            key = (u, v) if u < v else (v, u)
            und_weights[key] += 1.0
        for (u, v), w in und_weights.items():
            nk_g.addEdge(u, v, w)
        plm = nk.community.PLM(nk_g, refine=True)
        plm.run()
        partition = plm.getPartition()
        return {idx: int(partition[idx]) for idx in range(n)}

    # networkx backend
    G_und = nx.Graph()
    G_und.add_nodes_from(G_rx.node_indices())
    und_weights: dict = defaultdict(float)
    for u, v in G_rx.edge_list():
        key = (u, v) if u < v else (v, u)
        und_weights[key] += 1.0
    for (u, v), w in und_weights.items():
        G_und.add_edge(u, v, weight=w)
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_und, weight="weight", seed=seed)
    except (ImportError, AttributeError):
        from networkx.algorithms.community.louvain import louvain_communities
        communities = louvain_communities(G_und, weight="weight", seed=seed)
    node_to_comm: dict = {}
    for cid, members in enumerate(communities):
        for node in members:
            node_to_comm[node] = cid
    return node_to_comm


# ============================================================================
# STATISTICS
# ============================================================================

def _node_group(payload: Any) -> Any:
    """Extract 'group' label from node payload."""
    if isinstance(payload, dict):
        for key in ("group", "group_id", "_group", "demographic_group"):
            if key in payload:
                return payload[key]
    return None


def _node_community(payload: Any) -> Any:
    """Extract 'community' label from node payload."""
    if isinstance(payload, dict):
        for key in ("community", "community_id", "_community"):
            if key in payload:
                return payload[key]
    return None


def compute_stats(G_rx: rx.PyDiGraph, G_nx: nx.DiGraph,
                  partition: "dict | None" = None) -> dict:
    """Compute all matching statistics."""
    n_nodes = len(G_rx)
    n_edges = G_rx.num_edges()

    # Degree stats
    in_deg = np.array([G_rx.in_degree(n) for n in G_rx.node_indices()], dtype=np.int64)
    out_deg = np.array([G_rx.out_degree(n) for n in G_rx.node_indices()], dtype=np.int64)
    total_deg = in_deg + out_deg

    s: dict = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": float(in_deg.mean()) if n_nodes else 0.0,
        "std_degree": float(in_deg.std()) if n_nodes else 0.0,
        "max_degree": int(in_deg.max()) if n_nodes else 0,
        "median_degree": float(np.median(in_deg)) if n_nodes else 0.0,
        "q25_degree": float(np.quantile(in_deg, 0.25)) if n_nodes else 0.0,
        "q75_degree": float(np.quantile(in_deg, 0.75)) if n_nodes else 0.0,
        "skew_degree": float(scipy_stats.skew(in_deg)) if n_nodes else 0.0,
        "frac_isolates": float((total_deg == 0).mean()) if n_nodes else 0.0,
        "frac_degree_1": float((total_deg == 1).mean()) if n_nodes else 0.0,
    }

    # Transitivity
    try:
        s["transitivity"] = float(rx.transitivity(G_rx))
    except Exception:
        s["transitivity"] = 0.0

    # Community stats
    if partition is not None:
        node_to_comm = partition
        communities: dict = defaultdict(list)
        for idx, cid in node_to_comm.items():
            communities[cid].append(idx)
        has_communities = len(communities) > 0
    else:
        communities: dict = defaultdict(list)
        has_communities = False
        for idx in G_rx.node_indices():
            c = _node_community(G_rx[idx])
            if c is not None:
                communities[c].append(idx)
                has_communities = True
        node_to_comm = {idx: c for c, members in communities.items() for idx in members}

    within = 0
    between = 0
    bridge_counts: dict = defaultdict(int)
    for u, v in G_rx.edge_list():
        cu = node_to_comm.get(u)
        cv = node_to_comm.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            within += 1
        else:
            between += 1
            key = (cu, cv) if cu < cv else (cv, cu)
            bridge_counts[key] += 1
    total = within + between
    s["frac_within_community"] = within / total if total else 0.0
    s["frac_between_community"] = between / total if total else 0.0
    s["n_communities"] = len(communities)
    s["bridge_widths"] = sorted(bridge_counts.values(), reverse=True)

    # Modularity using NetworkX label propagation (matches input script)
    G_undir = G_nx.to_undirected()
    try:
        communities_nx = list(nx.community.label_propagation_communities(G_undir))
        s["modularity"] = nx.community.modularity(G_undir, communities_nx)
    except Exception:
        s["modularity"] = 0.0

    # Group mixing matrix
    groups: set = set()
    node_to_group: dict = {}
    for idx in G_rx.node_indices():
        g = _node_group(G_rx[idx])
        if g is not None:
            node_to_group[idx] = g
            groups.add(g)

    if groups:
        group_list = sorted(groups, key=lambda x: str(x))
        gi = {g: i for i, g in enumerate(group_list)}
        k = len(group_list)
        M = np.zeros((k, k), dtype=np.float64)
        for u, v in G_rx.edge_list():
            gu = node_to_group.get(u)
            gv = node_to_group.get(v)
            if gu is None or gv is None:
                continue
            M[gi[gu], gi[gv]] += 1.0
        row_sums = M.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            M_norm = np.where(row_sums > 0, M / row_sums, 0.0)
        s["mixing_matrix"] = M_norm.tolist()
        s["mixing_matrix_groups"] = [str(g) for g in group_list]
    else:
        s["mixing_matrix"] = []
        s["mixing_matrix_groups"] = []

    # Degree distribution
    s["degree_distribution"] = in_deg.tolist()

    return s


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def _rel_diff_sq(a: float, b: float) -> float:
    """Squared relative difference."""
    if abs(b) < 1e-12:
        return float((a - b) ** 2)
    return float(((a - b) / abs(b)) ** 2)


def _mixing_matrix_distance(M_gen: list, groups_gen: list,
                            M_emp: list, groups_emp: list) -> float:
    """Frobenius distance between normalized mixing matrices."""
    if not M_gen or not M_emp:
        return 1.0
    A = np.array(M_gen)
    B = np.array(M_emp)
    all_groups = sorted(set(groups_gen) | set(groups_emp))
    n = len(all_groups)
    idx = {g: i for i, g in enumerate(all_groups)}
    A_full = np.zeros((n, n))
    B_full = np.zeros((n, n))
    for i, g1 in enumerate(groups_gen):
        for j, g2 in enumerate(groups_gen):
            A_full[idx[g1], idx[g2]] = A[i, j]
    for i, g1 in enumerate(groups_emp):
        for j, g2 in enumerate(groups_emp):
            B_full[idx[g1], idx[g2]] = B[i, j]
    return float(np.linalg.norm(A_full - B_full))


def _ks_distance(a: list, b: list) -> float:
    """Kolmogorov-Smirnov distance between two empirical distributions."""
    if not a or not b:
        return 1.0
    try:
        return float(scipy_stats.ks_2samp(a, b).statistic)
    except Exception:
        return 1.0


def compute_loss(stats_gen: dict, stats_emp: dict, weights: dict) -> tuple[float, dict]:
    """Composite loss."""
    terms: dict = {}

    scalar_keys = [
        "mean_degree", "std_degree", "skew_degree",
        "frac_isolates", "frac_degree_1",
        "transitivity", "modularity", "frac_within_community",
    ]
    for k in scalar_keys:
        if k in stats_gen and k in stats_emp:
            terms[k] = _rel_diff_sq(stats_gen[k], stats_emp[k])

    terms["mixing_matrix_frobenius"] = _mixing_matrix_distance(
        stats_gen.get("mixing_matrix", []),
        stats_gen.get("mixing_matrix_groups", []),
        stats_emp.get("mixing_matrix", []),
        stats_emp.get("mixing_matrix_groups", []),
    )
    terms["degree_distribution_ks"] = _ks_distance(
        stats_gen.get("degree_distribution", []),
        stats_emp.get("degree_distribution", []),
    )

    total = sum(weights.get(k, 1.0) * v for k, v in terms.items())
    return total, terms


# ============================================================================
# GENERATION AND EVALUATION
# ============================================================================

def generate_one(pop_file: Path, link_file: Path, params: dict,
                 comm_file: Path, base_path: Path) -> tuple[rx.PyDiGraph, nx.DiGraph]:
    """Run create_communities + generate once."""
    create_communities(
        str(pop_file), str(link_file),
        scale=params["scale"],
        number_of_communities=params["number_of_communities"],
        output_path=str(comm_file),
        mode="capacity_fast",
        mixing_floor=params["mixing_floor"],
        isolation_threshold=params["isolation_threshold"],
    )
    graph = generate(
        str(pop_file), str(link_file),
        preferential_attachment=params["preferential_attachment"],
        scale=params["scale"],
        reciprocity=params["reciprocity"],
        transitivity=params["transitivity"],
        community_file=str(comm_file),
        base_path=str(base_path),
        bridge_probability=params["bridge_probability"],
        fully_connect_communities=params["fully_connect_communities"],
        fill_unfulfilled=params["fill_unfulfilled"],
    )
    G_nx = graph.graph
    G_rx, _ = nx_to_rx(G_nx)
    return G_rx, G_nx


def evaluate_params(pop_file: Path, link_file: Path, params: dict,
                    target_stats: dict, weights: dict,
                    comm_file: Path, base_path: Path,
                    n_runs: int) -> tuple[float, list[dict], list[dict]]:
    """Run n_runs generations and compute mean loss."""
    losses: list[float] = []
    all_stats: list[dict] = []
    all_terms: list[dict] = []
    for run_idx in range(n_runs):
        run_base = Path(f"{base_path}_run{run_idx}")
        run_comm = Path(f"{comm_file}.run{run_idx}.json")
        try:
            G_rx, G_nx = generate_one(pop_file, link_file, params, run_comm, run_base)
            stats_gen = compute_stats(G_rx, G_nx)
            loss, terms = compute_loss(stats_gen, target_stats, weights)
        except Exception as e:
            print(f"    run {run_idx} failed: {e}")
            traceback.print_exc()
            loss = 1e6
            stats_gen = {}
            terms = {}
        losses.append(loss)
        all_stats.append(stats_gen)
        all_terms.append(terms)
    return float(np.mean(losses)), all_stats, all_terms


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def make_objective(pop_file: Path, link_file: Path, target_stats: dict,
                   weights: dict, fixed_params: dict, search_space: dict,
                   work_dir: Path, n_runs: int):
    """Build Optuna objective closure."""

    def objective(trial: optuna.Trial) -> float:
        params: dict = dict(fixed_params)
        for name, spec in search_space.items():
            kind = spec[0]
            if kind == "float":
                _, lo, hi = spec
                params[name] = trial.suggest_float(name, lo, hi)
            elif kind == "int":
                _, lo, hi = spec
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                raise ValueError(f"Unknown spec kind for {name}: {kind}")

        comm_file = work_dir / f"comm_trial{trial.number}.json"
        base_path = work_dir / f"net_trial{trial.number}"

        loss, all_stats, all_terms = evaluate_params(
            pop_file, link_file, params, target_stats, weights,
            comm_file, base_path, n_runs,
        )

        trial.set_user_attr("per_run_stats", all_stats)
        trial.set_user_attr("per_run_terms", all_terms)

        # Clean up intermediate files
        for run_idx in range(n_runs):
            for p in [Path(f"{base_path}_run{run_idx}"),
                      Path(f"{comm_file}.run{run_idx}.json")]:
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                    except Exception:
                        pass

        return loss

    return objective


# ============================================================================
# EMPIRICAL NETWORK LOADING
# ============================================================================

def load_empirical_network(path: Path) -> tuple[rx.PyDiGraph, nx.DiGraph]:
    """Load empirical network from pickle."""
    print(f"Loading empirical network from {path} ...")
    with open(path, "rb") as f:
        G_nx = pickle.load(f)
    if not isinstance(G_nx, nx.DiGraph):
        G_nx = nx.DiGraph(G_nx)
    G_rx, _ = nx_to_rx(G_nx)
    print(f"  loaded: {len(G_rx)} nodes, {G_rx.num_edges()} edges")
    return G_rx, G_nx


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for a single network pair"
    )
    parser.add_argument(
        "--pop_file", type=Path, required=True,
        help="Path to population CSV"
    )
    parser.add_argument(
        "--link_file", type=Path, required=True,
        help="Path to interactions CSV"
    )
    parser.add_argument(
        "--suffix", type=str, required=True,
        help="Suffix for output directory and logging"
    )
    parser.add_argument(
        "--n_trials", type=int, default=N_TRIALS,
        help=f"Number of Optuna trials (default: {N_TRIALS})"
    )
    parser.add_argument(
        "--n_runs", type=int, default=N_RUNS_PER_TRIAL,
        help=f"Runs per trial for averaging (default: {N_RUNS_PER_TRIAL})"
    )
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"Network Matching Optimization: {args.suffix}")
    print(f"{'='*70}")
    print(f"Population: {args.pop_file}")
    print(f"Interactions: {args.link_file}")
    print(f"Trials: {args.n_trials}, Runs per trial: {args.n_runs}")
    print()

    # Verify input files exist
    if not args.pop_file.exists():
        print(f"ERROR: Population file not found: {args.pop_file}")
        sys.exit(1)
    if not args.link_file.exists():
        print(f"ERROR: Interactions file not found: {args.link_file}")
        sys.exit(1)

    # Load empirical network
    if not EMPIRICAL_NETWORK_PATH.exists():
        print(f"ERROR: Empirical network not found: {EMPIRICAL_NETWORK_PATH}")
        print("Set EMPIRICAL_NETWORK_PATH in run_optimization.py")
        sys.exit(1)

    G_emp, G_nx_emp = load_empirical_network(EMPIRICAL_NETWORK_PATH)
    
    # Compute or load empirical partition
    if USE_DETECTED_COMMUNITIES:
        if EMPIRICAL_PARTITION_CACHE.exists():
            print(f"Loading cached empirical partition from {EMPIRICAL_PARTITION_CACHE}")
            with open(EMPIRICAL_PARTITION_CACHE, "rb") as f:
                emp_partition = pickle.load(f)
        else:
            print("Computing empirical partition (Louvain)...")
            emp_partition = detect_communities_louvain(G_emp, backend=COMMUNITY_DETECTION_BACKEND)
            EMPIRICAL_PARTITION_CACHE.parent.mkdir(exist_ok=True, parents=True)
            with open(EMPIRICAL_PARTITION_CACHE, "wb") as f:
                pickle.dump(emp_partition, f)
            print(f"  cached to {EMPIRICAL_PARTITION_CACHE}")
    else:
        emp_partition = None

    # Compute target stats
    target_stats = compute_stats(G_emp, G_nx_emp, partition=emp_partition)
    print("\nEmpirical stats (selected):")
    for k in ["n_nodes", "n_edges", "mean_degree", "std_degree",
              "transitivity", "modularity", "frac_within_community",
              "n_communities"]:
        val = target_stats.get(k)
        print(f"  {k}: {val}")

    # Setup work directory
    work_dir = OUTPUT_DIR / args.suffix
    work_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nOutput directory: {work_dir}")

    # Create or resume Optuna study
    study_db = work_dir / "optuna.db"
    storage = f"sqlite:///{study_db}"
    study = optuna.create_study(
        study_name=args.suffix,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = make_objective(
        args.pop_file, args.link_file, target_stats, LOSS_WEIGHTS,
        FIXED_PARAMS, SEARCH_SPACE, work_dir, args.n_runs,
    )

    # Determine how many trials to run
    completed = sum(1 for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, args.n_trials - completed)

    if remaining > 0:
        print(f"\nRunning {remaining} trials ({completed} already complete)")
        print()

        def progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            print(f"  trial {trial.number}: loss={trial.value:.4f} "
                  f"params={trial.params}")

        study.optimize(objective, n_trials=remaining, callbacks=[progress_cb])
    else:
        print(f"\nAll {args.n_trials} trials already complete")

    # Save results
    best_trial = study.best_trial
    best_record = {
        "suffix": args.suffix,
        "best_loss": best_trial.value,
        "best_params": best_trial.params,
        "fixed_params": FIXED_PARAMS,
        "n_trials": len(study.trials),
        "best_trial_per_run_stats": best_trial.user_attrs.get("per_run_stats", []),
        "best_trial_per_run_terms": best_trial.user_attrs.get("per_run_terms", []),
    }
    with open(work_dir / "best_params.json", "w") as f:
        json.dump(best_record, f, indent=2, default=str)

    history = []
    for t in study.trials:
        history.append({
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "params": t.params,
        })
    with open(work_dir / "trials.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save last generated network
    net_dir = work_dir / "net"
    net_dir.mkdir(exist_ok=True)
    print(f"\nBest trial network saved to {net_dir}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Optimization complete for: {args.suffix}")
    print(f"Best loss: {best_trial.value:.4f}")
    print(f"Best params: {best_trial.params}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())