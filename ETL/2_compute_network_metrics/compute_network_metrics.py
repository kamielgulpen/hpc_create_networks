"""
Stage: compute topological features for every generated network.

Reads networks/{network_id}/edges.npz from the data lake, computes
graph-theoretic metrics with igraph, and merges them into
networks/{network_id}/network_stats.parquet -- alongside the basic
n_nodes/n_edges/gen_time_s that PAWN_analysis.py already wrote there during
generation. This is the "Compute topological features" step in the general
pipeline; results land directly in network_stats, queryable immediately,
no separate aggregate-into-one-big-csv step needed afterward.

If networks/{network_id}/nodes.parquet has a 'community_id' column (see
PAWN_analysis.py's load_node_communities()), also computes inter/intra
community structure stats: for every pair of communities (i, j), the raw
edge count e_ij plus three normalizations (by n_i*n_j, by min(n_i,n_j), by
community total-degree product), summarized as mean/std/min/quartiles/max
across all pairs -- separately for inter-community (i != j) and
intra-community (i == j) pairs, so quartiles reflect mixing structure
rather than being an artifact of community-size differences.

One network per invocation, selected by --network_id or --task_id (index
into the sorted list of all networks that have an edges.npz). Run many of
these in parallel with run_parallel_metrics.py.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy import stats

import data_lake

BIG = 900_000


def load_edges(npz_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        edges        : (m, 2) int32 -- deduplicated, self-loop-free, and
                        REINDEXED to a compact 0..k-1 space, where k is the
                        number of distinct nodes that appear in at least
                        one edge. Isolated nodes are dropped entirely --
                        they're not part of this space at all.
        original_ids : (k,) int32 -- original_ids[i] is the ORIGINAL
                        positional node index (matching the row order of
                        nodes.parquet / edges_from_nx()'s numbering in
                        PAWN_analysis.py) that compact index i refers to.
                        Needed to correctly join anything from
                        nodes.parquet (like community_id) onto this graph.
    """
    with np.load(npz_file, allow_pickle=True) as data:
        arr = np.asarray(data[list(data.keys())[0]])
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"bad shape {arr.shape}")

    edges = np.ascontiguousarray(arr, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return edges.astype(np.int32), np.empty(0, dtype=np.int32)

    # Directed: don't sort axis=1
    packed = (edges[:, 0].astype(np.uint64) << 32) | edges[:, 1].astype(np.uint64)
    packed = np.unique(packed)
    edges = np.empty((len(packed), 2), dtype=np.int32)
    edges[:, 0] = (packed >> 32).astype(np.int32)
    edges[:, 1] = (packed & 0xFFFFFFFF).astype(np.int32)

    original_ids, inv = np.unique(edges.ravel(), return_inverse=True)
    return inv.reshape(-1, 2).astype(np.int32), original_ids.astype(np.int32)


def dist_stats(x, prefix):
    if len(x) == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ["mean", "std", "min", "q25", "median", "q75", "max", "skew"]}
    return {
        f"{prefix}_mean":   float(np.mean(x)),
        f"{prefix}_std":    float(np.std(x)),
        f"{prefix}_min":    float(np.min(x)),
        f"{prefix}_q25":    float(np.quantile(x, 0.25)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q75":    float(np.quantile(x, 0.75)),
        f"{prefix}_max":    float(np.max(x)),
        f"{prefix}_skew":   float(stats.skew(x)),
    }


def community_structure_stats(G, community_compact: np.ndarray, label: str = "community") -> dict:
    """
    Inter/intra-community edge structure for ANY per-node community
    assignment aligned with G's node indices -- doesn't care whether that
    assignment came from asnu's community file (the network's intended/
    generative structure) or from topology-based community detection
    (community_label_propagation()'s membership, the network's realized
    structure). Call this twice with different `label`s to compare both.

    For every ordered pair of observed communities (i, j):
        e_ij                     raw directed edge count from i to j
        e_ij / (n_i * n_j)       density_by_size   -- comparable across
                                  pairs regardless of community size
        e_ij / min(n_i, n_j)     density_by_minsize -- normalizes by the
                                  smaller side, useful when one community
                                  is much bigger than the other
        e_ij / (deg_i * deg_j)   density_by_degree -- normalized by each
                                  community's total (in+out) degree, i.e.
                                  relative to how much total connectivity
                                  each community has, not just node count

    Each is summarized with mean/std/min/quartiles/max/skew, separately
    over i != j pairs (inter) and i == j pairs (intra). Pairs where the
    relevant denominator is 0 are excluded from that stat's distribution
    (not coerced to 0), so a handful of degree-0 communities don't distort
    the density_by_degree quartiles.
    """
    n = G.vcount()
    comm_ids, sizes = np.unique(community_compact, return_counts=True)
    k = len(comm_ids)

    # compact community index per node, aligned with G's node order
    node_comm_idx = np.searchsorted(comm_ids, community_compact)

    edges_arr = np.array(G.get_edgelist(), dtype=np.int64) if G.ecount() else np.empty((0, 2), dtype=np.int64)
    src_idx = node_comm_idx[edges_arr[:, 0]] if len(edges_arr) else np.empty(0, dtype=np.int64)
    dst_idx = node_comm_idx[edges_arr[:, 1]] if len(edges_arr) else np.empty(0, dtype=np.int64)

    e_matrix = np.zeros((k, k), dtype=np.int64)
    if len(edges_arr):
        np.add.at(e_matrix, (src_idx, dst_idx), 1)

    tot_deg = np.asarray(G.degree(mode="all"), dtype=np.int64) if n else np.empty(0, dtype=np.int64)
    deg_sum = np.bincount(node_comm_idx, weights=tot_deg, minlength=k) if n else np.zeros(k)

    size_outer = np.outer(sizes, sizes).astype(np.float64)
    min_outer  = np.minimum.outer(sizes, sizes).astype(np.float64)
    deg_outer  = np.outer(deg_sum, deg_sum).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        density_size = np.where(size_outer > 0, e_matrix / size_outer, np.nan)
        density_min  = np.where(min_outer  > 0, e_matrix / min_outer,  np.nan)
        density_deg  = np.where(deg_outer  > 0, e_matrix / deg_outer,  np.nan)

    inter_mask = ~np.eye(k, dtype=bool)
    intra_mask = np.eye(k, dtype=bool)

    def _finite(mat, mask):
        vals = mat[mask]
        return vals[np.isfinite(vals)]

    rec = {f"{label}_n_observed": float(k)}
    for name, mat in [
        ("edges",              e_matrix.astype(np.float64)),
        ("density_by_size",    density_size),
        ("density_by_minsize", density_min),
        ("density_by_degree",  density_deg),
    ]:
        rec.update(dist_stats(_finite(mat, inter_mask), f"{label}_{name}_inter"))
        rec.update(dist_stats(_finite(mat, intra_mask), f"{label}_{name}_intra"))

    return rec


def compute_metrics(edges: np.ndarray, community_compact: np.ndarray | None = None) -> dict:
    n = int(edges.max()) + 1 if len(edges) else 0
    G = ig.Graph(n=n, directed=True)
    try:
        G.add_edges(edges)
    except TypeError:
        G.add_edges(edges.tolist())

    in_deg  = np.asarray(G.indegree(),  dtype=np.int32)
    out_deg = np.asarray(G.outdegree(), dtype=np.int32)
    tot_deg = in_deg + out_deg

    local_clust = np.asarray(G.transitivity_local_undirected(mode="zero"), dtype=np.float64)
    coreness    = np.asarray(G.coreness(mode="all"),  dtype=np.int32)
    pagerank    = np.asarray(G.pagerank(directed=True), dtype=np.float64)
    knn_vals, _ = G.knn()
    knn = np.asarray([v if v is not None else 0.0 for v in knn_vals], dtype=np.float64)

    weak    = G.connected_components(mode="weak")
    weak_m  = np.asarray(weak.membership)
    weak_sz = np.bincount(weak_m)
    is_weak = len(weak_sz) == 1
    lcc_id  = int(np.argmax(weak_sz))
    lcc_sz  = int(weak_sz[lcc_id])

    strong    = G.connected_components(mode="strong")
    strong_sz = np.bincount(strong.membership)
    is_strong = len(strong_sz) == 1

    rec = {
        'nodes': n,
        'edges': G.ecount(),
        **dist_stats(in_deg, "in_degree"),
        **dist_stats(out_deg, "out_degree"),
        **dist_stats(tot_deg, "total_degree"),
        **dist_stats(np.log1p(in_deg),  "log_in_degree"),
        **dist_stats(np.log1p(out_deg), "log_out_degree"),
        **dist_stats(local_clust, "local_clustering"),
        **dist_stats(coreness,    "coreness"),
        **dist_stats(pagerank,    "pagerank"),
        **dist_stats(knn,         "avg_neighbor_degree"),
        'frac_isolates':          float((tot_deg == 0).mean()) if n else 0.0,
        'frac_sources':           float((in_deg  == 0).mean()) if n else 0.0,
        'frac_sinks':             float((out_deg == 0).mean()) if n else 0.0,
        'frac_degree_1':          float((tot_deg == 1).mean()) if n else 0.0,
        'global_clustering':      float(G.transitivity_undirected(mode="zero")),
        'avg_local_clustering':   float(G.transitivity_avglocal_undirected(mode="zero")),
        'reciprocity':            float(G.reciprocity()),
        'density':                float(G.density()),
        'max_coreness':           int(coreness.max()) if n else 0,
        'is_weakly_connected':    is_weak,
        'is_strongly_connected':  is_strong,
        'num_weak_components':    int(len(weak_sz)),
        'num_strong_components':  int(len(strong_sz)),
        'frac_in_lcc_weak':       lcc_sz / n if n else 0.0,
        'frac_in_lscc':           int(strong_sz.max()) / n if n else 0.0,
    }

    if n > 1:
        sub = G if is_weak else G.induced_subgraph(np.where(weak_m == lcc_id)[0].tolist())
        start = np.random.randint(sub.vcount())
        d1 = sub.distances(source=start, mode="all")[0]
        far1 = int(np.argmax([x if x != float('inf') else -1 for x in d1]))
        d2 = sub.distances(source=far1, mode="all")[0]
        rec['approx_diameter'] = int(max(x for x in d2 if x != float('inf')))
    else:
        rec['approx_diameter'] = 0

    if n < BIG:
        und = G.as_undirected(mode="collapse")
        part = und.community_label_propagation()
        rec['modularity']      = float(und.modularity(part))
        rec['num_communities'] = len(set(part.membership))

        # Topology-DETECTED communities -- membership is already in G's own
        # node-index space (as_undirected() doesn't reorder vertices), so
        # this needs nothing external: no nodes.parquet, no community_id,
        # no reindexing. Measures the network's REALIZED structure.
        detected = np.asarray(part.membership, dtype=np.int64)
        rec.update(community_structure_stats(G, detected, label="detected_community"))
    else:
        rec['modularity']      = None
        rec['num_communities'] = None

    # Generative/ASSIGNED communities from asnu's community file, if
    # available (see load_community_compact) -- measures whether the
    # network's edges actually respect the community structure it was
    # generated to have, which can differ from what label propagation
    # detects above.
    if community_compact is not None and len(community_compact) == n and n > 0:
        rec.update(community_structure_stats(G, community_compact, label="assigned_community"))

    return rec


def load_community_compact(net_dir: Path, original_ids: np.ndarray) -> np.ndarray | None:
    """
    Read nodes.parquet's community_id column (indexed by ORIGINAL position
    0..N-1) and reindex it to the compact 0..k-1 space load_edges() uses,
    via original_ids[i] = original position of compact index i.

    Returns None (with a printed reason) if nodes.parquet doesn't exist,
    has no community_id column, or any referenced node is missing a
    community assignment -- callers should just skip community stats in
    that case rather than compute them on incomplete data.
    """
    nodes_path = net_dir / "nodes.parquet"
    if not nodes_path.exists():
        print(f"  [{net_dir.name}] no nodes.parquet, skipping community stats")
        return None

    nodes_df = pd.read_parquet(nodes_path)
    if "community_id" not in nodes_df.columns:
        print(f"  [{net_dir.name}] no community_id column, skipping community stats")
        return None

    community_full = nodes_df["community_id"].to_numpy()
    if len(original_ids) and original_ids.max() >= len(community_full):
        print(f"  [{net_dir.name}] original_ids out of range for nodes.parquet "
              f"({original_ids.max()} >= {len(community_full)}), skipping community stats")
        return None

    community_compact = community_full[original_ids]
    if pd.isna(community_compact).any():
        n_missing = int(pd.isna(community_compact).sum())
        print(f"  [{net_dir.name}] {n_missing}/{len(community_compact)} nodes missing "
              f"community_id, skipping community stats")
        return None

    return community_compact.astype(np.int64)


def existing_stats(network_id: str) -> dict:
    stats_path = data_lake.ROOT / 'networks' / network_id / 'network_stats.parquet'
    if not stats_path.exists():
        return {}
    df = pd.read_parquet(stats_path)
    return dict(zip(df['stat_name'], df['stat_value']))


def is_done(network_id: str) -> bool:
    """
    'global_clustering' is only ever written by this stage (never by
    generation), so its presence means the full metrics stage already ran
    -- distinct from just having the basic generation-stage stats.

    NOTE: this does NOT distinguish "ran before community_id existed" from
    "ran with community stats" -- if you want community stats backfilled
    onto networks processed before that feature existed, you'll need to
    reprocess them explicitly (delete network_stats.parquet, or add a
    --force flag) rather than relying on this check.
    """
    return 'global_clustering' in existing_stats(network_id)


def _write_error(net_dir: Path, message: str) -> None:
    (net_dir / 'metrics_error.json').write_text(json.dumps({"error": message}))
    print(f"  [{net_dir.name}] ERROR: {message}")


def process_one(network_id: str) -> None:
    if is_done(network_id):
        print(f"  [{network_id}] already done, skipping")
        return

    net_dir = data_lake.network_dir(network_id)
    npz_path = net_dir / 'edges.npz'
    if not npz_path.exists():
        print(f"  [{network_id}] no edges.npz found, skipping")
        return

    error_path = net_dir / 'metrics_error.json'

    try:
        edges, original_ids = load_edges(npz_path)
        if len(edges) == 0:
            _write_error(net_dir, "no edges")
            return
        community_compact = load_community_compact(net_dir, original_ids)
        rec = compute_metrics(edges, community_compact)
        del edges
        gc.collect()
    except Exception as e:
        _write_error(net_dir, f"{type(e).__name__}: {e}")
        return

    # Only numeric fields go into network_stats (float/int/bool -- bool
    # casts fine via float()). None entries (e.g. modularity skipped on BIG
    # graphs) are left out rather than coerced. Merge with whatever
    # generation already wrote (n_nodes, n_edges, gen_time_s) instead of
    # clobbering it -- write_network_stats() overwrites the whole file.
    numeric_rec = {k: v for k, v in rec.items() if v is not None}
    merged = {**existing_stats(network_id), **numeric_rec}
    data_lake.write_network_stats(network_id, merged)

    if error_path.exists():
        error_path.unlink()

    print(f"  [{network_id}] DONE: {rec['nodes']} nodes, {rec['edges']} edges", flush=True)


def discover_network_ids() -> list[str]:
    net_dir = data_lake.ROOT / 'networks'
    if not net_dir.exists():
        return []
    return sorted(p.name for p in net_dir.iterdir() if p.is_dir() and (p / 'edges.npz').exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_id', type=str, default=None,
                        help='Specific network_id to process. If omitted, uses --task_id.')
    parser.add_argument('--task_id', type=int, default=None,
                        help='Index into sorted list of all network_ids with edges.npz.')
    args = parser.parse_args()

    if args.network_id:
        process_one(args.network_id)
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --network_id, --task_id, or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    network_ids = discover_network_ids()
    if task_id >= len(network_ids):
        print(f"task_id {task_id} out of range ({len(network_ids)} networks). Exiting.")
        return

    process_one(network_ids[task_id])


if __name__ == '__main__':
    main()