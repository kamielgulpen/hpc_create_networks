"""
Stage 2: Compute network metrics from saved .npz edge files.

Reads networks from pawn_results/networks/sample_XXXXX/<label>/edges.npz,
computes metrics, writes one JSON per (sample, label) into
pawn_results/metrics/.

Run aggregate_pawn_metrics.py afterwards to merge with samples.csv into
results.csv ready for analyze_pawn.py.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import igraph as ig
import numpy as np
from scipy import stats


NETWORKS_DIR = Path('pawn_results/networks')
METRICS_DIR  = Path('pawn_results/metrics')
BIG          = 900_000


def load_edges(npz_file):
    with np.load(npz_file, allow_pickle=True) as data:
        arr = np.asarray(data[list(data.keys())[0]])
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"bad shape {arr.shape}")

    edges = np.ascontiguousarray(arr, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return edges.astype(np.int32)

    # Directed: don't sort axis=1
    packed = (edges[:, 0].astype(np.uint64) << 32) | edges[:, 1].astype(np.uint64)
    packed = np.unique(packed)
    edges = np.empty((len(packed), 2), dtype=np.int32)
    edges[:, 0] = (packed >> 32).astype(np.int32)
    edges[:, 1] = (packed & 0xFFFFFFFF).astype(np.int32)

    _, inv = np.unique(edges.ravel(), return_inverse=True)
    return inv.reshape(-1, 2).astype(np.int32)


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


def compute_metrics(edges):
    n = int(edges.max()) + 1
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
        'frac_isolates':          float((tot_deg == 0).mean()),
        'frac_sources':           float((in_deg  == 0).mean()),
        'frac_sinks':             float((out_deg == 0).mean()),
        'frac_degree_1':          float((tot_deg == 1).mean()),
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
    else:
        rec['modularity']      = None
        rec['num_communities'] = None

    return rec


def process_one(npz_path):
    rel = npz_path.relative_to(NETWORKS_DIR)
    sample_dir = rel.parts[0]               # sample_XXXXX
    label      = '/'.join(rel.parts[1:-1])  # label
    safe = f"{sample_dir}__{label.replace('/', '__')}"
    out_file = METRICS_DIR / f"{safe}.json"

    if out_file.exists():
        return  # already done

    rec = {'sample_dir': sample_dir, 'label': label}
    try:
        edges = load_edges(npz_path)
        if len(edges) == 0:
            rec['error'] = 'no edges'
        else:
            rec.update(compute_metrics(edges))
            del edges
            gc.collect()
    except Exception as e:
        rec['error'] = f"{type(e).__name__}: {e}"

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(rec, f, default=float)
    print(f"DONE: {safe}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_file', type=str, default=None,
                        help='Specific edges.npz to process. If omitted, uses --task_id.')
    parser.add_argument('--task_id', type=int, default=None,
                        help='Index into sorted list of all edges.npz under networks/.')
    args = parser.parse_args()

    if args.npz_file:
        process_one(Path(args.npz_file))
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --npz_file, --task_id, or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    files = sorted(NETWORKS_DIR.rglob('edges.npz'))
    if task_id >= len(files):
        print(f"task_id {task_id} out of range ({len(files)} files). Exiting.")
        return

    process_one(files[task_id])


if __name__ == '__main__':
    main()