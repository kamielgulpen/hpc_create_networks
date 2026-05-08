import argparse
import json
import gc
from pathlib import Path
import numpy as np
import igraph as ig
from scipy import stats


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

    # edges.sort(axis=1)

    packed = (edges[:, 0].astype(np.uint64) << 32) | edges[:, 1].astype(np.uint64)
    packed = np.unique(packed)
    edges = np.empty((len(packed), 2), dtype=np.int32)
    edges[:, 0] = (packed >> 32).astype(np.int32)
    edges[:, 1] = (packed & 0xFFFFFFFF).astype(np.int32)

    _, inv = np.unique(edges.ravel(), return_inverse=True)
    return inv.reshape(-1, 2).astype(np.int32)


def metrics(edges):
    n = int(edges.max()) + 1
    G = ig.Graph(n=n, directed=True)
    print("edges are loaded, computing metrics...", flush=True)
    try:
        G.add_edges(edges)
    except TypeError:
        G.add_edges(edges.tolist())

    in_deg = np.asarray(G.indegree(), dtype=np.int32)
    out_deg = np.asarray(G.outdegree(), dtype=np.int32)
    tot_deg = in_deg + out_deg

    local_clust = np.asarray(
        G.transitivity_local_undirected(mode="zero"), dtype=np.float64
    )
    coreness = np.asarray(G.coreness(mode="all"), dtype=np.int32)
    pagerank = np.asarray(G.pagerank(directed=True), dtype=np.float64)
    knn_vals, _ = G.knn()
    knn = np.asarray([v if v is not None else 0.0 for v in knn_vals], dtype=np.float64)

    # Connectivity
    clusters = G.connected_components(mode="weak")
    membership = np.asarray(clusters.membership)
    sizes = np.bincount(membership)
    is_weakly_connected = len(sizes) == 1
    lcc_id = int(np.argmax(sizes))
    lcc_size = int(sizes[lcc_id])

    strong_clusters = G.connected_components(mode="strong")
    strong_sizes = np.bincount(strong_clusters.membership)
    is_strongly_connected = len(strong_sizes) == 1
    largest_scc = int(strong_sizes.max())

    BIG = 900_000

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

    rec = {
        "nodes": n,
        "edges": G.ecount(),
        # Node-level distribution stats
        **dist_stats(in_deg, "in_degree"),
        **dist_stats(out_deg, "out_degree"),
        **dist_stats(tot_deg, "total_degree"),
        **dist_stats(np.log1p(in_deg), "log_in_degree"),
        **dist_stats(np.log1p(out_deg), "log_out_degree"),
        **dist_stats(local_clust, "local_clustering"),
        **dist_stats(coreness, "coreness"),
        **dist_stats(pagerank, "pagerank"),
        **dist_stats(knn, "avg_neighbor_degree"),
        # Special-node fractions
        "frac_isolates": float((tot_deg == 0).mean()),
        "frac_sources":  float((in_deg == 0).mean()),
        "frac_sinks":    float((out_deg == 0).mean()),
        "frac_degree_1": float((tot_deg == 1).mean()),
        # Global structural metrics
        "global_clustering":     float(G.transitivity_undirected(mode="zero")),
        "avg_local_clustering":  float(G.transitivity_avglocal_undirected(mode="zero")),
        "reciprocity":           float(G.reciprocity()),
        "density":               float(G.density()),
        "max_coreness":          int(coreness.max()) if n else 0,
        # Connectivity
        "is_weakly_connected":   is_weakly_connected,
        "is_strongly_connected": is_strongly_connected,
        "num_weak_components":   int(len(sizes)),
        "num_strong_components": int(len(strong_sizes)),
        "frac_in_lcc_weak":      lcc_size / n if n else 0.0,
        "frac_in_lscc":          largest_scc / n if n else 0.0,
    }

    # Approx diameter on largest weakly-connected component
    if n > 1:
        if is_weakly_connected:
            sub = G
        else:
            lcc_nodes = np.where(membership == lcc_id)[0].tolist()
            sub = G.induced_subgraph(lcc_nodes)
        start = np.random.randint(sub.vcount())
        d1 = sub.distances(source=start, mode="all")[0]
        far1 = int(np.argmax([x if x != float('inf') else -1 for x in d1]))
        d2 = sub.distances(source=far1, mode="all")[0]
        rec["approx_diameter"] = int(max(x for x in d2 if x != float('inf')))
    else:
        rec["approx_diameter"] = 0

    # Modularity
    if n < BIG:
        und = G.as_undirected(mode="collapse")
        part = und.community_label_propagation()
        rec["modularity"] = float(und.modularity(part))
        rec["num_communities"] = len(set(part.membership))
        rec["modularity_method"] = "label_propagation_undirected"
    else:
        rec["modularity"] = None
        rec["num_communities"] = None
        rec["modularity_method"] = "skipped_large_graph"

    print(rec, flush=True)
    return rec


def process_file(npz_path, base_dir, output_dir):
    npz_path = Path(npz_path).resolve()
    base = Path(base_dir).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        rel = npz_path.relative_to(base)
    except ValueError:
        rel = Path(npz_path.name)

    rec = {
        "rel_path": str(rel),
        "experiment": rel.parts[0] if len(rel.parts) > 1 else "",
        "subgroup": "/".join(rel.parts[1:-1]),
        "filename": npz_path.stem,
    }

    try:
        if npz_path.stat().st_size == 0:
            rec["error"] = "empty file"
        else:
            edges = load_edges(npz_path)
            if len(edges) == 0:
                rec["error"] = "no edges"
            else:
                rec.update(metrics(edges))
                del edges
                gc.collect()
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"

    # Per-file output - safe for parallel writes
    safe_name = str(rel).replace("/", "__").replace("\\", "__")
    out_file = out / f"{safe_name}.json"
    with open(out_file, "w") as f:
        json.dump(rec, f, default=float)

    print(f"DONE: {rel}", flush=True)
    print(rec, flush=True)
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", required=True, help="Path to .npz edge file")
    parser.add_argument("--base_dir", default="my_networks", help="Base dir for relative paths")
    parser.add_argument("--output_dir", default="network_metrics/per_file", help="Where to write per-file JSON")
    args = parser.parse_args()

    process_file(args.npz_file, args.base_dir, args.output_dir)