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

    edges.sort(axis=1)

    packed = (edges[:, 0].astype(np.uint64) << 32) | edges[:, 1].astype(np.uint64)
    packed = np.unique(packed)
    edges = np.empty((len(packed), 2), dtype=np.int32)
    edges[:, 0] = (packed >> 32).astype(np.int32)
    edges[:, 1] = (packed & 0xFFFFFFFF).astype(np.int32)

    _, inv = np.unique(edges.ravel(), return_inverse=True)
    return inv.reshape(-1, 2).astype(np.int32)


def metrics(edges):
    n = int(edges.max()) + 1
    G = ig.Graph(n=n, directed=False)
    print("edges are loaded, computing metrics...", flush=True)
    try:
        G.add_edges(edges)
    except TypeError:
        G.add_edges(edges.tolist())

    degrees = np.asarray(G.degree(), dtype=np.int32)

    clusters = G.connected_components()
    membership = np.asarray(clusters.membership)
    sizes = np.bincount(membership)
    is_connected = len(sizes) == 1
    lcc_id = int(np.argmax(sizes))
    lcc_size = int(sizes[lcc_id])

    BIG = 900_000

    rec = {
        "clustering": float(G.transitivity_avglocal_undirected(mode="zero")),
        "skewness": float(stats.skew(degrees)),
        "log_degree_skewness": float(stats.skew(np.log1p(degrees))),
        "nodes": n,
        "edges": G.ecount(),
        "mean_degree": float(degrees.mean()),
        "q25_degree": float(np.quantile(degrees, 0.25)) if n else 0.0,
        "q75_degree": float(np.quantile(degrees, 0.75)) if n else 0.0,
        "skew_degree": float(stats.skew(degrees)) if n else 0.0,
        "frac_isolates": float((degrees == 0).mean()) if n else 0.0,
        "frac_degree_1": float((degrees == 1).mean()) if n else 0.0,
        "is_connected": is_connected,
        "num_components": int(len(sizes)),
        "frac_in_lcc": lcc_size / n if n else 0.0,
        "frac_articulation_points": (
            len(G.articulation_points()) / n if n and n < BIG else None
        ),
    }

    if n > 1:
        if is_connected:
            sub = G
        else:
            lcc_nodes = np.where(membership == lcc_id)[0].tolist()
            sub = G.induced_subgraph(lcc_nodes)
        start = np.random.randint(sub.vcount())
        d1 = sub.distances(source=start)[0]
        far1 = int(np.argmax([x if x != float('inf') else -1 for x in d1]))
        d2 = sub.distances(source=far1)[0]
        rec["approx_diameter"] = int(max(x for x in d2 if x != float('inf')))
    else:
        rec["approx_diameter"] = 0

    if n < BIG:
        part = G.community_label_propagation()
        rec["modularity"] = float(G.modularity(part))
        rec["modularity_method"] = "label_propagation"
    else:
        rec["modularity"] = None
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