import json, gc, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import igraph as ig
from scipy import stats
from tqdm import tqdm

LOUVAIN_EDGE_LIMIT = 5_000


def load_edges(npz_file):
    with np.load(npz_file, allow_pickle=True) as data:
        arr = np.asarray(data[list(data.keys())[0]])

    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"bad shape {arr.shape}")

    edges = arr.astype(np.int32, copy=False)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return edges

    np.sort(edges, axis=1)
    # Memory-lean dedupe via structured view
    view = edges.view([("u", edges.dtype), ("v", edges.dtype)]).ravel()
    view = np.unique(view)
    edges = view.view(edges.dtype).reshape(-1, 2)
    del view

    _, inv = np.unique(edges.ravel(), return_inverse=True)
    return inv.reshape(-1, 2).astype(np.int32)


def metrics(edges):
    n = int(edges.max()) + 1
    G = ig.Graph(n=n, directed=False)
    try:
        G.add_edges(edges)            # numpy array, no .tolist()
    except TypeError:
        G.add_edges(edges.tolist())   # fallback for older igraph
    degrees = np.asarray(G.degree(), dtype=np.int32)
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
    }
    
    # Average path length (stratified sampling for 800k nodes)
    n = 0
    if n > 1:
        if G.is_connected():
            component = list(range(n))
        else:
            clusters = G.clusters()
            component = max(clusters, key=len)
        
        # Stratified sampling: low, medium, high degree nodes
        degrees_subset = degrees[component]
        low_deg = np.where(degrees_subset <= np.quantile(degrees_subset, 0.33))[0]
        mid_deg = np.where((degrees_subset > np.quantile(degrees_subset, 0.33)) & 
                           (degrees_subset <= np.quantile(degrees_subset, 0.67)))[0]
        high_deg = np.where(degrees_subset > np.quantile(degrees_subset, 0.67))[0]
        
        sample_per_stratum = 100  # 900 total samples
        sampled_idx = np.concatenate([
            np.random.choice(low_deg, size=min(sample_per_stratum, len(low_deg)), replace=False),
            np.random.choice(mid_deg, size=min(sample_per_stratum, len(mid_deg)), replace=False),
            np.random.choice(high_deg, size=min(sample_per_stratum, len(high_deg)), replace=False),
        ])
        sampled_nodes = [component[i] for i in sampled_idx]
        
        # Compute shortest paths from sampled nodes
        path_lengths = []
        for node in sampled_nodes:
            paths = G.shortest_paths_dijkstra(source=node)[0]
            finite_paths = [p for p in paths if p != float('inf') and p > 0]
            if finite_paths:
                path_lengths.extend(finite_paths)
        
        rec["avg_path_length"] = float(np.mean(path_lengths)) if path_lengths else 0.0
        rec["avg_path_length_sample_size"] = len(sampled_nodes)
    else:
        rec["avg_path_length"] = 0.0
        rec["avg_path_length_sample_size"] = 0
    
    print("GO")
    part = G.community_label_propagation()
    rec["modularity"] = float(G.modularity(part))
    rec["modularity_method"] = "label_propagation"
    return rec

def run(base_dir="my_networks", output_dir="network_metrics"):
    base = Path(base_dir).resolve()
    out = Path(output_dir); out.mkdir(exist_ok=True)
    files = sorted(base.rglob("*.npz"))
    print(f"Found {len(files)} files")

    results = []
    with open(out / "results.jsonl", "w") as jsonl:
        for f in tqdm(files):
            rel = f.relative_to(base)
            rec = {
                "rel_path": str(rel),
                "experiment": rel.parts[0] if rel.parts else "",
                "subgroup": "/".join(rel.parts[1:-1]),
                "filename": f.stem,
            }
            try:
                if f.stat().st_size == 0:
                    rec["error"] = "empty file"
                else:
                    edges = load_edges(f)
                    if len(edges) == 0:
                        rec["error"] = "no edges"
                    else:
                        rec.update(metrics(edges))
                        del edges
                        gc.collect()
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            results.append(rec)
            jsonl.write(json.dumps(rec, default=float) + "\n")
            jsonl.flush()

    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    errs = sum("error" in r for r in results)
    print(f"Done. {len(results)-errs} ok, {errs} errors.")


if __name__ == "__main__":
    run()