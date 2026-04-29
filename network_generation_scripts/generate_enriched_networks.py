import json, gc, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import igraph as ig
from scipy import stats
from tqdm import tqdm


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
    _, inv = np.unique(edges.ravel(), return_inverse=True)
    return inv.reshape(-1, 2).astype(np.int32)


def metrics(edges):
    n = int(edges.max()) + 1
    G = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    G.simplify()
    part = G.community_multilevel()
    degrees = np.asarray(G.degree(), dtype=np.int64)
    return {
        "modularity": float(G.modularity(part)),
        "clustering": float(G.transitivity_undirected(mode="zero")),
        "skewness": float(stats.skew(degrees)),
        "nodes": n,
        "edges": G.ecount(),
        "mean_degree": float(degrees.mean()),
    }


def run(base_dir="my_networks", output_dir="network_metrics"):
    base = Path(base_dir).resolve()
    out = Path(output_dir); out.mkdir(exist_ok=True)
    files = sorted(base.rglob("*.npz"))
    print(f"Found {len(files)} files")

    results = []
    jsonl = open(out / "results.jsonl", "w")
    for f in tqdm(files):
        rel = f.relative_to(base)
        rec = {"rel_path": str(rel),
               "experiment": rel.parts[0],
               "subgroup": "/".join(rel.parts[1:-1])}
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
        jsonl.write(json.dumps(rec, default=float) + "\n"); jsonl.flush()
    jsonl.close()

    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    errs = sum("error" in r for r in results)
    print(f"Done. {len(results)-errs} ok, {errs} errors.")


if __name__ == "__main__":
    run()