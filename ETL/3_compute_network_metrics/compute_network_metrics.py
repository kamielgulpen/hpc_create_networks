import itertools

import igraph as ig
import numpy as np
from scipy import stats


def _edge_array(G) -> np.ndarray:
    """(m, 2) int64 directed edge endpoints, extracted the fast way.

    Replaces `np.array(G.get_edgelist(), dtype=np.int64)`. get_edgelist()
    yields a list of (src, dst) tuples; chaining them into one flat stream
    and using fromiter with an explicit count skips NumPy's per-tuple shape
    inference. ~4x faster at ~6M edges (1.8s vs 7.4s), bit-identical result.
    """
    m = G.ecount()
    if m == 0:
        return np.empty((0, 2), dtype=np.int64)
    flat = np.fromiter(
        itertools.chain.from_iterable(G.get_edgelist()),
        dtype=np.int64,
        count=m * 2,
    )
    return flat.reshape(m, 2)

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
def community_structure_stats(
    G,
    community_compact: np.ndarray,
    label: str = "community",
    edges_arr: np.ndarray | None = None,
) -> dict:
    """
    Inter/intra-community edge structure for ANY per-node community
    assignment aligned with G's node indices -- doesn't care whether that
    assignment came from asnu's community file (the network's intended/
    generative structure) or from topology-based community detection
    (community_label_propagation()'s membership, the network's realized
    structure). Call this twice with different `label`s to compare both.

    Pass `edges_arr` (the (m,2) int64 directed edgelist from _edge_array)
    to avoid re-extracting it from G on every call -- compute_metrics does
    this once and hands the same array to both invocations. If omitted, it
    is extracted here (same result, just not shared).

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

    if edges_arr is None:
        edges_arr = _edge_array(G)
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

    # Extract the directed edgelist ONCE here; both community_structure_stats
    # calls below reuse it instead of re-extracting from G each time.
    edges_arr = _edge_array(G)
    BIG = 900_000  # threshold for community detection; label propagation is O(n^2) in memory
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
        rec.update(community_structure_stats(G, detected, label="detected_community",
                                             edges_arr=edges_arr))
    else:
        rec['modularity']      = None
        rec['num_communities'] = None

    # Generative/ASSIGNED communities from asnu's community file, if
    # available (see load_community_compact) -- measures whether the
    # network's edges actually respect the community structure it was
    # generated to have, which can differ from what label propagation
    # detects above.
    if community_compact is not None and len(community_compact) == n and n > 0:
        rec.update(community_structure_stats(G, community_compact, label="assigned_community",
                                             edges_arr=edges_arr))

    return rec