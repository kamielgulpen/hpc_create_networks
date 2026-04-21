"""
Calibration via simulated annealing for enriched network generation.

For each aggregation level, finds (n_communities, preferential_attachment,
transitivity_p) that best reproduce the target real-network metrics:

    Transitivity = 0.4941  (avg local clustering coefficient)
    Modularity   = 0.8774  (label-propagation community detection)
    Degree_skew  = 7.0774  (skewness of degree distribution)

Each task handles one aggregation level (one pop/links pair).

Usage:
    python calibrate.py --list_tasks               # print number of tasks
    python calibrate.py --task_id N                # run SA for pair N
    python calibrate.py --task_id N --max_iter 80  # more iterations
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy import stats

from asnu import generate, create_communities


# =============================================================================
# Targets  (from the real network)
# =============================================================================

TARGET_TRANSITIVITY = 0.4941   # transitivity_avglocal_undirected(mode="zero")
TARGET_MODULARITY   = 0.8774   # community_label_propagation modularity
TARGET_DEGREE_SKEW  = 7.0774   # stats.skew of degree distribution


# =============================================================================
# Fixed generation parameters  (same as run_task.py)
# =============================================================================

ENRICHED_AGG_DIR = Path('Data/Data/enriched/aggregated')
SCALE            = 1
RECIPROCITY_P    = 1
BRIDGE_PROB      = 0.2


# =============================================================================
# Per-level SA search bounds
# Edit these to restrict or widen the search for each aggregation level.
# Keys must match the combo_str derived from each pop_*.csv filename.
#
# Fields:
#   pa_min, pa_max        — preferential attachment range  [0.0, 0.9999]
#   comms_min, comms_max  — n_communities range            [1, ...]
#   trans_min, trans_max  — transitivity_p range           [0.0, 1.0]
# =============================================================================

LEVEL_BOUNDS = {
    'geslacht': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1000, 'comms_max': 50000,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'lft_oplniv': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 500, 'comms_max': 10000,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'etngrp_geslacht': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 500, 'comms_max': 20000,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'geslacht_lft_oplniv': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 100, 'comms_max': 1000,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'etngrp_geslacht_lft_oplniv': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1, 'comms_max': 200,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'etngrp_lft_inkomensniveau_arbeidsstatus_burgerlijke_staat': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1, 'comms_max': 200,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'etngrp_geslacht_lft_oplniv_burgerlijke_staat': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1, 'comms_max': 200,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'geslacht_lft_oplniv_inkomensniveau_arbeidsstatus_uitkeringstype': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1, 'comms_max': 200,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'lft_inkomensniveau_arbeidsstatus_uitkeringstype_burgerlijke_staat': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1, 'comms_max': 200,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
    'etngrp_geslacht_lft_oplniv_inkomensniveau_arbeidsstatus_uitkeringstype_burgerlijke_staat': {
        'pa_min': 0.0, 'pa_max': 0.9999,
        'comms_min': 1, 'comms_max': 200,
        'trans_min': 0.0, 'trans_max': 1.0,
    },
}

# Fallback bounds used if a level is not listed in LEVEL_BOUNDS above
DEFAULT_BOUNDS = {
    'pa_min': 0.0, 'pa_max': 0.9999,
    'comms_min': 1, 'comms_max': 200,
    'trans_min': 0.0, 'trans_max': 1.0,
}


# =============================================================================
# Helpers
# =============================================================================

def discover_enriched_pairs():
    pairs = []
    for pop_file in sorted(ENRICHED_AGG_DIR.glob('pop_*.csv')):
        combo_str  = pop_file.stem[len('pop_'):]
        links_file = ENRICHED_AGG_DIR / f'interactions_{combo_str}.csv'
        if links_file.exists():
            pairs.append((f'enriched/{combo_str}', str(pop_file), str(links_file)))
    return pairs


def nx_to_igraph(nx_graph):
    nodes = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=nx_graph.is_directed())
    if nodes:
        for attr in nx_graph.nodes[nodes[0]].keys():
            ig_graph.vs[attr] = [nx_graph.nodes[n].get(attr) for n in nodes]
    ig_graph.vs["name"] = nodes
    return ig_graph

def robust_label_prop(graph, timeout_s=60):
    """
    Run label propagation with a timeout. Returns None on any failure.

    Failure modes handled:
      - Hang / infinite loop  -> SIGALRM fires, returns None
      - Any Python exception  -> caught, returns None
      - Empty / trivial graph -> returns None
      - Invalid result        -> returns None

    Not handled (requires subprocess isolation):
      - C-level crashes (segfault, double free, abort) kill the process
    """
    import signal

    # Guard against degenerate graphs that can trip up C code
    if graph.vcount() == 0 or graph.ecount() == 0:
        return None

    class Timeout(Exception):
        pass

    def handler(signum, frame):
        raise Timeout()

    # Save any existing handler so we restore it cleanly
    prev_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_s)

    try:
        part = graph.community_label_propagation()
        # Sanity-check the result before returning
        if part is None or len(part.membership) != graph.vcount():
            return None
        return part
    except Timeout:
        return None
    except Exception:
        # Any other Python-level failure from igraph: log-and-continue
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def compute_metrics(G_ig):
    """
    Compute the three calibration metrics.
    Returns NaN for modularity if label propagation fails.
    """
    degrees      = np.asarray(G_ig.degree(), dtype=np.int32)
    transitivity = float(G_ig.transitivity_avglocal_undirected(mode="zero"))

    part = robust_label_prop(G_ig)
    if part is None:
        modularity = float("nan")
    else:
        try:
            modularity = float(G_ig.modularity(part))
        except Exception:
            modularity = float("nan")

    try:
        degree_skew = float(stats.skew(degrees))
    except Exception:
        degree_skew = float("nan")

    return transitivity, modularity, degree_skew

def loss(transitivity, modularity, degree_skew):
    """
    Normalized relative squared error across the three metrics.
    Each term is dimensionless so all three have equal weight.
    """
    return (
        ((transitivity - TARGET_TRANSITIVITY) / TARGET_TRANSITIVITY) ** 2
        + ((modularity   - TARGET_MODULARITY)   / TARGET_MODULARITY)   ** 2
        + ((degree_skew  - TARGET_DEGREE_SKEW)  / TARGET_DEGREE_SKEW)  ** 2
    )


def evaluate(n_comms, pref_att, trans_p, label, pops, links):
    """
    Generate one network and return its metrics.
    Uses temp files for both the communities JSON and the graph output so
    nothing accumulates on disk between SA iterations.

    Returns
    -------
    transitivity, modularity, degree_skew, loss_value, elapsed_s
    """
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_comm:
        communities_path = tmp_comm.name

    with tempfile.TemporaryDirectory() as tmp_net_dir:
        base_path = str(Path(tmp_net_dir) / 'net')
        start = time.perf_counter()
        try:
            create_communities(
                pops, links,
                scale=SCALE,
                number_of_communities=n_comms,
                output_path=communities_path,
                mode='capacity_capacity',
                allow_new_communities=False,
                verbose=False,
            )
            graph = generate(
                pops, links,
                preferential_attachment=pref_att,
                scale=SCALE,
                reciprocity=RECIPROCITY_P,
                transitivity=trans_p,
                community_file=communities_path,
                base_path=base_path,
                bridge_probability=BRIDGE_PROB,
                fully_connect_communities=False,
                fill_unfulfilled=True,
                verbose=False,
            )
        finally:
            try:
                os.unlink(communities_path)
            except OSError:
                pass

        elapsed = time.perf_counter() - start
        G_ig = nx_to_igraph(graph.graph)

    t, m, d = compute_metrics(G_ig)
    l = loss(t, m, d)
    return t, m, d, l, elapsed


# =============================================================================
# Simulated Annealing
# =============================================================================

def simulated_annealing(label, pops, links, bounds, max_iter=60, t0=1.0, cooling=0.97, seed=42):
    """
    Minimise loss(n_communities, pref_attachment, transitivity_p) for one
    aggregation level.

    Neighbourhood moves  (all applied each iteration)
    -------------------
    - n_communities  : current ± N(0, sigma_c),     sigma_c   = max(2, range * 0.1 * T)
    - pref_attachment: current ± N(0, sigma_pa),    sigma_pa  = max(range*0.01, range*0.15*T)
    - transitivity_p : current ± N(0, sigma_trans), sigma_trans = max(0.01, 0.15 * T)

    Returns
    -------
    best_params : dict
    trajectory  : list of dicts  (one row per evaluated point)
    """
    rng = np.random.RandomState(seed)

    pa_min    = bounds['pa_min']
    pa_max    = bounds['pa_max']
    comms_min = bounds['comms_min']
    comms_max = bounds['comms_max']
    trans_min = bounds['trans_min']
    trans_max = bounds['trans_max']

    # ── Initial solution: midpoint of the search bounds ───────────────────────
    cur_comms = int((comms_min + comms_max) / 2)
    cur_pa    = (pa_min + pa_max) / 2
    cur_trans = (trans_min + trans_max) / 2

    print(f"  Bounds: comms=[{comms_min}, {comms_max}]  "
          f"pa=[{pa_min}, {pa_max}]  trans=[{trans_min}, {trans_max}]")
    print(f"  Initial point: comms={cur_comms}, pa={cur_pa:.4f}, trans={cur_trans:.4f}")
    cur_t, cur_m, cur_d, cur_loss, elapsed = evaluate(
        cur_comms, cur_pa, cur_trans, label, pops, links
    )
    print(f"    loss={cur_loss:.4f}  T={cur_t:.4f}  M={cur_m:.4f}  D={cur_d:.4f}  ({elapsed:.1f}s)")

    best_comms, best_pa, best_trans = cur_comms, cur_pa, cur_trans
    best_t, best_m, best_d         = cur_t, cur_m, cur_d
    best_loss                      = cur_loss

    trajectory = [{
        'iteration':       0,
        'n_communities':   cur_comms,
        'pref_attachment': cur_pa,
        'trans_param':     cur_trans,
        'metric_transit':  cur_t,
        'modularity':      cur_m,
        'degree_skew':     cur_d,
        'loss':            cur_loss,
        'accepted':        True,
        'temperature':     t0,
        'elapsed_s':       elapsed,
    }]

    # ── SA loop ───────────────────────────────────────────────────────────────
    T = t0
    for iteration in range(1, max_iter + 1):
        T *= cooling

        comms_range = comms_max - comms_min
        pa_range    = pa_max - pa_min
        trans_range = trans_max - trans_min

        sigma_c     = max(2.0, comms_range * 0.1 * T)
        sigma_pa    = max(pa_range * 0.01,    pa_range    * 0.15 * T)
        sigma_trans = max(trans_range * 0.01, trans_range * 0.15 * T)

        new_comms = int(np.clip(
            round(cur_comms + rng.normal(0, sigma_c)),
            comms_min, comms_max,
        ))
        new_pa    = float(np.clip(cur_pa    + rng.normal(0, sigma_pa),    pa_min,    pa_max))
        new_trans = float(np.clip(cur_trans + rng.normal(0, sigma_trans), trans_min, trans_max))

        print(f"  Iter {iteration:3d}/{max_iter}  T={T:.4f}  "
              f"trying comms={new_comms}, pa={new_pa:.4f}, trans={new_trans:.4f}", flush=True)

        new_t, new_m, new_d, new_loss, elapsed = evaluate(
            new_comms, new_pa, new_trans, label, pops, links
        )

        delta    = new_loss - cur_loss
        accepted = delta < 0 or rng.random() < np.exp(-delta / (T + 1e-12))

        if accepted:
            cur_comms, cur_pa, cur_trans  = new_comms, new_pa, new_trans
            cur_t, cur_m, cur_d, cur_loss = new_t, new_m, new_d, new_loss

        if new_loss < best_loss:
            best_comms, best_pa, best_trans = new_comms, new_pa, new_trans
            best_t, best_m, best_d         = new_t, new_m, new_d
            best_loss                      = new_loss

        flag = "A" if accepted else "R"
        print(f"    [{flag}] loss={new_loss:.4f}  T={new_t:.4f}  M={new_m:.4f}  "
              f"D={new_d:.4f}  best={best_loss:.4f}  ({elapsed:.1f}s)", flush=True)

        trajectory.append({
            'iteration':       iteration,
            'n_communities':   new_comms,
            'pref_attachment': new_pa,
            'trans_param':     new_trans,
            'metric_transit':  new_t,
            'modularity':      new_m,
            'degree_skew':     new_d,
            'loss':            new_loss,
            'accepted':        accepted,
            'temperature':     T,
            'elapsed_s':       elapsed,
        })

    best_params = {
        'label':               label,
        'n_communities':       int(best_comms),
        'pref_attachment':     round(best_pa, 4),
        'transitivity_param':  round(best_trans, 4),
        'best_loss':           round(best_loss, 6),
        'metric_transit':      round(best_t, 4),
        'modularity':          round(best_m, 4),
        'degree_skew':         round(best_d, 4),
        'target_transitivity': TARGET_TRANSITIVITY,
        'target_modularity':   TARGET_MODULARITY,
        'target_degree_skew':  TARGET_DEGREE_SKEW,
        'comms_min':           comms_min,
        'comms_max':           comms_max,
        'pa_min':              pa_min,
        'pa_max':              pa_max,
        'trans_min':           trans_min,
        'trans_max':           trans_max,
        'max_iter':            max_iter,
        'cooling':             cooling,
        't0':                  t0,
        'seed':                seed,
    }
    return best_params, trajectory


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SA calibration for one enriched aggregation level"
    )
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task index (0-based). Defaults to SLURM_ARRAY_TASK_ID env var.')
    parser.add_argument('--list_tasks', action='store_true',
                        help='Print number of aggregation levels and exit.')
    parser.add_argument('--max_iter', type=int, default=60,
                        help='SA budget — evaluations after the initial point (default 60).')
    parser.add_argument('--output_dir', type=str, default='results/calibration')
    args = parser.parse_args()

    pairs = discover_enriched_pairs()

    if args.list_tasks:
        print(len(pairs))
        return

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if slurm_id is None:
            raise RuntimeError("Provide --task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(slurm_id)

    if task_id >= len(pairs):
        print(f"Task {task_id} out of range (only {len(pairs)} pairs). Exiting.")
        return

    label, pops, links = pairs[task_id]
    combo_name = Path(label).name
    bounds     = LEVEL_BOUNDS.get(combo_name, DEFAULT_BOUNDS)

    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_file = out_dir / f'{combo_name}_best.json'
    traj_file = out_dir / f'{combo_name}_trajectory.csv'

    if best_file.exists():
        print(f"Already calibrated: {best_file}. Skipping.")
        return

    print(f"Task {task_id}/{len(pairs) - 1}: calibrating [{label}]")
    print(f"Targets — T={TARGET_TRANSITIVITY}  M={TARGET_MODULARITY}  D={TARGET_DEGREE_SKEW}")
    print(f"SA budget: {args.max_iter} iterations  (+ 1 initial evaluation)")

    best_params, trajectory = simulated_annealing(
        label, pops, links,
        bounds=bounds,
        max_iter=args.max_iter,
    )

    with open(best_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    pd.DataFrame(trajectory).to_csv(traj_file, index=False)

    print(f"\nBest found:")
    print(f"  n_communities    = {best_params['n_communities']}")
    print(f"  pref_attachment  = {best_params['pref_attachment']}")
    print(f"  transitivity_p   = {best_params['transitivity_param']}")
    print(f"  loss             = {best_params['best_loss']}")
    print(f"  metric_transit   = {best_params['metric_transit']}  (target {TARGET_TRANSITIVITY})")
    print(f"  modularity       = {best_params['modularity']}  (target {TARGET_MODULARITY})")
    print(f"  degree_skew      = {best_params['degree_skew']}  (target {TARGET_DEGREE_SKEW})")
    print(f"\nSaved: {best_file}")
    print(f"Saved: {traj_file}")
    print(f"Task {task_id} complete.")


if __name__ == '__main__':
    main()
