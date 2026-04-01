"""
Standalone validation of the household community algorithm — convergent rewrite.

Root cause of the original's poor fit:
  • p and T were never symmetrised (cross-group pairs set in one direction only).
  • Every group's multinomial draw was independent: zero inter-group coordination.
  • Single pass, no feedback loop between actual and target interaction counts.

This version fixes all three with a two-phase algorithm:

  Phase 1 – Sequential conditional init
    Groups are placed in decreasing-size order.  Each group's multinomial weight
    for community k is proportional to the affinity accumulated from all groups
    already placed there, seeding the cross-group correlations that independent
    draws cannot produce.

  Phase 2 – Error-corrected Gibbs sweeps
    Each group's column is resampled conditioned on all other groups' current
    placements, with weights scaled by sqrt(T / A) — a dampened correction that
    steers every pair toward its target without overshooting.

Group totals are maintained exactly throughout (multinomial draws always sum to
N[g]); no post-hoc rounding is needed.

Run from the repo root:
    python network_generation_scripts/validate_household_algorithm.py
"""

import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict

from asnu.core.graph import NetworkXGraph
from asnu.core.generate import init_nodes, _compute_maximum_num_links
from asnu.core.community import connect_all_within_communities
from asnu import check_group_interactions, plot_group_interactions

# ── Config ──────────────────────────────────────────────────────────────────
POPS_PATH      = 'Data/tab_n_(with oplniv).csv'
LINKS_PATH     = 'Data/tab_huishouden.csv'
SCALE          = 0.1
N_COMMUNITIES  = 50_000
N_GIBBS_ITER   = 15       # sweeps; 10–20 is usually sufficient
SEED           = 42
OUTPUT_SCATTER = 'household_scatter.png'
OUTPUT_HEATMAP = 'household_heatmap.png'


# ── Helpers ──────────────────────────────────────────────────────────────────

def _actual_interactions(C: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Actual interaction counts from integer composition matrix C (K × G).

        A[i, j] = Σ_k  C[k,i] · C[k,j]           cross-group  (i ≠ j)
        A[i, i] = Σ_k  C[k,i] · (C[k,i] − 1)     within-group pairs
    """
    Cf = C.astype(np.float64)
    A  = Cf.T @ Cf                              # C^T C  →  (G, G)
    np.fill_diagonal(A, np.diag(A) - N)         # adjust diagonal for ordered pairs
    return A


def _mean_rel_error(A: np.ndarray, T: np.ndarray) -> float:
    """Mean |A − T| / T over entries where T > 0."""
    mask = T > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(A[mask] - T[mask]) / T[mask]))


# ── Algorithm ────────────────────────────────────────────────────────────────

def populate_communities_households(
    G,
    num_communities: int | None = None,
    n_iter: int = 15,
    seed: int | None = None,
) -> None:
    """
    Convergent household community assignment.

    Finds integer composition matrix C (K × G) such that:

        _actual_interactions(C, N)  ≈  T         (entry-wise)
        C[:, g].sum()  ==  N[g]                  (exact, by construction)

    where T[g1, g2] = target edge count from G.maximum_num_links.

    Parameters
    ----------
    G               : NetworkXGraph, mutated in-place
    num_communities : override K (default: inferred from avg household size)
    n_iter          : number of Gibbs refinement sweeps
    seed            : numpy RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    group_ids   = sorted(G.group_ids)
    n_groups    = len(group_ids)
    g_idx       = {g: i for i, g in enumerate(group_ids)}
    N           = np.array(
        [len(G.group_to_nodes.get(g, [])) for g in group_ids], dtype=np.float64
    )
    total_nodes = int(N.sum())

    # ── Build symmetric target T and co-occurrence rate p ────────────
    # The original code assigned p[i,j] without symmetrising; cross-group
    # entries from maximum_num_links may only cover one direction.
    T = np.zeros((n_groups, n_groups))
    p = np.zeros((n_groups, n_groups))

    for (g1, g2), E in G.maximum_num_links.items():
        i, j  = g_idx[int(g1)], g_idx[int(g2)]
        T[i, j] = T[j, i] = float(E)                   # symmetrise T
        ni, nj  = int(N[i]), int(N[j])
        if i == j:
            d = ni * (ni - 1)
            if d > 0:
                p[i, i] = float(E) / d
        else:
            d = ni * nj
            if d > 0:
                p[i, j] = p[j, i] = float(E) / d       # symmetrise p

    # Store probability matrix for downstream consumers
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-5
    G.probability_matrix = T / row_sums

    # ── Determine K ──────────────────────────────────────────────────
    H_g   = 1.0 - np.diag(p) + p @ N          # expected community size per group
    H_avg = float(N @ H_g / total_nodes) if total_nodes > 0 else 2.0
    K     = (int(num_communities) if num_communities is not None
             else max(1, round(total_nodes / H_avg)))
    print(f"  Avg household size : {H_avg:.2f}  |  K = {K}  |  nodes = {total_nodes}")

    # Soft floor: keeps every community reachable; prevents permanent starvation
    floor = 1.0 / K

    # ── Phase 1 – Sequential conditional initialisation ──────────────
    # Place the largest group evenly, then each subsequent group is drawn
    # proportional to the affinity from groups already placed.  This seeds
    # the cross-group correlation structure that independent draws miss.
    C     = np.zeros((K, n_groups), dtype=np.int64)
    order = list(np.argsort(-N))                # descending group size

    for step, gi in enumerate(order):
        n_g = int(N[gi])
        if n_g == 0:
            continue
        if step == 0:
            w = np.full(K, floor)               # spread first group evenly
        else:
            # Attraction from groups already seated in each community
            placed = order[:step]
            w  = C[:, placed].astype(np.float64) @ p[gi, placed]  # (K,)
            w += floor
            w  = np.maximum(w, 0.0)
        w /= w.sum()
        C[:, gi] = rng.multinomial(n_g, w)

    A   = _actual_interactions(C, N)
    print(f"  After init         : MRE = {_mean_rel_error(A, T):.4f}")

    # ── Phase 2 – Error-corrected Gibbs sweeps ────────────────────────
    # Each sweep: compute a per-pair correction factor sqrt(T / A), then
    # resample each group's column using corrected affinity weights.
    # The sqrt dampens the correction to prevent overshoot/oscillation.
    prev_mre = np.inf
    for it in range(n_iter):

        # Correction matrix: > 1 where we need more interactions, < 1 where fewer.
        # T = 0  →  correction = 0  (zero target ⇒ repel the pairing)
        # A ≈ 0  →  correction = clip max  (missing interactions ⇒ attract strongly)
        A = _actual_interactions(C, N)
        correction = np.where(
            T > 0,
            np.clip(np.sqrt(T / np.maximum(A, 1e-9)), 0.1, 10.0),
            0.0,
        )

        for gi in rng.permutation(n_groups):
            n_g = int(N[gi])
            if n_g == 0:
                continue

            # Error-corrected affinity weights for group gi:
            #   w[k] = Σ_{j ≠ i}  p[i,j] · correction[i,j] · C[k,j]
            p_eff  = p[gi] * correction[gi]                    # (n_groups,)
            w      = C.astype(np.float64) @ p_eff              # (K,)
            w     -= p_eff[gi] * C[:, gi].astype(np.float64)   # remove stale self
            w     += floor
            w      = np.maximum(w, 0.0)
            w     /= w.sum()
            C[:, gi] = rng.multinomial(n_g, w)

        A   = _actual_interactions(C, N)
        mre = _mean_rel_error(A, T)
        print(f"  Gibbs {it + 1:2d}/{n_iter}        : MRE = {mre:.4f}")

        # Early stop when improvement stalls
        if prev_mre - mre < 5e-5:
            print(f"  Converged (ΔMRE < 5e-5).")
            break
        prev_mre = mre

    # ── Assign nodes to communities ───────────────────────────────────
    G.nodes_to_communities  = {}
    G.communities_to_nodes  = defaultdict(list)
    G.communities_to_groups = defaultdict(list)

    for gi, g in enumerate(group_ids):
        nodes = list(G.group_to_nodes.get(g, []))
        random.shuffle(nodes)
        ptr = 0
        for k in range(K):
            c = int(C[k, gi])
            if c == 0:
                continue
            batch = nodes[ptr : ptr + c]
            ptr  += c
            G.nodes_to_communities.update({node: k for node in batch})
            G.communities_to_nodes[(k, g)].extend(batch)
            G.communities_to_groups[k].append(g)

    G.number_of_communities = K

    sizes    = C.sum(axis=1)
    nonempty = sizes[sizes > 0]
    print(f"  Sizes : min={int(nonempty.min())}, "
          f"max={int(nonempty.max())}, mean={float(nonempty.mean()):.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Household algorithm (convergent) — validation")
    print("=" * 60)

    G = NetworkXGraph('_validate_hh')
    init_nodes(G, POPS_PATH, SCALE)
    print(f"Nodes: {G.graph.number_of_nodes()}  Groups: {len(G.group_ids)}")

    _compute_maximum_num_links(G, LINKS_PATH, SCALE, verbose=False)

    print("\n-- Running populate_communities_households --")
    populate_communities_households(
        G, num_communities=N_COMMUNITIES, n_iter=N_GIBBS_ITER, seed=SEED
    )

    print("\n-- Connecting communities --")
    connect_all_within_communities(G, verbose=True)
    print(f"Edges after full connection: {G.graph.number_of_edges()}")

    print("\n-- Checking group interactions --")
    results = check_group_interactions(G, print_report=True)

    targets = [r['target'] for r in results.values() if r['target'] > 0]
    actuals = [r['actual'] for r in results.values() if r['target'] > 0]
    ratios  = [r['ratio']  for r in results.values() if r['target'] > 0]

    total_target = sum(targets)
    total_actual = sum(actuals)
    print(f"\n  Total target edges : {total_target:,}")
    print(f"  Total actual edges : {total_actual:,}  "
          f"({100 * total_actual / total_target:.1f}%)")
    print(f"  Median ratio       : {float(np.median(ratios)):.3f}")
    print(f"  Pairs within 5%    : "
          f"{sum(1 for r in ratios if abs(r - 1) < 0.05)} / {len(ratios)}")

    print("\n-- Saving plots --")
    plot_group_interactions(results, G,
                            scatter_path=OUTPUT_SCATTER,
                            bar_path=OUTPUT_HEATMAP)

    import shutil
    if os.path.exists('_validate_hh'):
        shutil.rmtree('_validate_hh')


if __name__ == '__main__':
    main()