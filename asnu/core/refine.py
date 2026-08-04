"""Numba community refinement -- pure-Python replacement for the Rust
`refine_communities_move` kernel.

Same greedy local-search algorithm:
  * repeatedly pick a random node and a random target community,
  * compute the incremental change in the edge-budget loss
        loss = sum_{g,h} | achieved(g,h) - budget(g,h) |,
        achieved(g,h) = sum_communities count_g(c) * count_h(c)
    for only the group-pairs the move touches (delta update),
  * accept the move iff it lowers the loss.
"""

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # numba not installed
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        # no-op decorator so the module imports; the loop then runs as pure
        # Python (correct, just ~300x slower). Only hit if numba is missing.
        def wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return wrap


@njit
def _pair_delta(comp, c_old, c_new, g, h):
    """Change in achieved(g,*) from moving one group-g node c_old -> c_new,
    as seen through group h. comp is not mutated."""
    cog = comp[c_old, g]; coh = comp[c_old, h]
    cng = comp[c_new, g]; cnh = comp[c_new, h]
    cog_n = cog - 1
    cng_n = cng + 1
    coh_n = cog_n if h == g else coh
    cnh_n = cng_n if h == g else cnh
    return (cog_n * coh_n + cng_n * cnh_n) - (cog * coh + cng * cnh)


@njit
def _refine_loop(current, groups, budget, comp, achieved,
                 n_communities, n_groups, n_iterations, seed):
    """In-place greedy swap loop. Mutates current/comp/achieved. Returns loss."""
    np.random.seed(seed)
    n = current.shape[0]

    # initial loss over all group-pairs
    cur_loss = 0.0
    for g in range(n_groups):
        for h in range(n_groups):
            diff = achieved[g, h] - budget[g, h]
            cur_loss += diff if diff >= 0 else -diff

    for _ in range(n_iterations):
        i = np.random.randint(0, n)
        g = groups[i]
        c_old = current[i]
        c_new = np.random.randint(0, n_communities)
        if c_old == c_new:
            continue

        # delta_loss: only group g's counts change, affecting pairs (g,h)/(h,g)
        # for groups h present in c_old or c_new (plus g itself).
        delta_loss = 0.0
        for h in range(n_groups):
            if comp[c_old, h] == 0 and comp[c_new, h] == 0 and h != g:
                continue
            d = _pair_delta(comp, c_old, c_new, g, h)
            if d == 0:
                continue
            if g == h:
                av = achieved[g, g]; bv = budget[g, g]
                old_c = av - bv; old_c = old_c if old_c >= 0 else -old_c
                nw = av + d - bv; nw = nw if nw >= 0 else -nw
                delta_loss += nw - old_c
            else:
                a1 = achieved[g, h]; b1 = budget[g, h]
                o1 = a1 - b1; o1 = o1 if o1 >= 0 else -o1
                n1 = a1 + d - b1; n1 = n1 if n1 >= 0 else -n1
                a2 = achieved[h, g]; b2 = budget[h, g]
                o2 = a2 - b2; o2 = o2 if o2 >= 0 else -o2
                n2 = a2 + d - b2; n2 = n2 if n2 >= 0 else -n2
                delta_loss += (n1 - o1) + (n2 - o2)

        if delta_loss < 0.0:
            # apply achieved deltas (comp not yet mutated, so recompute d)
            for h in range(n_groups):
                if comp[c_old, h] == 0 and comp[c_new, h] == 0 and h != g:
                    continue
                d = _pair_delta(comp, c_old, c_new, g, h)
                if d == 0:
                    continue
                if g == h:
                    achieved[g, g] += d
                else:
                    achieved[g, h] += d
                    achieved[h, g] += d
            comp[c_old, g] -= 1
            comp[c_new, g] += 1
            current[i] = c_new
            cur_loss += delta_loss
    return cur_loss


def refine_communities_move(assignments, node_groups, budget, n_groups,
                            n_communities, n_iterations, loss_goal=0.0,
                            _unused_flag=1, seed=42):
    """Drop-in replacement for the Rust refine_communities_move.

    Parameters mirror the Rust call in community.py:
        assignments   : int64 array [N] of community id per node
        node_groups   : int64 array [N] of group id per node
        budget        : dict {(g,h): int} edge budget per group-pair
        n_groups      : number of demographic groups
        n_communities : number of communities
        n_iterations  : number of swap attempts
        loss_goal     : accepted for signature parity (early-stop not applied)
        _unused_flag  : accepted for signature parity (Rust's 8th positional arg)
        seed          : RNG seed

    Returns (new_assignments, final_loss): refined int64 array [N] and float loss.
    """
    current = np.ascontiguousarray(assignments, dtype=np.int64).copy()
    groups = np.ascontiguousarray(node_groups, dtype=np.int64)

    budget_mat = np.zeros((n_groups, n_groups), dtype=np.int64)
    for (g, h), v in budget.items():
        if 0 <= g < n_groups and 0 <= h < n_groups:
            budget_mat[g, h] = v

    # dense composition matrix comp[community, group] and achieved = comp^T comp
    comp = np.zeros((n_communities, n_groups), dtype=np.int64)
    for idx in range(current.shape[0]):
        comp[current[idx], groups[idx]] += 1
    achieved = (comp.T @ comp).astype(np.int64)

    final_loss = _refine_loop(
        current, groups, budget_mat, comp, achieved,
        int(n_communities), int(n_groups), int(n_iterations), int(seed))

    return current, float(final_loss)