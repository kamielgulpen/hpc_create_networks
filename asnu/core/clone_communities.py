"""
Community CLONING for ASNU -- an alternative to create_communities().

Where create_communities() derives a community partition from scratch (assign +
expensive Rust refinement), clone_communities() takes a partition already
computed at a SMALL scale and scales it up by making integer copies of each
community, reconciling the (non-multiplicative) per-group node counts. This
skips the refinement entirely at the large scale.

Why it's valid
--------------
The refinement loss is  sum_pairs |achieved(g,h) - budget(g,h)|  where
achieved(g,h) = sum_communities count_g(c)*count_h(c) is a within-community
pairing count over demographic GROUPS. Budgets scale ~linearly with `scale`
(they are n*scale). Cloning each community f times keeps per-copy group counts
fixed, so total achieved(g,h) becomes f*original -- matching f*budget. The loss
scales by f and loss-per-unit-budget (quality) is unchanged.

Non-multiplicative counts (the reconciliation step)
---------------------------------------------------
init_nodes() sizes each group via stratified_allocate: floor(scale*n) then a
remainder handed to the largest groups. floor(0.01*n)*10 != floor(0.10*n) in
general, and the remainder lands differently at each scale, so the true
large-scale per-group counts are NOT exactly f x the small-scale counts. Cloning
implies count_g*f per group; the true new node set wants a different number.
Per group we reconcile:
  * deficit (true > clone-implied): distribute leftover new nodes across that
    group's clone communities, proportional to how many of the group each holds.
  * surplus (true < clone-implied): drop the excess slots, largest holders first.
Result: per-group totals match a fresh init_nodes() at the new scale EXACTLY,
node ids are contiguous 0..N-1, and every node gets a community -- so generate()
and load_communities() consume the output unchanged.

Output schema is identical to what populate_communities() writes:
number_of_communities, nodes_to_communities, communities_to_nodes,
communities_to_groups, node_coordinates.

Public function
---------------
clone_communities(old_nodes_path, pops_path, scale_old, scale_new,
                  output_path='communities.json', ...) -> (output_path, loss)

The return shape mirrors create_communities() -- (path, loss) -- so callers can
swap one for the other. `loss` here is the CLONED loss estimate (~f x the small-
scale loss); it is not a freshly optimized value. It's returned for bookkeeping
parity, and clone_source=True is recorded so downstream code can tell.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from asnu.core.utils import desc_groups, stratified_allocate, read_file


# ---------------------------------------------------------------------------
# group lookup (stable across scales: group ids come from population.parquet
# sorted by n desc, same file at both scales)
# ---------------------------------------------------------------------------

def _norm_val(val):
    """Type-normalise a demographic value so parquet reads match pop-file values."""
    if isinstance(val, np.integer):
        val = int(val)
    if isinstance(val, np.floating):
        val = float(val)
    if isinstance(val, float):
        return int(val) if val.is_integer() else round(val, 10)
    if isinstance(val, int):
        return int(val)
    return str(val)


def _build_group_lookup(pops_path, pop_column='n'):
    """Return (attrs_to_group, characteristic_cols) matching init_nodes/desc_groups."""
    group_desc, characteristic_cols = desc_groups(pops_path, pop_column=pop_column)
    attrs_to_group = {}
    for gid, info in group_desc.items():
        key = tuple(sorted((str(c), _norm_val(info[c])) for c in characteristic_cols))
        attrs_to_group[key] = gid
    return attrs_to_group, characteristic_cols


def _old_community_signatures(old_nodes_path, attrs_to_group, characteristic_cols):
    """
    Recover {old_community_id: {group_id: count}} from the small-scale
    nodes.parquet by mapping each node's demographic columns back to a stable
    group id (never trusting the small-scale positional node ids).

    Returns (sig, unmatched_row_count).
    """
    df = read_file(old_nodes_path)

    if 'community_id' not in df.columns:
        raise ValueError(
            f"{old_nodes_path}: no 'community_id' column; cannot clone from it")

    missing = [c for c in characteristic_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{old_nodes_path}: missing demographic columns {missing}; "
            f"columns present: {list(df.columns)}")

    df = df[df['community_id'].notna()].copy()
    df['community_id'] = df['community_id'].astype(np.int64)

    lookup = pd.DataFrame(
        [{**{c: v for c, v in key}, '_gid': gid} for key, gid in attrs_to_group.items()])

    norm = df[characteristic_cols].copy()
    for c in characteristic_cols:
        norm[c] = norm[c].map(_norm_val)
    norm['community_id'] = df['community_id'].values

    merged = norm.merge(lookup, on=characteristic_cols, how='left')
    unmatched = int(merged['_gid'].isna().sum())
    merged = merged[merged['_gid'].notna()]
    merged['_gid'] = merged['_gid'].astype(np.int64)

    sig = defaultdict(lambda: defaultdict(int))
    for (comm, gid), cnt in merged.groupby(['community_id', '_gid']).size().items():
        sig[int(comm)][int(gid)] = int(cnt)
    return sig, unmatched


def _new_group_node_lists(pops_path, scale_new, pop_column='n'):
    """
    Reproduce init_nodes() numbering at the NEW scale.
    Returns (group_to_nodes, group_total, n_nodes_total).
    """
    group_desc, _ = desc_groups(pops_path, pop_column=pop_column)
    alloc = stratified_allocate(
        [(gid, group_desc[gid][pop_column]) for gid in group_desc], scale_new)

    group_to_nodes = {}
    node_id = 0
    for gid in group_desc:                 # SAME order as init_nodes()
        n = alloc[gid]
        group_to_nodes[gid] = list(range(node_id, node_id + n))
        node_id += n
    return group_to_nodes, {g: len(v) for g, v in group_to_nodes.items()}, node_id


# ---------------------------------------------------------------------------
# clone + reconcile
# ---------------------------------------------------------------------------

def _shrink_demand(demand, remove_n):
    for _ in range(remove_n):
        j = max(range(len(demand)), key=lambda i: demand[i][1])
        if demand[j][1] <= 0:
            break
        demand[j][1] -= 1


def _grow_demand(demand, add_n):
    sizes = np.array([d[1] for d in demand], dtype=np.float64)
    if sizes.sum() == 0:
        sizes = np.ones_like(sizes)
    weights = sizes / sizes.sum()
    exact = weights * add_n
    base = np.floor(exact).astype(int)
    for i in range(len(demand)):
        demand[i][1] += int(base[i])
    rem = add_n - int(base.sum())
    order = np.argsort(-(exact - base))
    for i in range(rem):
        demand[order[i % len(demand)]][1] += 1


def _clone_and_assign(sig, factor, group_to_nodes, group_total, seed=42):
    """
    Build node->community assignments by cloning each old community `factor`
    times and reconciling per-group counts to the true new node set.

    Returns (nodes_to_communities, n_communities, per_group_reconcile) where
    per_group_reconcile[g] = (clone_implied, true_new_total, delta).
    """
    rng = np.random.default_rng(seed)

    clone_specs = []          # (new_comm_id, {group: count})
    raw_id = 0
    for c in sorted(sig.keys()):
        for _k in range(factor):
            clone_specs.append((raw_id, dict(sig[c])))
            raw_id += 1
    n_communities = raw_id

    group_demand = defaultdict(list)      # group -> [[clone_idx, desired_count], ...]
    for idx, (_cid, gc) in enumerate(clone_specs):
        for g, cnt in gc.items():
            if cnt > 0:
                group_demand[g].append([idx, cnt])

    nodes_to_communities = {}
    per_group_reconcile = {}

    for g in sorted(set(group_total) | set(group_demand)):
        pool = list(group_to_nodes.get(g, []))
        rng.shuffle(pool)
        demand = group_demand.get(g, [])
        clone_implied = sum(d[1] for d in demand)
        true_total = len(pool)
        delta = true_total - clone_implied

        if delta < 0 and demand:
            _shrink_demand(demand, -delta)
        elif delta > 0:
            if demand:
                _grow_demand(demand, delta)
            else:
                # group absent from every clone community (too rare at small
                # scale): attach its nodes to the largest clone community so
                # they aren't lost.
                sizes = [sum(gc.values()) for _cid, gc in clone_specs]
                j = int(np.argmax(sizes)) if sizes else 0
                demand = [[j, delta]]
                group_demand[g] = demand

        cursor = 0
        for idx, cnt in demand:
            if cnt <= 0:
                continue
            new_comm_id = clone_specs[idx][0]
            for nid in pool[cursor:cursor + cnt]:
                nodes_to_communities[int(nid)] = int(new_comm_id)
            cursor += cnt
        if cursor < len(pool) and demand:       # rounding remainder
            new_comm_id = clone_specs[demand[-1][0]][0]
            for nid in pool[cursor:]:
                nodes_to_communities[int(nid)] = int(new_comm_id)

        per_group_reconcile[g] = (clone_implied, true_total, delta)

    return nodes_to_communities, n_communities, per_group_reconcile


# ---------------------------------------------------------------------------
# serialise (populate_communities()'s exact schema)
# ---------------------------------------------------------------------------

def _serialise(nodes_to_communities, n_communities, nodes_to_group, seed=42):
    communities_to_nodes = defaultdict(list)
    communities_to_groups = defaultdict(set)
    for node, comm in nodes_to_communities.items():
        g = nodes_to_group[node]
        communities_to_nodes[comm].append(int(node))
        communities_to_groups[comm].add(int(g))

    rng = np.random.default_rng(seed)
    K = max(n_communities, 1)
    coord_pos = rng.permutation(K)
    node_coordinates = {node: float(coord_pos[comm]) / K % 1.0
                        for node, comm in nodes_to_communities.items()}

    return {
        'number_of_communities': int(n_communities),
        'nodes_to_communities': {str(k): int(v) for k, v in nodes_to_communities.items()},
        'communities_to_nodes': {str(c): [int(n) for n in ns]
                                 for c, ns in communities_to_nodes.items()},
        'communities_to_groups': {str(c): [int(g) for g in sorted(gs)]
                                  for c, gs in communities_to_groups.items()},
        'node_coordinates': {str(k): float(v) for k, v in node_coordinates.items()},
    }


def _estimate_cloned_loss(sig, factor, budget):
    """
    Estimate the loss of the cloned assignment against a NEW-scale budget:
        loss = sum_pairs |achieved - budget|,  achieved = f * sum_c count_g*count_h.
    budget: {(g,h): int}. Returns float, or None if budget is None.
    """
    if budget is None:
        return None
    achieved = defaultdict(int)
    for _c, gc in sig.items():
        for g, cg in gc.items():
            for h, ch in gc.items():
                achieved[(g, h)] += cg * ch
    for k in achieved:
        achieved[k] *= factor

    loss = 0.0
    seen = set()
    for (g, h), av in achieved.items():
        bv = budget.get((g, h), 0)
        loss += abs(av - bv)
        seen.add((g, h))
    for (g, h), bv in budget.items():
        if (g, h) not in seen:
            loss += abs(bv)
    return float(loss)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def clone_communities(old_nodes_path, pops_path, scale_old, scale_new,
                      output_path='communities.json',
                      factor=None, pop_column='n', seed=42,
                      budget=None, verbose=True):
    """
    Clone a small-scale community partition up to a larger scale.

    Parameters
    ----------
    old_nodes_path : str
        nodes.parquet from the small-scale run (needs 'community_id' + the
        demographic characteristic columns).
    pops_path : str
        population.parquet (same file drives group ids at both scales).
    scale_old, scale_new : float
        The small and target scales, e.g. 0.01 and 0.10.
    output_path : str
        Where to write the communities JSON.
    factor : int, optional
        Clones per community. Default round(scale_new / scale_old).
    budget : dict, optional
        {(g,h): int} new-scale edge budget, for the cloned-loss estimate only.
    verbose : bool
        Print a per-group reconciliation summary.

    Returns
    -------
    (output_path, loss) : (str, float | None)
        Mirrors create_communities(). `loss` is the CLONED-loss estimate
        (None if no budget given).
    """
    f = factor if factor is not None else int(round(scale_new / scale_old))
    if f < 1:
        raise ValueError(f"clone factor must be >= 1 (got {f})")

    attrs_to_group, char_cols = _build_group_lookup(pops_path, pop_column)
    sig, unmatched = _old_community_signatures(old_nodes_path, attrs_to_group, char_cols)
    group_to_nodes, group_total, n_new = _new_group_node_lists(
        pops_path, scale_new, pop_column)

    nodes_to_communities, n_comm, recon = _clone_and_assign(
        sig, f, group_to_nodes, group_total, seed=seed)

    nodes_to_group = {}
    for g, nodelist in group_to_nodes.items():
        for nid in nodelist:
            nodes_to_group[int(nid)] = int(g)

    data = _serialise(nodes_to_communities, n_comm, nodes_to_group, seed=seed)
    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)

    loss = _estimate_cloned_loss(sig, f, budget)

    if verbose:
        total_deficit = sum(-d for *_x, d in recon.values() if d < 0)
        total_surplus = sum(d for *_x, d in recon.values() if d > 0)
        print(f"  [clone] factor={f}  old_communities={len(sig)} -> "
              f"new={n_comm}  unmatched_old_rows={unmatched}")
        print(f"  [clone] new nodes={n_new}  assigned={len(nodes_to_communities)}  "
              f"reconcile: +{total_surplus} added / {total_deficit} dropped")
        if loss is not None:
            print(f"  [clone] estimated cloned loss={loss:.6g}")

    return output_path, loss
