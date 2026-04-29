"""
Community detection and management module for ASNU.

This module provides functions for creating, populating, and analyzing
community structures within population networks. It implements agent-based
decision making for community assignment and provides various community
size distributions.

Functions
---------
build_group_pair_to_communities_lookup : Create lookup for group pairs to communities
populate_communities : Assign nodes to communities using group-aligned decision making
find_separated_groups : Identify groups with minimal inter-connections to seed communities
analyze_community_distribution : Analyze distribution of communities and groups
connect_all_within_communities : Create fully connected subgraphs within communities
fill_unfulfilled_group_pairs : Complete group pairs that didn't reach target edge count
export_community_edge_distribution : Export edge distribution to CSV file
export_community_node_distribution : Export node distribution to CSV file
"""
import csv
import random
from collections import Counter
from itertools import product
from tqdm import tqdm
from asnu_rust import process_nodes_capacity
from asnu_rust import process_nodes

import numpy as np


def build_group_pair_to_communities_lookup(G, verbose=False):
    """
    Create a lookup dictionary mapping each group pair to their shared communities.
    The algorithm will also resamble the structure when some groups are more representative
    in communities than others.

    This precomputes which communities contain which group pairs, making link
    creation much faster by avoiding repeated community membership checks.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community information
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Mapping from (src_id, dst_id) to list of shared community IDs
    """
    if verbose:
        print("Building community lookup for group pairs...")

    # Fast path: single community — every group pair maps to [0]; use a
    # defaultdict to avoid O(n_groups^2) pre-allocation.
    if G.number_of_communities == 1:
        from collections import defaultdict as _dd

        class _SingleCommLookup(_dd):
            """Lazy defaultdict that returns [0] for every missing key."""
            def __missing__(self, key):
                return [0]

        if verbose:
            print("  Single community: using lazy lookup (skipping O(n^2) build)")
        return _SingleCommLookup(list)

    # Build group → set-of-communities map.
    # A group can appear in multiple communities when the SA splits a large group
    # across many small communities (e.g. low new_comm_penalty on large populations).
    from collections import defaultdict as _dd
    group_to_comms = _dd(list)
    for community_id in range(G.number_of_communities):
        for g in G.communities_to_groups.get(community_id, []):
            group_to_comms[g].append(community_id)

    # Build pair → shared-communities only for pairs that have interactions.
    # Using max_links as the filter avoids the O(groups_per_community²) Cartesian product
    # that was infeasible for large group counts (70k groups → 487M pairs).
    group_pair_to_communities = {}
    for (src_id, dst_id) in G.maximum_num_links:
        sc = group_to_comms.get(src_id)
        dc = group_to_comms.get(dst_id)
        if not sc or not dc:
            continue
        shared = list(set(sc) & set(dc))
        if shared:
            group_pair_to_communities[(src_id, dst_id)] = shared

    if verbose:
        avg_communities = np.mean([len(v) for v in group_pair_to_communities.values()]) if group_pair_to_communities else 0
        print(f"  Found {len(group_pair_to_communities)} group pairs")
        print(f"  Average communities per pair: {avg_communities:.1f}")

    return group_pair_to_communities


def _process_nodes_python(G, all_nodes, node_groups, community_composition,
                          community_sizes, group_exposure, ideal,
                          target_counts, total_nodes):
    """Pure-Python fallback for probability-based node assignment."""
    for node_idx in range(len(all_nodes)):
        node = all_nodes[node_idx]
        group = node_groups[node_idx]

        current_exp = group_exposure[group, :]

        hypothetical_exp = current_exp + community_composition
        hypothetical_totals = hypothetical_exp.sum(axis=1, keepdims=True)
        hypothetical_totals = np.maximum(hypothetical_totals, 1e-10)
        hypothetical_dist = hypothetical_exp / hypothetical_totals

        diff = hypothetical_dist - ideal[group, :]
        distances = np.sqrt((diff ** 2).sum(axis=1))

        if target_counts is not None:
            full_mask = community_sizes >= target_counts
            distances[full_mask] = np.inf

        temperature = 1.0 - (node_idx / total_nodes)
        if temperature > 0.05:
            valid_mask = distances < np.inf
            if valid_mask.sum() > 1:
                d = distances[valid_mask]
                scaled = -d / (temperature + 1e-10)
                scaled = scaled - scaled.max()
                probs = np.exp(scaled)
                probs = probs / probs.sum()
                valid_indices = np.where(valid_mask)[0]
                best_community = np.random.choice(valid_indices, p=probs)
            else:
                best_community = np.argmin(distances)
        else:
            best_community = np.argmin(distances)

        G.communities_to_nodes[(best_community, group)].append(node)
        G.nodes_to_communities[node] = best_community
        G.communities_to_groups[best_community].append(group)

        group_exposure[group, :] += community_composition[best_community, :]
        mask = community_composition[best_community, :] > 0
        group_exposure[mask, group] += 1

        community_composition[best_community, group] += 1
        community_sizes[best_community] += 1

        if (node_idx + 1) % 500 == 0:
            print(f"Assigned {node_idx + 1}/{total_nodes} nodes ({100*(node_idx+1)/total_nodes:.1f}%)")


def populate_communities(G, num_communities, community_size_distribution='natural'):
    """
    Assign nodes to communities using probability-based group alignment.

    Each node chooses a community by minimising the distance between the
    community's hypothetical group-exposure distribution and the ideal
    probability distribution derived from link counts.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Number of communities to create
    community_size_distribution : str or array-like, optional
        Controls community size distribution ('natural', 'uniform',
        'powerlaw', or a custom array of fractions summing to 1)
    """
    total_nodes = len(list(G.graph.nodes))
    n_groups = int(len(G.group_ids))

    # Build probability matrix from link counts
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G.maximum_num_links.items():
        affinity[i, j] = count

    epsilon = 1e-5
    normalized = affinity / (affinity.sum(axis=1, keepdims=True) + epsilon)
    normalized[normalized == 0] = epsilon
    G.probability_matrix = normalized.copy()
    G.number_of_communities = num_communities

    # Target community sizes
    if isinstance(community_size_distribution, (list, np.ndarray)):
        target_sizes = np.array(community_size_distribution)
        if not np.isclose(target_sizes.sum(), 1.0):
            raise ValueError("Custom community_size_distribution must sum to 1")
    elif community_size_distribution == 'powerlaw':
        ranks = np.arange(1, num_communities + 1)
        sizes = 1.0 / ranks
        target_sizes = sizes / sizes.sum()
    elif community_size_distribution == 'uniform':
        target_sizes = np.ones(num_communities) / num_communities
    else:
        target_sizes = None

    # Initialise community structures
    for community_idx in range(num_communities):
        for group_id in range(n_groups):
            G.communities_to_nodes[(community_idx, group_id)] = []
        G.communities_to_groups[community_idx] = []

    community_composition = np.zeros((num_communities, n_groups), dtype=np.float64)
    community_sizes = np.zeros(num_communities, dtype=np.int32)
    group_exposure = np.zeros((n_groups, n_groups), dtype=np.float64)
    ideal = G.probability_matrix.copy()

    target_counts = None
    if target_sizes is not None:
        target_counts = (target_sizes * total_nodes).astype(np.int32)
        remainder = total_nodes - target_counts.sum()
        for i in range(remainder):
            target_counts[i % num_communities] += 1

    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    try:
        from asnu_rust import process_nodes_capacity, process_nodes_capacity_sparse
        print("Using Rust backend for node processing...")
        assignments = process_nodes_capacity(
            all_nodes.astype(np.int64), node_groups.astype(np.int64),
            community_composition, community_sizes,
            group_exposure, ideal,
            target_counts if target_sizes is not None else None,
            total_nodes,
        )
        for i in range(len(all_nodes)):
            node = int(all_nodes[i])
            comm = int(assignments[i])
            group = int(node_groups[i])
            G.communities_to_nodes[(comm, group)].append(node)
            G.nodes_to_communities[node] = comm
            G.communities_to_groups[comm].append(group)
    except ImportError:
        _process_nodes_python(
            G, all_nodes, node_groups, community_composition,
            community_sizes, group_exposure, ideal,
            target_counts,
            total_nodes,
        )

    print(f"\nCommunity population complete: {len(all_nodes)} nodes assigned to {num_communities} communities")


def _process_nodes_capacity_python(G, all_nodes, node_groups, num_communities,
                                   target, total_nodes, target_counts, new_comm_penalty,
                                   initial_comp=None, allow_new_communities=True):
    """Pure-Python fallback for capacity-based node assignment using matrix ops.

    Optimized: only evaluates non-empty communities, avoids matrix copies,
    pre-allocates with room to grow.

    Uses soft penalty for budget overshoot (instead of hard feasibility rejection)
    and always includes a new empty community as a candidate, so communities are
    only created when that genuinely minimises the remaining edge-budget distance.

    Parameters
    ----------
    initial_comp : list of dict, optional
        Pre-seeded compositions: initial_comp[i] = {group_id: count}.
        Communities 0..len(initial_comp)-1 are pre-populated; seed nodes must
        be excluded from all_nodes before calling this function.
    """
    OVERSHOOT_PENALTY = 10.0

    n_groups = target.shape[0]

    # Pre-allocate with extra room for dynamic growth
    capacity = num_communities + num_communities // 20
    comp = np.zeros((capacity, n_groups), dtype=np.float64)
    community_sizes = np.zeros(capacity, dtype=np.int64)
    num_active = num_communities  # number of communities actually in use

    # Initialise composition from pre-seeded communities (acc stays zero —
    # each seed is alone in its own community so no intra-community pairs yet)
    if initial_comp:
        for comm_id, comp_dict in enumerate(initial_comp):
            for group_id, count in comp_dict.items():
                comp[comm_id, int(group_id)] += count
                community_sizes[comm_id] += count

    # Sparse accumulated: only non-zero (g, h) entries stored
    from collections import defaultdict as _dd
    _acc = _dd(float)  # keys: (g, h) → count

    # Target rows/cols accessed on demand (avoid 2×n_groups² pre-allocation)
    _tgt_row_cache: dict = {}
    _tgt_col_cache: dict = {}

    def _get_tgt_row(g):
        if g not in _tgt_row_cache:
            _tgt_row_cache[g] = target[g, :].copy()
        return _tgt_row_cache[g]

    def _get_tgt_col(g):
        if g not in _tgt_col_cache:
            _tgt_col_cache[g] = target[:, g].copy()
        return _tgt_col_cache[g]

    def _get_acc_row(g):
        row = np.zeros(n_groups, dtype=np.float64)
        for (gi, hi), v in _acc.items():
            if gi == g:
                row[hi] = v
        return row

    def _get_acc_col(g):
        col = np.zeros(n_groups, dtype=np.float64)
        for (gi, hi), v in _acc.items():
            if hi == g:
                col[gi] = v
        return col

    random.shuffle(all_nodes)

    for node_idx in range(len(all_nodes)):
        node = all_nodes[node_idx]
        g = node_groups[node_idx]

        tgt_row = _get_tgt_row(g)
        tgt_col = _get_tgt_col(g)
        acc_row_g = _get_acc_row(g)
        acc_col_g = _get_acc_col(g)

        # Baseline distance: zero-count contribution for all groups h
        # rem_row[h] = tgt_row[h] - acc_row[g,h],  rem_col[h] = tgt_col[h] - acc_col[h,g]
        # base = sum_h(rem_row[h]^2) + sum_{h!=g}(rem_col[h]^2)
        rem_row_sq = (tgt_row - acc_row_g) ** 2      # (n_groups,)
        rem_col_sq = (tgt_col - acc_col_g) ** 2      # (n_groups,)
        base = float(rem_row_sq.sum() + rem_col_sq.sum() - rem_col_sq[g])

        # New empty community has zero composition → distance is exactly sqrt(base),
        # penalized to discourage creating new communities and encourage larger ones.
        # When allow_new_communities is False, make it impossible to pick a new community.
        if allow_new_communities:
            new_comm_dist = new_comm_penalty * (base ** 0.5)
        else:
            new_comm_dist = np.inf

        eval_count = num_active  # index of the "new community" candidate

        if num_active > 0:
            active = comp[:num_active]   # (num_active, n_groups) view, no copy

            # Fully vectorized — no Python loop over groups
            # Step 1: soft-penalty costs assuming hyp = acc + count (h != g treatment)
            rem_row = tgt_row - (acc_row_g + active)   # (num_active, n_groups)
            rem_col = tgt_col - (acc_col_g + active)   # (num_active, n_groups)
            eff_row = np.where(rem_row < 0, OVERSHOOT_PENALTY * rem_row ** 2, rem_row ** 2)
            eff_col = np.where(rem_col < 0, OVERSHOOT_PENALTY * rem_col ** 2, rem_col ** 2)

            # Step 2: where count == 0, revert to zero-count baseline (no penalty)
            has_active = active > 0   # (num_active, n_groups)
            eff_row = np.where(has_active, eff_row, rem_row_sq)
            eff_col = np.where(has_active, eff_col, rem_col_sq)

            # Step 3: sum all h, drop col term for h == g (no col term in Rust for self-group)
            dist_sq = eff_row.sum(axis=1) + eff_col.sum(axis=1) - eff_col[:, g]

            # Step 4: fix row term for h == g — Rust uses 2*count, not 1*count
            hyp_gg = acc_row_g[g] + 2.0 * active[:, g]
            rem_gg = tgt_row[g] - hyp_gg
            gg_cost = np.where(rem_gg < 0, OVERSHOOT_PENALTY * rem_gg ** 2, rem_gg ** 2)
            dist_sq += np.where(has_active[:, g], gg_cost - eff_row[:, g], 0.0)

            distances = np.sqrt(np.maximum(0.0, dist_sq))

            # Hard size limit (if specified) — still respected
            if target_counts is not None:
                tc_len = min(num_active, len(target_counts))
                distances[:tc_len][community_sizes[:tc_len] >= target_counts[:tc_len]] = np.inf

            all_distances = np.append(distances, new_comm_dist)
        else:
            all_distances = np.array([new_comm_dist])

        # Temperature-based SA selection (matches Rust schedule)
        temperature = 1.0 - (node_idx / total_nodes)
        if temperature > 0.05:
            valid_mask = np.isfinite(all_distances)
            n_valid = valid_mask.sum()
            if n_valid > 1:
                d = all_distances[valid_mask]
                scaled = -d / (temperature + 1e-10)
                scaled -= scaled.max()
                probs = np.exp(scaled)
                probs /= probs.sum()
                valid_indices = np.where(valid_mask)[0]
                choice = int(np.random.choice(valid_indices, p=probs))
            elif n_valid == 1:
                choice = int(np.where(valid_mask)[0][0])
            else:
                # All distances are inf: if new communities are allowed, open one;
                # otherwise fall back to the least-bad existing community.
                if allow_new_communities:
                    choice = eval_count
                else:
                    choice = int(np.argmin(distances)) if num_active > 0 else eval_count
        else:
            finite_mask = np.isfinite(all_distances)
            if finite_mask.any():
                choice = int(np.argmin(np.where(finite_mask, all_distances, np.inf)))
            else:
                # All distances are inf: same logic as above.
                if allow_new_communities:
                    choice = eval_count
                else:
                    choice = int(np.argmin(distances)) if num_active > 0 else eval_count

        # If the chosen index is beyond current active communities, create a new one
        if choice >= num_active:
            if num_active >= comp.shape[0]:
                extra = max(500, comp.shape[0] // 2)
                comp = np.vstack([comp, np.zeros((extra, n_groups), dtype=np.float64)])
                community_sizes = np.append(community_sizes, np.zeros(extra, dtype=np.int64))
            best_community = num_active
            num_active += 1
        else:
            best_community = choice

        # Update sparse accumulated (only non-zero groups in this community)
        bc_comp = comp[best_community]
        nz_groups = np.nonzero(bc_comp)[0]
        for h in nz_groups:
            count_h = float(bc_comp[h])
            if h != g:
                _acc[(g, h)] = _acc.get((g, h), 0.0) + count_h
                _acc[(h, g)] = _acc.get((h, g), 0.0) + count_h
            else:
                _acc[(g, g)] = _acc.get((g, g), 0.0) + 2 * count_h

        # Update community composition
        comp[best_community, g] += 1
        community_sizes[best_community] += 1

        # Assign node
        key = (best_community, g)
        if key not in G.communities_to_nodes:
            G.communities_to_nodes[key] = []
        G.communities_to_nodes[key].append(node)
        G.nodes_to_communities[node] = best_community
        if best_community not in G.communities_to_groups:
            G.communities_to_groups[best_community] = []
        G.communities_to_groups[best_community].append(g)

        if (node_idx + 1) % 500 == 0:
            print(f"Capacity assignment: {node_idx + 1}/{total_nodes} nodes "
                  f"({100*(node_idx+1)/total_nodes:.1f}%), {num_active} communities")

    G.number_of_communities = num_active


def find_separated_groups(G, num_communities):
    """
    Select groups with minimal mutual interaction for community seeding.

    Uses greedy farthest-point selection: starts with the group that has the
    lowest total interaction, then repeatedly picks the group with the least
    accumulated interaction toward already-selected groups.
    """
    import heapq
    from collections import defaultdict

    # Only consider groups that have at least one node
    groups_with_nodes = [g for g in G.group_ids if G.group_to_nodes.get(g)]
    if not groups_with_nodes:
        return []
    groups_set = set(groups_with_nodes)
    group_index = {g: i for i, g in enumerate(groups_with_nodes)}  # stable tiebreak

    # One pass over edges builds both group_totals AND per-group adjacency.
    # neighbors[B][g] = amount added to interaction_sum[g] when B is selected
    #                = maximum_num_links.get((g, B), 0) + maximum_num_links.get((B, g), 0)
    neighbors = defaultdict(dict)
    group_totals = defaultdict(int)
    for (a, b), cnt in G.maximum_num_links.items():
        a_in = a in groups_set
        b_in = b in groups_set
        if a_in:
            group_totals[a] += cnt
        if b_in:
            group_totals[b] += cnt
        if a_in and b_in:
            nb = neighbors[a]; nb[b] = nb.get(b, 0) + cnt
            nb = neighbors[b]; nb[a] = nb.get(a, 0) + cnt

    interaction_sum = dict.fromkeys(groups_with_nodes, 0)
    selected_groups = set()
    used_nodes = set()
    selected = []
    n_target = min(num_communities, len(groups_with_nodes))

    heap = None  # built lazily after the first seed is placed

    for _ in range(n_target):
        if not selected:
            # First seed: group with lowest overall interaction
            best_group = min(groups_with_nodes, key=lambda g: group_totals.get(g, 0))
        else:
            if heap is None:
                heap = [(interaction_sum[g], group_index[g], g)
                        for g in groups_with_nodes if g not in selected_groups]
                heapq.heapify(heap)
            best_group = None
            # Lazy deletion: skip stale (value no longer matches) or already-selected entries
            while heap:
                val, _, g = heapq.heappop(heap)
                if g in selected_groups:
                    continue
                if val != interaction_sum[g]:
                    continue
                best_group = g
                break
            if best_group is None:
                break

        # Pick a node from this group not already used as a seed
        candidates = [n for n in G.group_to_nodes[best_group] if n not in used_nodes]
        if not candidates:
            selected_groups.add(best_group)  # exhausted; skip in future iterations
            continue
        node = random.choice(candidates)
        selected.append((best_group, node))
        selected_groups.add(best_group)
        used_nodes.add(node)

        # Only walk the groups that actually share an edge with best_group
        nb = neighbors.get(best_group)
        if nb:
            for g, w in nb.items():
                if g in selected_groups:
                    continue
                interaction_sum[g] += w
                if heap is not None:
                    heapq.heappush(heap, (interaction_sum[g], group_index[g], g))

    # If more seeds are needed than unique groups, fill from least-interactive nodes.
    # Pre-sort once (O(n log n)) instead of rebuilding+sorting on every iteration.
    if len(selected) < num_communities:
        extra_candidates = sorted(
            [(g, n) for g in groups_with_nodes for n in G.group_to_nodes[g]],
            key=lambda gn: group_totals.get(gn[0], 0),
        )
        for best_group, node in extra_candidates:
            if len(selected) >= num_communities:
                break
            if node not in used_nodes:
                selected.append((best_group, node))
                used_nodes.add(node)

    return selected


def populate_communities_capacity(G, num_communities, community_size_distribution='natural', new_comm_penalty=3.0, allow_new_communities=True, fast=False, sa_fraction=1.0, overcap_penalty=float('inf'), max_eval=100):
    """
    Assign nodes to communities by matching absolute edge counts against
    maximum_num_links budget, with feasibility constraints ensuring
    communities can be fully connected without exceeding the budget.

    Uses same SA temperature schedule as populate_communities() but with
    capacity-based distance (absolute edge counts, not probabilities).

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Initial number of communities (may grow if needed)
    community_size_distribution : str or array-like, optional
        Controls community size distribution
    """
    total_nodes = len(list(G.graph.nodes))
    n_groups = int(len(G.group_ids))

    G.number_of_communities = num_communities

    # Target community sizes
    if isinstance(community_size_distribution, (list, np.ndarray)):
        target_sizes = np.array(community_size_distribution)
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'powerlaw':
        ranks = np.arange(1, num_communities + 1)
        sizes = 1.0 / (ranks ** 1)
        target_sizes = sizes / sizes.sum()
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'uniform':
        target_sizes = np.ones(num_communities) / num_communities
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'normal':
        mean_size = total_nodes / num_communities
        std_size = mean_size * 0.3
        raw = np.random.normal(mean_size, std_size, num_communities)
        raw = np.maximum(raw, 1.0)
        target_counts = np.maximum(
            np.round(raw * (total_nodes / raw.sum())).astype(np.int32), 1
        )
    else:
        target_counts = None

    # Fast path: single community — skip SA and all matrix allocations
    if num_communities == 1 and target_counts is None:
        all_nodes_fp = np.array(list(G.graph.nodes))
        for node in all_nodes_fp:
            group = G.nodes_to_group[node]
            G.communities_to_nodes.setdefault((0, group), []).append(int(node))
            G.nodes_to_communities[int(node)] = 0
            if 0 not in G.communities_to_groups:
                G.communities_to_groups[0] = []
            G.communities_to_groups[0].append(group)
        G.number_of_communities = 1
        G.probability_matrix = np.zeros((0, 0))  # empty placeholder; matrix not needed for 1 community
        print(f"\nCapacity-based community assignment complete: "
              f"{total_nodes} nodes -> 1 community (fast path)")
        return

    # Build target matrix from maximum_num_links (sparse construction)
    from scipy.sparse import csr_matrix as _csr_mat
    if G.maximum_num_links:
        _ri, _ci, _vi = zip(*[(i, j, v) for (i, j), v in G.maximum_num_links.items() if v > 0]) or ([], [], [])
    else:
        _ri, _ci, _vi = [], [], []
    target_sp = _csr_mat((list(_vi), (list(_ri), list(_ci))), shape=(n_groups, n_groups), dtype=np.float64) if _vi else _csr_mat((n_groups, n_groups), dtype=np.float64)

    epsilon = 1e-5
    row_sums = np.asarray(target_sp.sum(axis=1)).flatten() + epsilon
    target = np.asarray(target_sp.multiply(1.0 / row_sums[:, np.newaxis]).todense())
    G.probability_matrix = target

    # Build all_nodes in round-robin order across groups for pre-seeding.
    # This interleaves groups so the SA sees a balanced mix from the start.
    # Implementation: shuffle once, then sort by within-group position so that
    # node k of group g ends up in round k — O(n log n), no Python loops over groups.
    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    group_counts: dict = {}
    within_pos = np.empty(len(all_nodes), dtype=np.int64)
    for i, g in enumerate(node_groups):
        within_pos[i] = group_counts.get(g, 0)
        group_counts[g] = within_pos[i] + 1

    order = np.argsort(within_pos, kind='stable')
    all_nodes = all_nodes[order]
    node_groups = node_groups[order]

    # --- Pre-seed communities from least-interacting groups ---
    # find_separated_groups selects num_communities groups with minimal mutual
    # interaction; one node per group seeds each community before the SA starts.
    seeds = find_separated_groups(G, num_communities)
    seed_node_set = {node for _, node in seeds}

    initial_comp = []
    for comm_id, (group, node) in enumerate(seeds):
        G.communities_to_nodes.setdefault((comm_id, group), []).append(node)
        G.nodes_to_communities[node] = comm_id
        G.communities_to_groups.setdefault(comm_id, []).append(group)
        initial_comp.append({group: 1})

    # Remove seed nodes from the SA input
    mask = np.array([int(n) not in seed_node_set for n in all_nodes])
    sa_nodes = all_nodes[mask]
    sa_groups = node_groups[mask]
    sa_total_nodes = len(sa_nodes)  # SA backends must use this, not total_nodes
    print(f"  Pre-seeded {len(seeds)} communities from least-interacting groups; "
          f"{sa_total_nodes} nodes remaining for SA")

    # Community structures are populated on demand via setdefault throughout the SA loop;
    # pre-initialising every (community, group) pair is O(num_communities × n_groups)
    # and unnecessary — omitted for large-scale performance.

    # Try Rust backend, fall back to Python
    try:
        if fast == 'greedy':
            from asnu_rust import process_nodes_capacity_greedy as _rust_fn
            print("Using Rust greedy group-level assignment...")
        elif fast:
            from asnu_rust import process_nodes_capacity_fast as _rust_fn
            print("Using Rust fast assignment (no SA)...")
        else:
            from asnu_rust import process_nodes_capacity as _rust_fn
            print("Using Rust backend for capacity-based node processing...")

        budget = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}
        rust_initial_comp = {comm_id: {int(g): int(c) for g, c in d.items()}
                             for comm_id, d in enumerate(initial_comp)}

        effective_penalty = float('inf') if not allow_new_communities else new_comm_penalty
        assignments = _rust_fn(
            sa_nodes.astype(np.int64),
            sa_groups.astype(np.int64),
            budget,
            n_groups,
            num_communities,
            target_counts if target_counts is not None else None,
            sa_total_nodes,
            effective_penalty,
            rust_initial_comp if initial_comp else None,
            # sa_fraction,
            # overcap_penalty,
            # max_eval,
        )

        # Populate G structures from assignments
        for i in range(len(sa_nodes)):
            node = int(sa_nodes[i])
            comm = int(assignments[i])
            group = int(sa_groups[i])
            G.communities_to_nodes.setdefault((comm, group), []).append(node)
            G.nodes_to_communities[node] = comm
            G.communities_to_groups.setdefault(comm, []).append(group)

        all_assigned_comms = list(assignments) + list(range(len(seeds)))
        G.number_of_communities = max(all_assigned_comms) + 1 if all_assigned_comms else 0
    except ImportError:
        print("Using Python fallback for capacity-based node processing...")
        _process_nodes_capacity_python(
            G, sa_nodes, sa_groups, num_communities,
            target, sa_total_nodes, target_counts, new_comm_penalty,
            initial_comp=initial_comp if initial_comp else None,
            allow_new_communities=allow_new_communities,
        )

    print(f"\nCapacity-based community assignment complete: "
          f"{total_nodes} nodes -> {G.number_of_communities} communities")

    # Assign random uniform coordinates so Phase B spatial ring search activates.
    K = G.number_of_communities
    _rng = np.random.default_rng(42)
    comm_coords = _rng.uniform(0.0, 1.0, size=K)
    jitter_std = 1e-4
    node_coordinates = {}
    for node_int, comm in G.nodes_to_communities.items():
        theta_c = float(comm_coords[int(comm)])
        node_coordinates[int(node_int)] = (theta_c + float(_rng.normal(0, jitter_std))) % 1.0
    G.node_coordinates = node_coordinates


def connect_all_within_communities(G, verbose=True):
    """
    Connect all nodes within each community to each other.

    Creates a fully connected graph within each community using vectorized
    operations for maximum efficiency.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community assignments
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Statistics about edges created
    """
    verbose = False
    if verbose:
        print("\nConnecting all nodes within communities...")

    stats = {
        'total_edges': 0,
        'edges_per_community': {}
    }

    # OPTIMIZED: Build community membership lookup once
    communities_nodes = [[] for _ in range(G.number_of_communities)]
    for node, comm in G.nodes_to_communities.items():
        communities_nodes[comm].append(node)

    # For each community, connect all nodes within it
    for community_id in range(G.number_of_communities):
        community_nodes = communities_nodes[community_id]

        if len(community_nodes) == 0:
            continue

        edges_to_add = [(src, dst) for src, dst in product(community_nodes, repeat=2)
                       if src != dst]

        # Batch add edges (much faster than individual add_edge calls)
        G.graph.add_edges_from(edges_to_add)

        edges_added = len(edges_to_add)
        stats['edges_per_community'][community_id] = edges_added
        stats['total_edges'] += edges_added

        # Progress reporting for large numbers of communities
        if (community_id + 1) % 5000 == 0 or community_id == 0:
            print(f"  Connected {community_id + 1}/{G.number_of_communities} communities ({(community_id + 1) / G.number_of_communities * 100:.1f}%)")

        if verbose:
            print(f"  Community {community_id}: {len(community_nodes)} nodes, {edges_added} edges")

    if verbose:
        print(f"  Total edges created: {stats['total_edges']}")

    return stats


def fill_unfulfilled_group_pairs(G, reciprocity_p, verbose=True):
    """
    Complete any group pairs that didn't reach their target edge count.

    Randomly creates edges between nodes from unfulfilled group pairs until
    targets are met or maximum attempts are reached.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with existing edges
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Statistics about the filling process
    """
    if verbose:
        print("\nFilling unfulfilled group pairs...")

    unfulfilled_pairs = []
    stats = {
        'total_pairs': 0,
        'fulfilled_pairs': 0,
        'unfulfilled_pairs': 0,
        'edges_added': 0,
        'reciprocal_edges_added': 0
    }

    # Identify which group pairs need more edges
    for (src_id, dst_id) in G.maximum_num_links.keys():
        existing = G.existing_num_links.get((src_id, dst_id), 0)
        maximum = G.maximum_num_links[(src_id, dst_id)]

        stats['total_pairs'] += 1

        if maximum == 0:
            continue

        # Only try to fill pairs that are genuinely under the target
        if existing < maximum:
            unfulfilled_pairs.append((src_id, dst_id, existing, maximum))
            stats['unfulfilled_pairs'] += 1
        else:
            stats['fulfilled_pairs'] += 1

    if verbose:
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Fulfilled: {stats['fulfilled_pairs']}")
        print(f"  Unfulfilled: {stats['unfulfilled_pairs']}")

    def _fill_from_pool(src_pool, dst_pool, src_id, dst_id, needed):
        """Add up to `needed` edges by batch-sampling from src_pool/dst_pool."""
        if not src_pool or not dst_pool or needed <= 0:
            return
        src_arr = np.array(src_pool)
        dst_arr = np.array(dst_pool)
        batch = max(needed * 4, 512)   # oversample to tolerate rejections

        for _ in range(10):            # at most 10 numpy draws per pool
            if G.existing_num_links[(src_id, dst_id)] >= maximum:
                return
            srcs = np.random.choice(src_arr, size=batch).tolist()
            dsts = np.random.choice(dst_arr, size=batch).tolist()
            added_this_round = 0
            for s, d in zip(srcs, dsts):
                if G.existing_num_links[(src_id, dst_id)] >= maximum:
                    return
                if s == d or G.graph.has_edge(s, d):
                    continue
                G.graph.add_edge(s, d)
                G.existing_num_links[(src_id, dst_id)] += 1
                stats['edges_added'] += 1
                added_this_round += 1
                if reciprocity_p > 0 and random.random() < reciprocity_p:
                    if (G.existing_num_links.get((dst_id, src_id), 0) < G.maximum_num_links.get((dst_id, src_id), 0)
                            and not G.graph.has_edge(d, s)):
                        G.graph.add_edge(d, s)
                        G.existing_num_links[(dst_id, src_id)] += 1
                        stats['reciprocal_edges_added'] += 1
                        if dst_id == src_id:
                            stats['edges_added'] += 1
            if added_this_round == 0:
                return  # pool is saturated, stop early

    # Add edges to complete unfulfilled pairs
    total_needed = sum(max - ex for _, _, ex, max in unfulfilled_pairs)
    if unfulfilled_pairs:
        pbar = (tqdm(total=total_needed, unit='edge', desc='Filling pairs', dynamic_ncols=True)
                if tqdm and verbose else None)
        edges_before = stats['edges_added']

        for src_id, dst_id, existing, maximum in unfulfilled_pairs:
            existing = G.existing_num_links.get((src_id, dst_id), 0)
            maximum = G.maximum_num_links.get((src_id, dst_id), 0)
            needed = maximum - existing
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])

            if not src_nodes or not dst_nodes:
                continue

            edges_before_pair = stats['edges_added']

            # --- Phase 1: intra-community pool ---
            src_comm = {}
            for node in src_nodes:
                comm = G.nodes_to_communities.get(node)
                if comm is not None:
                    src_comm.setdefault(comm, []).append(node)
            dst_comm = {}
            for node in dst_nodes:
                comm = G.nodes_to_communities.get(node)
                if comm is not None:
                    dst_comm.setdefault(comm, []).append(node)

            shared = list(set(src_comm) & set(dst_comm))
            if shared:
                # Build numpy arrays per community for fast sampling
                src_arrs = [np.array(src_comm[c]) for c in shared]
                dst_arrs = [np.array(dst_comm[c]) for c in shared]
                # Weight community selection by pool size product
                weights = np.array([len(s) * len(d) for s, d in zip(src_arrs, dst_arrs)], dtype=float)
                weights /= weights.sum()
                n_comm = len(shared)
                batch = max((maximum - G.existing_num_links[(src_id, dst_id)]) * 4, 512)

                # Pre-build flat arrays + offsets so we can vectorize node sampling
                src_lens    = np.array([len(a) for a in src_arrs], dtype=np.int64)
                dst_lens    = np.array([len(a) for a in dst_arrs], dtype=np.int64)
                src_offsets = np.zeros(n_comm + 1, dtype=np.int64)
                dst_offsets = np.zeros(n_comm + 1, dtype=np.int64)
                src_offsets[1:] = src_lens.cumsum()
                dst_offsets[1:] = dst_lens.cumsum()
                src_flat = np.concatenate(src_arrs)
                dst_flat = np.concatenate(dst_arrs)

                for _ in range(10):
                    if G.existing_num_links[(src_id, dst_id)] >= maximum:
                        break
                    comm_indices = np.random.choice(n_comm, size=batch, p=weights)
                    # Vectorized: sample one node per candidate pair instead of per-iteration randint
                    s_batch = src_flat[src_offsets[comm_indices] +
                                       (np.random.random(batch) * src_lens[comm_indices]).astype(np.int64)]
                    d_batch = dst_flat[dst_offsets[comm_indices] +
                                       (np.random.random(batch) * dst_lens[comm_indices]).astype(np.int64)]
                    added_this_round = 0
                    for idx in range(batch):
                        if G.existing_num_links[(src_id, dst_id)] >= maximum:
                            break
                        s, d = int(s_batch[idx]), int(d_batch[idx])
                        if s == d or G.graph.has_edge(s, d):
                            continue
                        G.graph.add_edge(s, d)
                        G.existing_num_links[(src_id, dst_id)] += 1
                        stats['edges_added'] += 1
                        added_this_round += 1
                        if reciprocity_p > 0 and random.random() < reciprocity_p:
                            if (G.existing_num_links.get((dst_id, src_id), 0) < G.maximum_num_links.get((dst_id, src_id), 0)
                                    and not G.graph.has_edge(d, s)):
                                G.graph.add_edge(d, s)
                                G.existing_num_links[(dst_id, src_id)] += 1
                                stats['reciprocal_edges_added'] += 1
                                if dst_id == src_id:
                                    stats['edges_added'] += 1
                    if added_this_round == 0:
                        break  # pools saturated

            # # --- Phase 2: cross-community fallback ---
            _fill_from_pool(src_nodes, dst_nodes, src_id, dst_id,
                            needed=maximum - G.existing_num_links[(src_id, dst_id)])

            if pbar is not None:
                pbar.update(stats['edges_added'] - edges_before_pair)

        if pbar is not None:
            pbar.close()

    if verbose:
        print(f"  Edges added: {stats['edges_added']}")
        print(f"  Reciprocal edges added: {stats['reciprocal_edges_added']}")

    return stats


def _cluster_groups_louvain(budget, n_groups, target_k=None, resolution=1.0, seed=42):
    """
    Cluster groups into communities using Louvain on the budget interaction graph.

    Returns group_to_cluster (dict: group_id -> cluster_id) and actual_k (number of clusters).
    If target_k is given, binary-searches resolution to approximate that many clusters.
    Isolated groups (no interactions) are distributed round-robin across existing clusters.
    """
    import networkx as nx

    G_bud = nx.Graph()
    G_bud.add_nodes_from(range(n_groups))
    for (src, dst), w in budget.items():
        if w <= 0:
            continue
        s, d = int(src), int(dst)
        if 0 <= s < n_groups and 0 <= d < n_groups:
            if G_bud.has_edge(s, d):
                G_bud[s][d]['weight'] += float(w)
            else:
                G_bud.add_edge(s, d, weight=float(w))

    connected = [n for n in range(n_groups) if G_bud.degree(n) > 0]
    isolated  = [n for n in range(n_groups) if G_bud.degree(n) == 0]

    if not connected:
        k = max(1, target_k or 1)
        return {g: i % k for i, g in enumerate(range(n_groups))}, k

    try:
        from networkx.algorithms.community import louvain_communities as _louvain
    except ImportError:
        raise ImportError("networkx >= 3.0 is required for mode='clustered'")

    subgraph = G_bud.subgraph(connected)
    max_possible = len(connected)

    # When more communities are requested than there are groups, clustering can't produce
    # group-specialized communities — fall back to a single cluster so nodes are mixed
    # proportionally across all sub-communities (e.g. 2 genders → every community gets both).
    if target_k is not None and target_k >= max_possible:
        group_to_cluster = {g: 0 for g in range(n_groups)}
        return group_to_cluster, 1

    if target_k is not None and target_k > 1 and target_k < max_possible:
        # Binary-search resolution only when target is achievable (fewer clusters than groups).
        # When target_k >= n_groups, fall through to natural resolution so connected groups
        # stay merged (e.g. 2 genders with cross-group interactions → 1 cluster → mixed communities).
        lo, hi = 1e-6, 500.0
        best_comm, best_diff = None, float('inf')
        for _ in range(30):
            mid = (lo + hi) / 2.0
            comms = _louvain(subgraph, weight='weight', resolution=mid, seed=seed)
            diff = abs(len(comms) - target_k)
            if diff < best_diff:
                best_diff = diff
                best_comm = comms
            if best_diff == 0:
                break
            if len(comms) < target_k:
                lo = mid
            else:
                hi = mid
    else:
        best_comm = _louvain(subgraph, weight='weight', resolution=resolution, seed=seed)

    group_to_cluster = {}
    for cid, cluster in enumerate(best_comm):
        for g in cluster:
            group_to_cluster[g] = cid

    actual_k = len(best_comm)
    for i, g in enumerate(isolated):
        group_to_cluster[g] = i % actual_k

    return group_to_cluster, actual_k


def populate_communities_clustered(G, num_communities, community_size_distribution='uniform', resolution=1.0):
    """
    Assign nodes to communities via two phases:

    1. Louvain-cluster the groups on the budget graph, aiming for ~num_communities clusters.
       With few groups (e.g. 2) this produces few clusters; with many groups it produces ~num_communities.

    2. Split each cluster proportionally into sub-communities so the total reaches num_communities.
       Nodes of each group are distributed round-robin across their home cluster's sub-communities.

    This generalises across dataset sizes: with 2 groups every community gets a proportional
    mix of both; with thousands of groups each community gets a specialised subset.
    """
    from scipy.sparse import csr_matrix as _csr

    total_nodes = G.graph.number_of_nodes()
    n_groups = len(G.group_ids)

    # Probability matrix (kept for pipeline compatibility)
    if G.maximum_num_links:
        _ri, _ci, _vi = zip(*[(i, j, v) for (i, j), v in G.maximum_num_links.items() if v > 0]) or ([], [], [])
    else:
        _ri, _ci, _vi = [], [], []
    if _vi:
        _sp = _csr((_vi, (_ri, _ci)), shape=(n_groups, n_groups), dtype=np.float64)
    else:
        _sp = _csr((n_groups, n_groups), dtype=np.float64)
    _eps = 1e-5
    _rs = np.asarray(_sp.sum(axis=1)).flatten() + _eps
    G.probability_matrix = np.asarray(_sp.multiply(1.0 / _rs[:, np.newaxis]).todense())

    # ── Phase 1: cluster groups ───────────────────────────────────────────────
    print(f"  Clustering {n_groups} groups on budget graph (target {num_communities} communities)...")
    group_to_cluster, n_clusters = _cluster_groups_louvain(
        G.maximum_num_links, n_groups, target_k=num_communities, resolution=resolution,
    )
    print(f"  Louvain produced {n_clusters} group clusters")

    # ── Phase 2: allocate sub-communities per cluster ─────────────────────────
    # Weighted by the number of nodes in each cluster so larger clusters get more sub-comms.
    all_nodes = list(G.graph.nodes)
    node_groups_arr = [G.nodes_to_group[n] for n in all_nodes]

    cluster_mass = np.zeros(n_clusters, dtype=np.int64)
    for g_id in node_groups_arr:
        cluster_mass[group_to_cluster.get(int(g_id), 0)] += 1

    sub_per_cluster = np.ones(n_clusters, dtype=np.int64)
    if n_clusters < num_communities:
        total_mass = cluster_mass.sum()
        cumulative, prev = 0.0, 0
        for c in range(n_clusters):
            cumulative += num_communities * cluster_mass[c] / max(total_mass, 1)
            sub_per_cluster[c] = max(1, int(cumulative) - prev)
            prev = int(cumulative)
        # Fix rounding gap / overshoot
        diff = num_communities - int(sub_per_cluster.sum())
        order = np.argsort(-cluster_mass)
        for i in range(abs(diff)):
            idx = order[i % n_clusters]
            if diff > 0:
                sub_per_cluster[idx] += 1
            elif sub_per_cluster[idx] > 1:
                sub_per_cluster[idx] -= 1

    cluster_start = np.zeros(n_clusters + 1, dtype=np.int64)
    for c in range(n_clusters):
        cluster_start[c + 1] = cluster_start[c] + sub_per_cluster[c]
    actual_k = int(cluster_start[n_clusters])
    G.number_of_communities = actual_k
    print(f"  Split into {actual_k} sub-communities across {n_clusters} clusters")

    # ── Phase 3: build per-group assignment queues ────────────────────────────
    # Each group's nodes are distributed round-robin across its home sub-communities.
    rng = np.random.default_rng(seed=42)
    group_counts = Counter(int(g) for g in node_groups_arr)

    group_queues: dict = {}
    for g, count in group_counts.items():
        c         = group_to_cluster.get(g, 0)
        n_sub     = int(sub_per_cluster[c])
        base      = int(cluster_start[c])
        base_each = count // n_sub
        remainder = count % n_sub
        queue: list = []
        for i in range(n_sub):
            queue.extend([base + i] * (base_each + (1 if i < remainder else 0)))
        arr = np.array(queue, dtype=np.int64)
        rng.shuffle(arr)
        group_queues[g] = arr

    # ── Phase 4: assign nodes ─────────────────────────────────────────────────
    group_cursors: dict = {g: 0 for g in group_counts}
    for i, node in enumerate(all_nodes):
        g    = int(node_groups_arr[i])
        cur  = group_cursors[g]
        q    = group_queues.get(g)
        comm = int(q[cur]) if q is not None and cur < len(q) else group_to_cluster.get(g, 0)
        group_cursors[g] += 1
        G.communities_to_nodes.setdefault((comm, g), []).append(node)
        G.nodes_to_communities[node] = comm
        G.communities_to_groups.setdefault(comm, []).append(g)

    print(f"\nClustered assignment complete: {total_nodes} nodes -> {actual_k} communities")


def populate_communities_adaptive(G, num_communities, seed=42):
    """
    K_g-spread community assignment (capacity_fast style with group-specific spread).

    For each group g: K_g = min(K, floor(n_g² / X_gg)) is the clique-capacity spread
    factor — the number of communities group g participates in. Homophilic groups
    (large X_gg relative to n_g²) concentrate in fewer communities so each community
    can hold enough intra-group edges. Heterophilic groups (X_gg=0) spread across
    all K communities.

    K_g randomly selected communities are chosen per group; nodes are distributed
    evenly across them (same uniform split as capacity_fast). Communities end up
    mixed because different groups cover overlapping subsets of the K communities.
    """
    from scipy.sparse import csr_matrix as _csr
    rng = np.random.default_rng(seed)
    n_groups = len(G.group_ids)
    total_nodes = G.graph.number_of_nodes()

    # Probability matrix
    if G.maximum_num_links:
        _ri, _ci, _vi = zip(*[(i, j, v) for (i, j), v in G.maximum_num_links.items() if v > 0]) or ([], [], [])
    else:
        _ri, _ci, _vi = [], [], []
    if _vi:
        _sp = _csr((_vi, (_ri, _ci)), shape=(n_groups, n_groups), dtype=np.float64)
    else:
        _sp = _csr((n_groups, n_groups), dtype=np.float64)
    _eps = 1e-5
    _rs = np.asarray(_sp.sum(axis=1)).flatten() + _eps
    G.probability_matrix = np.asarray(_sp.multiply(1.0 / _rs[:, np.newaxis]).todense())

    # ── Step 1: K_g per group (uncapped) ─────────────────────────────────────
    sorted_groups = sorted(int(g) for g in G.group_ids)
    group_nodes_map = {g: list(G.group_to_nodes.get(g, [])) for g in sorted_groups}

    K_g: dict = {}
    for g in sorted_groups:
        n_g = len(group_nodes_map[g])
        x_gg = int(G.maximum_num_links.get((g, g), 0))
        if n_g == 0:
            K_g[g] = 1
        elif x_gg == 0:
            K_g[g] = None   # resolved to pool_size after we know it
        else:
            K_g[g] = min(num_communities, max(1, n_g * n_g // x_gg))

    # Pool size = max K_g among groups that have intra-group budget
    # num_communities acts as a floor so we always have at least that many communities
    capped_vals = [v for v in K_g.values() if v is not None]
    pool_size = max(max(capped_vals) if capped_vals else 1, num_communities)

    # Heterophilic groups spread across the full pool
    for g in sorted_groups:
        if K_g[g] is None:
            K_g[g] = pool_size

    G.number_of_communities = pool_size

    vals = list(K_g.values())
    print(f"  K_g: min={min(vals)}, max={max(vals)}, mean={np.mean(vals):.1f}, "
          f"pool_size={pool_size}")

    # ── Step 2: budget neighbour graph (sparse) ──────────────────────────────
    from collections import defaultdict as _dd
    budget_neighbors: dict = _dd(set)
    for (g, h), v in G.maximum_num_links.items():
        if v > 0:
            budget_neighbors[int(g)].add(int(h))
            budget_neighbors[int(h)].add(int(g))

    # ── Step 3: processing order — hubs first ────────────────────────────────
    # Groups with the largest total outgoing budget anchor communities first so
    # their budget-neighbours can co-locate with them in the next step.
    total_budget = {
        g: sum(int(G.maximum_num_links.get((g, h), 0)) for h in budget_neighbors[g])
        for g in sorted_groups
    }
    processing_order = sorted(sorted_groups,
                               key=lambda g: total_budget.get(g, 0),
                               reverse=True)

    # ── Step 4: community coordinates on [0,1) ───────────────────────────────
    comm_coords = rng.uniform(0.0, 1.0, size=pool_size)
    jitter_std = 1e-4
    node_coordinates: dict = {}

    # ── Step 5: triangle-aware community selection + node assignment ──────────
    # For each group (hubs first), score communities by how many of g's budget
    # neighbours are already assigned there.  Take the top-scored communities
    # first (up to K_g), then fill remaining slots randomly.  This maximises
    # the chance that groups A, B, C forming a triangle in the budget graph all
    # share at least one community, enabling Phase A to close that triangle.
    group_community_sets: dict = {}  # g -> list[int] of chosen community indices

    all_comm_indices = np.arange(pool_size, dtype=np.int64)

    for g in processing_order:
        nodes = group_nodes_map[g]
        if not nodes:
            group_community_sets[g] = []
            continue

        k = K_g[g]
        nodes_arr = np.array(nodes, dtype=np.int64)
        rng.shuffle(nodes_arr)

        # For each budget-neighbour already assigned, share exactly one of their
        # communities (chosen randomly from that neighbour's set).  This gives
        # one triangle-capable co-location per pair without collapsing all groups
        # into the same handful of communities.
        chosen_set: set = set()
        for h in budget_neighbors[g]:
            h_comms = group_community_sets.get(h)
            if h_comms and len(chosen_set) < k:
                anchor = int(rng.choice(h_comms))
                chosen_set.add(anchor)

        chosen: list = list(chosen_set)

        # Fill remaining slots from the rest of the pool at random
        remaining = k - len(chosen)
        if remaining > 0:
            mask = np.ones(pool_size, dtype=bool)
            if chosen_set:
                mask[np.array(list(chosen_set), dtype=np.int64)] = False
            available = all_comm_indices[mask]
            if len(available) >= remaining:
                extra = rng.choice(available, size=remaining, replace=False)
            else:
                extra = available
            chosen.extend(int(c) for c in extra)

        group_community_sets[g] = chosen
        k_actual = len(chosen)
        if k_actual == 0:
            continue

        # Distribute nodes evenly across chosen communities (capacity_fast style)
        base = len(nodes_arr) // k_actual
        remainder_n = len(nodes_arr) % k_actual
        idx = 0
        for i, comm_int in enumerate(chosen):
            count = base + (1 if i < remainder_n else 0)
            theta_c = float(comm_coords[comm_int])
            for node in nodes_arr[idx: idx + count]:
                node_int = int(node)
                G.communities_to_nodes.setdefault((comm_int, g), []).append(node_int)
                G.nodes_to_communities[node_int] = comm_int
                G.communities_to_groups.setdefault(comm_int, []).append(g)
                node_coordinates[node_int] = (theta_c + float(rng.normal(0, jitter_std))) % 1.0
            idx += count

    G.node_coordinates = node_coordinates
    print(f"\nAdaptive assignment complete: {total_nodes} nodes -> {pool_size} communities")

def populate_communities_segregation(G, num_communities, mixing_floor=0.1, isolation_threshold=0.05, seed=42):
    """
    Segregation-driven hierarchical community assignment.

    Measures isolation per characteristic from the edge budget, then partitions
    communities hierarchically: the most segregated characteristic anchors the
    primary split, the second most segregated subdivides within that. Each group
    maps exclusively to its block-specific communities — no shared mixing pool.
    Cross-block edges are handled by fill_unfulfilled / Phase B in edge creation.
    """
    rng = np.random.default_rng(seed)
    sorted_groups = sorted(int(g) for g in G.group_ids)
    group_nodes_map = {g: list(G.group_to_nodes.get(g, [])) for g in sorted_groups}
    N = G.graph.number_of_nodes()

    # Collect characteristic names
    all_chars = set()
    for gid in sorted_groups:
        all_chars.update(G.group_to_attrs.get(gid, {}).keys())
    all_chars = sorted(all_chars)

    # Compute isolation index per characteristic
    char_isolation = {}
    for char in all_chars:
        val_to_groups = {}
        for gid in sorted_groups:
            v = G.group_to_attrs.get(gid, {}).get(char)
            if v is not None:
                val_to_groups.setdefault(v, []).append(gid)

        if len(val_to_groups) <= 1:
            char_isolation[char] = 0.0
            continue

        isolations = []
        for v, vgroups in val_to_groups.items():
            vset = set(vgroups)
            n_v = sum(len(group_nodes_map[g]) for g in vset)
            if n_v == 0 or N == 0:
                continue
            intra = sum(G.maximum_num_links.get((g, h), 0) for g in vset for h in vset)
            total_out = sum(G.maximum_num_links.get((g, h), 0) for g in vset for h in sorted_groups)
            I_v = (intra / total_out - n_v / N) if total_out > 0 else 0.0
            isolations.append(I_v)
        char_isolation[char] = float(np.mean(isolations)) if isolations else 0.0

    meaningful = sorted(
        [(c, iso) for c, iso in char_isolation.items() if iso >= isolation_threshold],
        key=lambda x: x[1], reverse=True
    )
    print(f"  Isolation scores: { {c: f'{iso:.3f}' for c, iso in char_isolation.items()} }")
    print(f"  Meaningful characteristics (>={isolation_threshold}): {[(c, f'{iso:.3f}') for c, iso in meaningful]}")

    K = num_communities

    def _proportional_alloc(weights, total):
        """Allocate `total` integers proportional to weights (Bresenham style, min 1 each)."""
        w = np.array(weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum == 0:
            base = total // len(w)
            sizes = [base] * len(w)
            for i in range(total % len(w)):
                sizes[i] += 1
            return sizes
        floats = w / w_sum * total
        sizes = np.maximum(1, np.floor(floats).astype(int)).tolist()
        remainder = total - sum(sizes)
        fracs = (floats - np.floor(floats)).tolist()
        for i in sorted(range(len(fracs)), key=lambda x: -fracs[x]):
            if remainder <= 0:
                break
            sizes[i] += 1
            remainder -= 1
        return sizes

    num_blocks = len(meaningful) and len(set(
        tuple(G.group_to_attrs.get(gid, {}).get(c) for c, _ in meaningful[:2])
        for gid in sorted_groups
    ))
    print(f"  Blocks: {num_blocks}, communities: {K}")

    # Build group -> community list mapping (each group maps only to its block)
    if not meaningful:
        group_comms = {gid: list(range(K)) for gid in sorted_groups}

    elif len(meaningful) == 1:
        char1, _ = meaningful[0]
        val_to_groups = {}
        for gid in sorted_groups:
            v = G.group_to_attrs.get(gid, {}).get(char1)
            val_to_groups.setdefault(v, []).append(gid)
        vals1 = sorted(val_to_groups.keys(), key=str)

        pops = [sum(len(group_nodes_map[g]) for g in val_to_groups[v]) for v in vals1]
        sizes = _proportional_alloc(pops, K)

        val1_comms = {}
        start = 0
        for v, size in zip(vals1, sizes):
            val1_comms[v] = list(range(start, start + size))
            start += size

        group_comms = {}
        for gid in sorted_groups:
            v1 = G.group_to_attrs.get(gid, {}).get(char1)
            group_comms[gid] = val1_comms.get(v1, [])

    else:
        char1, _ = meaningful[0]
        char2, _ = meaningful[1]

        pair_pop = {}
        for gid in sorted_groups:
            v1 = G.group_to_attrs.get(gid, {}).get(char1)
            v2 = G.group_to_attrs.get(gid, {}).get(char2)
            pair_pop[(v1, v2)] = pair_pop.get((v1, v2), 0) + len(group_nodes_map[gid])

        pairs = sorted(pair_pop.keys(), key=lambda p: (str(p[0]), str(p[1])))
        pops = [pair_pop[p] for p in pairs]
        sizes = _proportional_alloc(pops, K)

        block_comms = {}
        start = 0
        for pair, size in zip(pairs, sizes):
            block_comms[pair] = list(range(start, start + size))
            start += size

        group_comms = {}
        for gid in sorted_groups:
            v1 = G.group_to_attrs.get(gid, {}).get(char1)
            v2 = G.group_to_attrs.get(gid, {}).get(char2)
            group_comms[gid] = block_comms.get((v1, v2), [])

    G.number_of_communities = K
    target = max(1, N // K)
    community_count = np.zeros(K, dtype=np.int64)

    for gid in sorted_groups:
        nodes = np.array(group_nodes_map[gid], dtype=np.int64)
        if len(nodes) == 0:
            continue
        rng.shuffle(nodes)
        chosen = group_comms.get(gid, list(range(K))) or list(range(K))

        # Weight each community by remaining capacity (headroom above current fill)
        headroom = np.array([max(0, target - community_count[c]) for c in chosen], dtype=np.float64)
        if headroom.sum() == 0:
            headroom = np.ones(len(chosen), dtype=np.float64)
        alloc = _proportional_alloc(headroom.tolist(), len(nodes))

        idx = 0
        for comm, count in zip(chosen, alloc):
            community_count[comm] += count
            for node in nodes[idx:idx + count]:
                node_int = int(node)
                G.communities_to_nodes.setdefault((comm, gid), []).append(node_int)
                G.nodes_to_communities[node_int] = comm
                G.communities_to_groups.setdefault(comm, []).append(gid)
            idx += count

    # Spread communities evenly across [0,1) with a deterministic permutation so
    # same-block communities don't cluster and Phase B ring search is unbiased.
    coord_pos = rng.permutation(K)
    jitter_std = 1e-4
    node_coordinates = {}
    for node_int, comm in G.nodes_to_communities.items():
        theta_c = float(coord_pos[comm]) / K
        node_coordinates[int(node_int)] = (theta_c + float(rng.normal(0, jitter_std))) % 1.0
    G.node_coordinates = node_coordinates

    print(f"\nSegregation-based assignment complete: {N} nodes -> {K} communities")

def create_communities(pops_path, links_path, scale, number_of_communities=None,
                       output_path='communities.json', community_size_distribution='natural',
                       pop_column='n', src_suffix='_src', dst_suffix='_dst',
                       link_column='n', min_group_size=0, verbose=True,
                       new_comm_penalty=3.0, allow_new_communities=True,
                       mode='capacity', sa_fraction=1.0, overcap_penalty=float('inf'), max_eval=100,
                       resolution=1.0, mixing_floor=0.1, isolation_threshold=0.05):
    """
    Create community assignments and save them to a JSON file.

    This is a standalone step that can be run independently from network
    generation. The output file can later be passed to generate() via
    the community_file parameter.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    scale : float
        Population scaling factor
    number_of_communities : int or None
        Number of communities to create. Required for mode='probability'.
        For mode='capacity', this is the initial count (may grow dynamically).
    output_path : str
        Path for the output JSON file
    community_size_distribution : str or array-like, optional
        Controls community size distribution (default 'natural')
    pop_column : str, optional
        Column name for population counts (default 'n')
    src_suffix : str, optional
        Suffix for source group columns (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns (default '_dst')
    link_column : str, optional
        Column name for link counts (default 'n')
    min_group_size : int, optional
        Minimum nodes per group after scaling (default 0)
    verbose : bool, optional
        Whether to print progress information
    mode : str, optional
        'probability' (default): match probability distributions
        'capacity': match absolute edge counts with feasibility constraints

    Returns
    -------
    str
        Path to the saved JSON file
    """
    import json
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    if verbose:
        print("="*60)
        print(f"COMMUNITY CREATION (mode={mode})")
        print("="*60)

    # Create a temporary graph and initialize nodes
    G = NetworkXGraph()
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes in {len(G.group_ids)} groups")

    # Compute maximum link counts (needed for affinity matrix)
    _compute_maximum_num_links(G, links_path, scale, src_suffix=src_suffix,
                                dst_suffix=dst_suffix, link_column=link_column,
                                verbose=verbose)

    if mode == 'probability':
        if number_of_communities is None:
            raise ValueError("number_of_communities is required for mode='probability'")
        if verbose:
            print(f"\nAssigning nodes to {number_of_communities} communities (probability mode)...")
        populate_communities(G, number_of_communities,
                             community_size_distribution=community_size_distribution)
    elif mode == 'clustered':
        if number_of_communities is None:
            number_of_communities = 100
        if verbose:
            print(f"\nAssigning nodes via Louvain group clustering "
                  f"(target={number_of_communities}, resolution={resolution})...")
        populate_communities_clustered(G, number_of_communities,
                                       community_size_distribution=community_size_distribution,
                                       resolution=resolution)
    elif mode == 'adaptive':
        if number_of_communities is None:
            number_of_communities = 100
        if verbose:
            print(f"\nAssigning nodes via K_g-coordinate adaptive assignment "
                  f"(target={number_of_communities} communities)...")
        populate_communities_adaptive(G, number_of_communities)
    elif mode == 'segregation':
        if number_of_communities is None:
            number_of_communities = 100
        if verbose:
            print(f"\nAssigning nodes via segregation-driven hierarchical assignment "
                  f"(target={number_of_communities} communities)...")

        populate_communities_segregation(G, number_of_communities,
                                         mixing_floor=mixing_floor,
                                         isolation_threshold=isolation_threshold)
    else:
        if number_of_communities is None:
            number_of_communities = 100
        fast = 'greedy' if mode == 'capacity_greedy' else (mode == 'capacity_fast')
        if verbose:
            if fast:
                print(f"\nAssigning nodes to {number_of_communities} communities (fast mode, no SA)...")
            else:
                print(f"\nAssigning nodes (penalty={new_comm_penalty}, initial communities={number_of_communities})...")
        populate_communities_capacity(G, number_of_communities,
                                      community_size_distribution=community_size_distribution,
                                      new_comm_penalty=new_comm_penalty,
                                      allow_new_communities=allow_new_communities,
                                      fast=fast,
                                      sa_fraction=sa_fraction,
                                      overcap_penalty=overcap_penalty,
                                      max_eval=max_eval)

    if verbose:
        # Compute per-community total sizes
        comm_sizes = {}
        for (comm_id, _group_id), nodes in G.communities_to_nodes.items():
            comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + len(nodes)
        sizes = list(comm_sizes.values())
        if sizes:
            print(f"\nCommunity size statistics ({len(sizes)} communities):")
            print(f"  Largest:  {max(sizes)}")
            print(f"  Smallest: {min(sizes)}")
            print(f"  Mean:     {np.mean(sizes):.1f}")
            print(f"  Median:   {np.median(sizes):.1f}")
            print(f"  Std:      {np.std(sizes):.1f}")
            print(f"  Q1/Q3:    {np.quantile(sizes, 0.25):.1f} / {np.quantile(sizes, 0.75):.1f}")

    # Serialize to JSON (convert numpy types to native Python types)
    # For large graphs store probability matrix sparsely to avoid huge JSON files
    _pm = G.probability_matrix
    _n = _pm.shape[0] if hasattr(_pm, 'shape') else len(_pm)
    if _n == 0:
        # Empty placeholder (e.g. 1-community fast path) — store as empty sparse
        _pm_serial = {'sparse': True, 'shape': [0, 0], 'rows': [], 'cols': [], 'vals': []}
    elif _n > 500:
        # Store as sparse COO without densifying
        import scipy.sparse as _sp
        if _sp.issparse(_pm):
            _coo = _pm.tocoo()
            _pm_serial = {
                'sparse': True,
                'shape': [int(_n), int(_n)],
                'rows': _coo.row.tolist(),
                'cols': _coo.col.tolist(),
                'vals': _coo.data.tolist(),
            }
        else:
            # Dense numpy array — use argwhere (only reaches here for small-to-medium n)
            _nz = np.argwhere(_pm > 0)
            _pm_serial = {
                'sparse': True,
                'shape': [int(_n), int(_n)],
                'rows': _nz[:, 0].tolist(),
                'cols': _nz[:, 1].tolist(),
                'vals': _pm[_nz[:, 0], _nz[:, 1]].tolist(),
            }
    else:
        _pm_serial = G.probability_matrix.tolist()

    data = {
        'number_of_communities': int(G.number_of_communities),
        'probability_matrix': _pm_serial,
        'nodes_to_communities': {
            str(k): int(v) for k, v in G.nodes_to_communities.items()
        },
        'communities_to_nodes': {
            str(k): [int(n) for n in v] for k, v in G.communities_to_nodes.items()
        },
        'communities_to_groups': {
            str(k): [int(g) for g in v] for k, v in G.communities_to_groups.items()
        }
    }
    if hasattr(G, 'node_coordinates') and G.node_coordinates:
        data['node_coordinates'] = {str(k): float(v) for k, v in G.node_coordinates.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    if verbose:
        print(f"\nCommunity assignments saved to {output_path}")
        print("="*60 + "\n")

    return output_path


def create_hierarchical_community_file(
    household_community_file,
    pops_path,
    links_path,
    scale,
    target_num_communities,
    output_path,
    pop_column='n',
    src_suffix='_src',
    dst_suffix='_dst',
    link_column='n',
    verbose=True,
):
    """
    Create a community file where household communities are grouped into
    super-communities (e.g. 5000 household communities → 5 buren communities).

    Nodes in the same household community are guaranteed to be in the same
    super-community. Uses block assignment for locality preservation.

    Parameters
    ----------
    household_community_file : str
        Path to the household community JSON (from create_communities())
    pops_path : str
        Path to population CSV (for computing probability matrix)
    links_path : str
        Path to interaction CSV for the target layer
    scale : float
        Population scaling factor
    target_num_communities : int
        Number of super-communities to create
    output_path : str
        Path to write the output JSON
    """
    import json
    import math
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    # Load household community assignments
    with open(household_community_file, 'r', encoding='utf-8') as f:
        hh_data = json.load(f)

    hh_num_communities = hh_data['number_of_communities']
    hh_nodes_to_communities = {int(k): v for k, v in hh_data['nodes_to_communities'].items()}

    # Block assignment: group household communities into super-communities
    block_size = math.ceil(hh_num_communities / target_num_communities)

    super_nodes_to_communities = {}
    for node, hh_comm in hh_nodes_to_communities.items():
        super_comm = min(hh_comm // block_size, target_num_communities - 1)
        super_nodes_to_communities[node] = super_comm

    # Compute probability matrix from this layer's own link data
    G_temp = NetworkXGraph('_temp_hierarchical')
    init_nodes(G_temp, pops_path, scale, pop_column=pop_column)
    _compute_maximum_num_links(G_temp, links_path, scale,
                               src_suffix=src_suffix, dst_suffix=dst_suffix,
                               link_column=link_column, verbose=False)

    n_groups = len(G_temp.group_ids)
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G_temp.maximum_num_links.items():
        affinity[i, j] = count
    row_sums = affinity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probability_matrix = affinity / row_sums

    # Clean up temp directory
    import shutil, os
    if os.path.exists('_temp_hierarchical'):
        shutil.rmtree('_temp_hierarchical')

    # Write community JSON
    data = {
        'number_of_communities': target_num_communities,
        'probability_matrix': probability_matrix.tolist(),
        'nodes_to_communities': {str(k): int(v) for k, v in super_nodes_to_communities.items()},
        'communities_to_nodes': {},
        'communities_to_groups': {},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    if verbose:
        print(f"  Hierarchical communities: {hh_num_communities} household -> "
              f"{target_num_communities} super-communities (block_size={block_size})")
        print(f"  Saved to {output_path}")

    return output_path


def load_communities(G, community_file_path):
    """
    Load community assignments from a JSON file into a NetworkXGraph object.

    Node-to-community assignments are loaded from the file, but
    communities_to_nodes and communities_to_groups are recalculated from
    the actual graph nodes. This ensures correctness when the graph has
    fewer groups than when the community file was created.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes already initialized
    community_file_path : str
        Path to the JSON file created by create_communities()
    """
    import json

    with open(community_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G.number_of_communities = data['number_of_communities']
    _pm_data = data['probability_matrix']
    if isinstance(_pm_data, dict) and _pm_data.get('sparse'):
        _n = _pm_data['shape'][0]
        if _n == 0:
            G.probability_matrix = np.zeros((0, 0), dtype=np.float64)
        elif _n > 5000:
            # Too large to densify — reconstruct as scipy sparse matrix
            from scipy.sparse import csr_matrix as _csr
            _rows = _pm_data['rows']
            _cols = _pm_data['cols']
            _vals = _pm_data['vals']
            G.probability_matrix = _csr(
                (_vals, (_rows, _cols)), shape=(_n, _n), dtype=np.float64
            )
        else:
            _pm_arr = np.zeros((_n, _n), dtype=np.float64)
            for r, c, v in zip(_pm_data['rows'], _pm_data['cols'], _pm_data['vals']):
                _pm_arr[r, c] = v
            G.probability_matrix = _pm_arr
    else:
        G.probability_matrix = np.array(_pm_data)

    # Load node-to-community assignments, keeping only nodes present in graph
    graph_nodes = set(G.graph.nodes)

    G.nodes_to_communities = {}
    for k, v in data['nodes_to_communities'].items():
        node = int(k)
        if node in graph_nodes:
            G.nodes_to_communities[node] = v

    unassigned = graph_nodes - set(G.nodes_to_communities.keys())
    if unassigned:
        print(f"Warning: {len(unassigned)} graph nodes have no community assignment")

    # Recalculate communities_to_nodes and communities_to_groups
    # from the actual graph nodes, so they reflect current groups
    communities_groups = {}

    for node, community_id in G.nodes_to_communities.items():
        group_id = G.nodes_to_group[node]
        key = (community_id, group_id)
        if key not in G.communities_to_nodes:
            G.communities_to_nodes[key] = []
        G.communities_to_nodes[key].append(node)

        if community_id not in communities_groups:
            communities_groups[community_id] = set()
        communities_groups[community_id].add(group_id)

    G.communities_to_groups = {
        comm: list(groups) for comm, groups in communities_groups.items()
    }

    if 'node_coordinates' in data:
        G.node_coordinates = {int(k): float(v) for k, v in data['node_coordinates'].items()}
