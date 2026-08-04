"""Community detection and management module for ASNU.

create_communities                    : build + refine a partition, serialise to JSON
build_group_pair_to_communities_lookup: map group pairs to their shared communities
populate_communities                  : assign nodes to communities, then refine
connect_all_within_communities        : fully connect nodes within each community
fill_unfulfilled_group_pairs          : complete group pairs short of their edge target
load_communities                      : load community assignments from JSON
"""
import json
import random
from collections import defaultdict
from itertools import product

import numpy as np
from tqdm import tqdm

from asnu.core.refine import refine_communities_move


def create_communities(pops_path, links_path, scale, fraction_of_communities=None,
                       output_path='communities.json',
                       pop_column='n', src_suffix='_src', dst_suffix='_dst',
                       link_column='n', verbose=True,
                       refine_swaps=300_000, loss_goal=0):
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    G = NetworkXGraph()
    init_nodes(G, pops_path, scale, pop_column=pop_column)
    _compute_maximum_num_links(G, links_path, scale, src_suffix=src_suffix,
                               dst_suffix=dst_suffix, link_column=link_column,
                               verbose=verbose)
    number_of_communities = (int(fraction_of_communities * G.graph.number_of_nodes())
                             if fraction_of_communities is not None else None)

    loss = populate_communities(G, number_of_communities, refine_swaps, loss_goal=loss_goal)

    data = {
        'number_of_communities': int(G.number_of_communities),
        'nodes_to_communities': {str(k): int(v) for k, v in G.nodes_to_communities.items()},
        'communities_to_nodes': {str(k): [int(n) for n in v]
                                 for k, v in G.communities_to_nodes.items()},
        'communities_to_groups': {str(k): [int(g) for g in v]
                                  for k, v in G.communities_to_groups.items()},
    }
    if getattr(G, 'node_coordinates', None):
        data['node_coordinates'] = {str(k): float(v) for k, v in G.node_coordinates.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"\nCommunity assignments saved to {output_path}")
    return output_path, loss


def build_group_pair_to_communities_lookup(G, verbose=False):
    if verbose:
        print("Building community lookup for group pairs...")

    if G.number_of_communities == 1:
        class _SingleCommLookup(defaultdict):
            def __missing__(self, key):
                return [0]

        if verbose:
            print("  Single community: using lazy lookup (skipping O(n^2) build)")
        return _SingleCommLookup(list)

    group_to_comms = defaultdict(list)
    for community_id in range(G.number_of_communities):
        for g in G.communities_to_groups.get(community_id, []):
            group_to_comms[g].append(community_id)

    group_pair_to_communities = {}
    for src_id, dst_id in G.maximum_num_links:
        sc = group_to_comms.get(src_id)
        dc = group_to_comms.get(dst_id)
        if sc and dc:
            shared = list(set(sc) & set(dc))
            if shared:
                group_pair_to_communities[(src_id, dst_id)] = shared

    if verbose:
        avg = np.mean([len(v) for v in group_pair_to_communities.values()]) \
            if group_pair_to_communities else 0
        print(f"  Found {len(group_pair_to_communities)} group pairs")
        print(f"  Average communities per pair: {avg:.1f}")

    return group_pair_to_communities


def _proportional_alloc(weights, total):
    """Split `total` across buckets proportional to `weights` (each bucket >= 1)."""
    w = np.array(weights, dtype=np.float64)
    if w.sum() == 0:
        base, rem = divmod(total, len(w))
        return [base + (1 if i < rem else 0) for i in range(len(w))]
    floats = w / w.sum() * total
    sizes = np.maximum(1, np.floor(floats).astype(int)).tolist()
    remainder = total - sum(sizes)
    for i in np.argsort(-(floats - np.floor(floats))):
        if remainder <= 0:
            break
        sizes[i] += 1
        remainder -= 1
    return sizes


def populate_communities(G, num_communities, refine_swaps, loss_goal=0, seed=42):
    """Assign nodes to communities, then refine.

    Nodes from each group are shuffled and distributed proportionally across all
    communities (favoring those with the most headroom), then handed to the Rust
    refinement pass to minimize the edge-budget loss.
    """
    rng = np.random.default_rng(seed)
    n_groups = len(G.group_ids)
    sorted_groups = sorted(int(g) for g in G.group_ids)
    N = G.graph.number_of_nodes()
    K = num_communities

    G.number_of_communities = K
    target = max(1, N // K)
    community_count = np.zeros(K, dtype=np.int64)

    for gid in sorted_groups:
        nodes = np.array(G.group_to_nodes.get(gid, []), dtype=np.int64)
        if len(nodes) == 0:
            continue
        rng.shuffle(nodes)

        headroom = np.maximum(0, target - community_count).astype(np.float64)
        if headroom.sum() == 0:
            headroom = np.ones(K, dtype=np.float64)
        alloc = _proportional_alloc(headroom.tolist(), len(nodes))

        idx = 0
        for comm, count in enumerate(alloc):
            community_count[comm] += count
            for node in nodes[idx:idx + count]:
                node_int = int(node)
                G.communities_to_nodes.setdefault((comm, gid), []).append(node_int)
                G.nodes_to_communities[node_int] = comm
                G.communities_to_groups.setdefault(comm, []).append(gid)
            idx += count

    print(f"\nRefining communities with {refine_swaps} swap iterations...")

    refine_nodes = np.array(list(G.graph.nodes), dtype=np.int64)
    refine_node_groups = np.array([G.nodes_to_group[n] for n in refine_nodes], dtype=np.int64)
    refine_assignments = np.array([G.nodes_to_communities[int(n)] for n in refine_nodes],
                                  dtype=np.int64)
    refine_budget = {(int(k[0]), int(k[1])): int(v)
                     for k, v in G.maximum_num_links.items() if v > 0}

    new_assignments, loss = refine_communities_move(
        refine_assignments, refine_node_groups, refine_budget,
        n_groups, G.number_of_communities, refine_swaps, loss_goal, 1, 42)

    print("Refinement complete.")

    K_new = int(new_assignments.max()) + 1
    G.number_of_communities = K_new
    coord_pos = rng.permutation(K_new)

    G.communities_to_nodes = {}
    G.communities_to_groups = {}
    G.nodes_to_communities = {}
    G.node_coordinates = {}
    for node_arr, comm_arr, group_arr in zip(refine_nodes, new_assignments, refine_node_groups):
        node, comm, group = int(node_arr), int(comm_arr), int(group_arr)
        G.nodes_to_communities[node] = comm
        G.communities_to_nodes.setdefault((comm, group), []).append(node)
        G.communities_to_groups.setdefault(comm, []).append(group)
        G.node_coordinates[node] = float(coord_pos[comm]) / K_new % 1.0

    print(f"\nAssignment complete: {N} nodes -> {K_new} communities")
    return loss


def connect_all_within_communities(G, verbose=False):
    """Connect all nodes within each community to each other."""
    stats = {'total_edges': 0, 'edges_per_community': {}}

    communities_nodes = [[] for _ in range(G.number_of_communities)]
    for node, comm in G.nodes_to_communities.items():
        communities_nodes[comm].append(node)

    for community_id, community_nodes in enumerate(communities_nodes):
        if not community_nodes:
            continue

        edges_to_add = [(src, dst) for src, dst in product(community_nodes, repeat=2)
                        if src != dst]
        G.graph.add_edges_from(edges_to_add)

        stats['edges_per_community'][community_id] = len(edges_to_add)
        stats['total_edges'] += len(edges_to_add)

        if (community_id + 1) % 5000 == 0 or community_id == 0:
            print(f"  Connected {community_id + 1}/{G.number_of_communities} communities "
                  f"({(community_id + 1) / G.number_of_communities * 100:.1f}%)")

    return stats


def fill_unfulfilled_group_pairs(G, reciprocity_p, verbose=True):
    """Complete any group pairs that didn't reach their target edge count."""
    if verbose:
        print("\nFilling unfulfilled group pairs...")

    stats = {'total_pairs': 0, 'fulfilled_pairs': 0, 'unfulfilled_pairs': 0,
             'edges_added': 0, 'reciprocal_edges_added': 0}

    unfulfilled_pairs = []
    for src_id, dst_id in G.maximum_num_links:
        maximum = G.maximum_num_links[(src_id, dst_id)]
        existing = G.existing_num_links.get((src_id, dst_id), 0)
        stats['total_pairs'] += 1
        if maximum == 0:
            continue
        if existing < maximum:
            unfulfilled_pairs.append((src_id, dst_id, existing, maximum))
            stats['unfulfilled_pairs'] += 1
        else:
            stats['fulfilled_pairs'] += 1

    if verbose:
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Fulfilled: {stats['fulfilled_pairs']}")
        print(f"  Unfulfilled: {stats['unfulfilled_pairs']}")

    def _try_add_edge(s, d, src_id, dst_id, maximum):
        """Add s->d (and maybe its reciprocal) if valid. Returns True if s->d added."""
        if s == d or G.graph.has_edge(s, d):
            return False
        G.graph.add_edge(s, d)
        G.existing_num_links[(src_id, dst_id)] += 1
        stats['edges_added'] += 1
        if reciprocity_p > 0 and random.random() < reciprocity_p:
            if (G.existing_num_links.get((dst_id, src_id), 0)
                    < G.maximum_num_links.get((dst_id, src_id), 0)
                    and not G.graph.has_edge(d, s)):
                G.graph.add_edge(d, s)
                G.existing_num_links[(dst_id, src_id)] += 1
                stats['reciprocal_edges_added'] += 1
                if dst_id == src_id:
                    stats['edges_added'] += 1
        return True

    def _fill_from_pool(src_pool, dst_pool, src_id, dst_id, maximum):
        if not src_pool or not dst_pool:
            return
        src_arr, dst_arr = np.array(src_pool), np.array(dst_pool)
        batch = max((maximum - G.existing_num_links[(src_id, dst_id)]) * 4, 512)
        for _ in range(10):
            if G.existing_num_links[(src_id, dst_id)] >= maximum:
                return
            srcs = np.random.choice(src_arr, size=batch).tolist()
            dsts = np.random.choice(dst_arr, size=batch).tolist()
            added_this_round = 0
            for s, d in zip(srcs, dsts):
                if G.existing_num_links[(src_id, dst_id)] >= maximum:
                    return
                added_this_round += _try_add_edge(s, d, src_id, dst_id, maximum)
            if added_this_round == 0:
                return

    def _fill_within_shared_communities(src_comm, dst_comm, src_id, dst_id, maximum):
        shared = list(set(src_comm) & set(dst_comm))
        if not shared:
            return
        src_arrs = [np.array(src_comm[c]) for c in shared]
        dst_arrs = [np.array(dst_comm[c]) for c in shared]
        n_comm = len(shared)

        weights = np.array([len(s) * len(d) for s, d in zip(src_arrs, dst_arrs)], dtype=float)
        weights /= weights.sum()
        src_lens = np.array([len(a) for a in src_arrs], dtype=np.int64)
        dst_lens = np.array([len(a) for a in dst_arrs], dtype=np.int64)
        src_offsets = np.concatenate([[0], src_lens.cumsum()])
        dst_offsets = np.concatenate([[0], dst_lens.cumsum()])
        src_flat = np.concatenate(src_arrs)
        dst_flat = np.concatenate(dst_arrs)

        batch = max((maximum - G.existing_num_links[(src_id, dst_id)]) * 4, 512)
        for _ in range(10):
            if G.existing_num_links[(src_id, dst_id)] >= maximum:
                return
            ci = np.random.choice(n_comm, size=batch, p=weights)
            s_batch = src_flat[src_offsets[ci] +
                               (np.random.random(batch) * src_lens[ci]).astype(np.int64)]
            d_batch = dst_flat[dst_offsets[ci] +
                               (np.random.random(batch) * dst_lens[ci]).astype(np.int64)]
            added_this_round = 0
            for s, d in zip(s_batch.tolist(), d_batch.tolist()):
                if G.existing_num_links[(src_id, dst_id)] >= maximum:
                    return
                added_this_round += _try_add_edge(int(s), int(d), src_id, dst_id, maximum)
            if added_this_round == 0:
                return

    if unfulfilled_pairs:
        total_needed = sum(mx - ex for _, _, ex, mx in unfulfilled_pairs)
        pbar = (tqdm(total=total_needed, unit='edge', desc='Filling pairs', dynamic_ncols=True)
                if tqdm and verbose else None)

        for src_id, dst_id, _existing, maximum in unfulfilled_pairs:
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])
            if not src_nodes or not dst_nodes:
                continue

            edges_before_pair = stats['edges_added']

            src_comm, dst_comm = defaultdict(list), defaultdict(list)
            for node in src_nodes:
                comm = G.nodes_to_communities.get(node)
                if comm is not None:
                    src_comm[comm].append(node)
            for node in dst_nodes:
                comm = G.nodes_to_communities.get(node)
                if comm is not None:
                    dst_comm[comm].append(node)

            _fill_within_shared_communities(src_comm, dst_comm, src_id, dst_id, maximum)
            _fill_from_pool(src_nodes, dst_nodes, src_id, dst_id, maximum)

            if pbar is not None:
                pbar.update(stats['edges_added'] - edges_before_pair)

        if pbar is not None:
            pbar.close()

    if verbose:
        print(f"  Edges added: {stats['edges_added']}")
        print(f"  Reciprocal edges added: {stats['reciprocal_edges_added']}")

    return stats


def load_communities(G, community_file_path):
    """Load community assignments from a JSON file into a NetworkXGraph object."""
    with open(community_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G.number_of_communities = data['number_of_communities']

    graph_nodes = set(G.graph.nodes)
    G.nodes_to_communities = {int(k): v for k, v in data['nodes_to_communities'].items()
                              if int(k) in graph_nodes}

    unassigned = graph_nodes - set(G.nodes_to_communities)
    if unassigned:
        print(f"Warning: {len(unassigned)} graph nodes have no community assignment")

    communities_groups = defaultdict(set)
    for node, community_id in G.nodes_to_communities.items():
        group_id = G.nodes_to_group[node]
        G.communities_to_nodes.setdefault((community_id, group_id), []).append(node)
        communities_groups[community_id].add(group_id)

    G.communities_to_groups = {comm: list(groups)
                               for comm, groups in communities_groups.items()}

    if 'node_coordinates' in data:
        G.node_coordinates = {int(k): float(v) for k, v in data['node_coordinates'].items()}