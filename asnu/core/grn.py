import bisect
import math
import random
from itertools import chain

PHASE_B_COMM_WINDOW = 200


def _links(G, pair):
    return G.existing_num_links.get(pair, 0)


def _max_links(G, pair):
    return G.maximum_num_links.get(pair, 0)


def _node_to_community(G):
    cache = getattr(G, "_node_to_community", None)
    if cache is None:
        cache = {n: comm_id
                 for (comm_id, _gid), nodes in G.communities_to_nodes.items()
                 for n in nodes}
        G._node_to_community = cache
    return cache


def _ensure_edge_index(G):
    """Lazily build a fast side-index for the hot edge-creation loop:
      G._edge_set : set of (u, v) for O(1) has_edge (networkx has_edge is slow)
      G._adj_out  : {u: [v, ...]} out-neighbours, insertion-ordered
      G._adj_in   : {v: [u, ...]} in-neighbours, insertion-ordered

    Seeded from whatever edges already exist on G.graph so transitivity sees the
    same neighbourhood networkx would report. The networkx graph stays
    authoritative; this index just answers has_edge / neighbours faster and is
    kept in sync inside _insert.
    """
    if getattr(G, "_edge_set", None) is not None:
        return
    edge_set = set()
    adj_out, adj_in = {}, {}
    for u, v in G.graph.edges():
        edge_set.add((u, v))
        adj_out.setdefault(u, []).append(v)
        adj_in.setdefault(v, []).append(u)
    G._edge_set = edge_set
    G._adj_out = adj_out
    G._adj_in = adj_in


def _undirected_neighbors(G, d):
    # out-neighbours first, then in-neighbours, deduplicated -- same order the
    # networkx successors()+predecessors() scan produced, so RNG-dependent
    # transitivity behaviour is unchanged.
    out = G._adj_out.get(d)
    inn = G._adj_in.get(d)
    if out is None and inn is None:
        return []
    if inn is None:
        return list(dict.fromkeys(out))
    if out is None:
        return list(dict.fromkeys(inn))
    return list(dict.fromkeys(chain(out, inn)))


def _insert(G, s, d, pair):
    if s == d or (s, d) in G._edge_set:
        return False
    # Deferred: write to the fast index only. G.graph is bulk-loaded once at the
    # end of edge creation (_flush_edges_to_graph). Nothing reads G.graph
    # mid-creation -- transitivity reads the index.
    G._edge_set.add((s, d))
    G._adj_out.setdefault(s, []).append(d)
    G._adj_in.setdefault(d, []).append(s)
    G.existing_num_links[pair] = _links(G, pair) + 1
    return True


def _flush_edges_to_graph(G):
    """Bulk-load every indexed edge into G.graph in one call, then drop the
    index. Call at the end of edge creation so G.graph is current for anything
    downstream (fill_unfulfilled, counts, save). add_edges_from ignores the
    duplicates re-included from the seed, so we pass the whole set directly."""
    idx = getattr(G, "_edge_set", None)
    if idx is None:
        return
    G.graph.add_edges_from(idx)
    G._edge_set = G._adj_out = G._adj_in = None


def _maybe_reciprocate(G, s, d, src_group, dst_group, reciprocity_p):
    # Deliberately checks only the reverse pair's MAX budget, not its target,
    # so a self-pair may overshoot its target by the reciprocal edge.
    if random.random() >= reciprocity_p:
        return
    rev = (dst_group, src_group)
    if _links(G, rev) < _max_links(G, rev):
        _insert(G, d, s, rev)


def _apply_transitivity(G, s, d, src_group, budget_pair, target,
                        int_trans_p, ext_trans_p, reciprocity_p):
    n2c = _node_to_community(G)
    d_comm = n2c.get(d)

    for n in _undirected_neighbors(G, d):
        if _links(G, budget_pair) >= target:
            break
        if n == s:
            continue

        internal = d_comm is not None and n2c.get(n) == d_comm
        p = int_trans_p if internal else ext_trans_p
        if random.random() >= p:
            continue

        n_group = G.nodes_to_group.get(n)
        if n_group is None:
            continue
        pair = (src_group, n_group)
        if pair not in G.maximum_num_links:
            continue
        if _links(G, pair) < G.maximum_num_links[pair] and _insert(G, s, n, pair):
            _maybe_reciprocate(G, s, n, src_group, n_group, reciprocity_p)


def _create_edge(G, s, d, src_group, dst_group, target,
                 reciprocity_p, int_trans_p, ext_trans_p):
    pair = (src_group, dst_group)
    if not _insert(G, s, d, pair):
        return False
    _maybe_reciprocate(G, s, d, src_group, dst_group, reciprocity_p)
    if int_trans_p > 0 or ext_trans_p > 0:
        _apply_transitivity(G, s, d, src_group, pair, target,
                            int_trans_p, ext_trans_p, reciprocity_p)
    return True


def _effective_transitivity(transitivity_p, internal_p, external_p):
    int_p = transitivity_p if internal_p < 0 else internal_p
    ext_p = transitivity_p if external_p < 0 else external_p
    return int_p, ext_p


def establish_links(G, src_id, dst_id,
                    target_link_count, reciprocity_p, transitivity_p,
                    valid_communities=None,
                    bridge_probability=0, number_of_communities=1,
                    internal_transitivity_p=-1.0, external_transitivity_p=-1.0):
    # Phase A for one (src, dst) pair. Run over ALL pairs before any Phase B.
    _ensure_edge_index(G)
    int_p, ext_p = _effective_transitivity(
        transitivity_p, internal_transitivity_p, external_transitivity_p)

    pair = (src_id, dst_id)
    if _links(G, pair) >= target_link_count:
        return True
    if not valid_communities:
        return False

    comm_order = list(dict.fromkeys(valid_communities))
    random.shuffle(comm_order)
    n_comms = len(comm_order)
    src_node_cache = {}
    MAX_PASSES = 3

    for _pass in range(MAX_PASSES):
        if _links(G, pair) >= target_link_count:
            break

        for community_id in comm_order:
            remaining = target_link_count - _links(G, pair)
            if remaining <= 0:
                break
            quota = max(1, math.ceil(remaining / n_comms))

            src_nodes = src_node_cache.setdefault(
                community_id,
                G.communities_to_nodes.get((community_id, src_id), []))
            if not src_nodes:
                continue

            if (bridge_probability > 0 and number_of_communities > 1
                    and random.random() < bridge_probability):
                # Bridge: a random shortcut to any OTHER community (small-world
                # rewiring). Community ids carry no spatial meaning (ring
                # coordinates are a random permutation), so a uniform pick over
                # the other communities is the coherent choice. Excludes the
                # source community so a bridge always leaves.
                dst_community = random.randrange(number_of_communities - 1)
                if dst_community >= community_id:
                    dst_community += 1
            else:
                dst_community = community_id

            dst_nodes = G.communities_to_nodes.get((dst_community, dst_id), [])
            if not dst_nodes:
                continue

            created = 0
            for _attempt in range(quota * 3):
                if created >= quota or _links(G, pair) >= target_link_count:
                    break
                s = random.choice(src_nodes)
                d = random.choice(dst_nodes)
                if _create_edge(G, s, d, src_id, dst_id, target_link_count,
                                reciprocity_p, int_p, ext_p):
                    created += 1

    return _links(G, pair) >= target_link_count


def establish_links_phase_b(G, src_id, dst_id, target_link_count,
                            reciprocity_p, transitivity_p,
                            internal_transitivity_p=-1.0,
                            external_transitivity_p=-1.0):
    # Run only after Phase A has completed for ALL pairs. No-op without
    # G.node_coordinates.
    _ensure_edge_index(G)
    int_p, ext_p = _effective_transitivity(
        transitivity_p, internal_transitivity_p, external_transitivity_p)

    pair = (src_id, dst_id)
    if _links(G, pair) >= target_link_count:
        return True

    node_coordinates = getattr(G, "node_coordinates", None)
    if node_coordinates is None:
        return False

    if not hasattr(G, "_phase_b_src_sorted"):
        G._phase_b_src_sorted = {}
        G._phase_b_dst_comm_sorted = {}

    if src_id not in G._phase_b_src_sorted:
        G._phase_b_src_sorted[src_id] = sorted(
            (node_coordinates.get(n, 0.5), n)
            for (_comm, gid), nodes in G.communities_to_nodes.items()
            if gid == src_id
            for n in nodes
        )
    if dst_id not in G._phase_b_dst_comm_sorted:
        G._phase_b_dst_comm_sorted[dst_id] = sorted(
            (sum(node_coordinates.get(n, 0.5) for n in nodes) / len(nodes), comm_id)
            for (comm_id, gid), nodes in G.communities_to_nodes.items()
            if gid == dst_id and nodes
        )

    src_sorted = G._phase_b_src_sorted[src_id]
    dst_comm_sorted = G._phase_b_dst_comm_sorted[dst_id]
    if not src_sorted or not dst_comm_sorted:
        return False

    n_dst_comm = len(dst_comm_sorted)
    win = min(PHASE_B_COMM_WINDOW, n_dst_comm)
    n_src = len(src_sorted)
    dst_centroids = [c for c, _ in dst_comm_sorted]
    src_indices = list(range(n_src))

    while True:
        remaining = target_link_count - _links(G, pair)
        if remaining <= 0:
            break
        random.shuffle(src_indices)
        edges_per_src = max(1, min(math.ceil(remaining / n_src), win))
        made_progress = False

        for si in src_indices:
            if _links(G, pair) >= target_link_count:
                break
            theta_s, s = src_sorted[si]
            center = bisect.bisect_left(dst_centroids, theta_s)
            found = 0

            for delta in range(win):
                if found >= edges_per_src:
                    break
                for j in ((center + delta) % n_dst_comm,
                          (center + n_dst_comm - delta - 1) % n_dst_comm):
                    if found >= edges_per_src or _links(G, pair) >= target_link_count:
                        break
                    _, comm_id = dst_comm_sorted[j]
                    dst_nodes = G.communities_to_nodes.get((comm_id, dst_id), [])
                    if not dst_nodes:
                        continue
                    d = random.choice(dst_nodes)
                    if _create_edge(G, s, d, src_id, dst_id, target_link_count,
                                    reciprocity_p, int_p, ext_p):
                        found += 1
                        made_progress = True

        if not made_progress:
            break

    return _links(G, pair) >= target_link_count