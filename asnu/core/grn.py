"""Python reference implementation of the Rust `run_edge_creation` kernel.

Mirrors the Rust EdgeBuilder structure so the two implementations stay
verifiably parallel:

    _insert              <-> EdgeBuilder::insert
    _maybe_reciprocate   <-> EdgeBuilder::maybe_reciprocate
    _apply_transitivity  <-> EdgeBuilder::apply_transitivity
    _create_edge         <-> EdgeBuilder::create_edge
    establish_links      <-> the Phase A pair loop (one pair)
    establish_links_phase_b <-> the Phase B pair loop (one pair)

IMPORTANT — call ordering must match the Rust two-pass structure:

    for (src, dst, target) in group_pairs:
        establish_links(G, src, dst, target, ...)          # Phase A, ALL pairs
    for (src, dst, target) in group_pairs:
        establish_links_phase_b(G, src, dst, target, ...)  # Phase B, ALL pairs

Running Phase B per-pair immediately after its Phase A (the old behaviour)
gives Phase B a less complete graph than the Rust version sees.

`G.existing_num_links` is the single source of truth for budgets — there is
no separately mirrored `num_links` local, matching the Rust `link_counts`.

"Exactly the same" means identical logic and distributions; RNG draw
sequences cannot be bit-identical across languages.
"""

import bisect
import math
import random
from itertools import chain

PHASE_B_COMM_WINDOW = 200


# ── Budget helpers (single source of truth: G.existing_num_links) ──────────

def _links(G, pair):
    return G.existing_num_links.get(pair, 0)


def _max_links(G, pair):
    return G.maximum_num_links.get(pair, 0)


def _node_to_community(G):
    """node -> community id (FIRST tuple element; group ignored). Cached on G."""
    cache = getattr(G, "_node_to_community", None)
    if cache is None:
        cache = {}
        for (comm_id, _gid), nodes in G.communities_to_nodes.items():
            for n in nodes:
                cache[n] = comm_id
        G._node_to_community = cache
    return cache


def _undirected_neighbors(G, d):
    """Snapshot of d's UNDIRECTED neighbourhood (out ∪ in), deduplicated,
    out-neighbours first — matches the Rust adjacency/in_adjacency scan.
    Snapshot (list) because closures created during the scan must not
    extend it."""
    g = G.graph
    if g.is_directed():
        return list(dict.fromkeys(chain(g.successors(d), g.predecessors(d))))
    return list(g.neighbors(d))


# ── Edge creation primitives (mirror EdgeBuilder methods) ──────────────────

def _insert(G, s, d, pair):
    """Insert edge s→d attributed to `pair`. False if self-loop or exists."""
    if s == d or G.graph.has_edge(s, d):
        return False
    G.graph.add_edge(s, d)
    G.existing_num_links[pair] = _links(G, pair) + 1
    return True


def _maybe_reciprocate(G, s, d, src_group, dst_group, reciprocity_p):
    """Reciprocity roll for a fresh s(src_group) → d(dst_group) edge.

    NOTE: like the Rust, this checks only the reverse pair's maximum budget —
    there is deliberately NO `num_links >= target` guard for self-pairs, so a
    self-pair may overshoot its target by the reciprocal edge.
    """
    if random.random() >= reciprocity_p:
        return
    rev = (dst_group, src_group)
    if _links(G, rev) < _max_links(G, rev):
        _insert(G, d, s, rev)


def _apply_transitivity(G, s, d, src_group, budget_pair, target,
                        int_trans_p, ext_trans_p, reciprocity_p):
    """Triadic closures through pivot d.

    PER-NEIGHBOUR roll (not one gate for the whole scan): int_trans_p when n
    shares d's community id, ext_trans_p otherwise; two unknown communities
    are NOT treated as equal. Stops once `budget_pair` reaches `target`.
    Each closure gets its own reciprocity roll.
    """
    n2c = _node_to_community(G)
    d_comm = n2c.get(d)

    for n in _undirected_neighbors(G, d):
        if _links(G, budget_pair) >= target:
            break
        if n == s:
            continue

        n_comm = n2c.get(n)
        internal = n_comm is not None and d_comm is not None and n_comm == d_comm
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
    """Primary edge s→d plus reciprocity and transitivity follow-ons.
    Returns True iff the primary edge was created."""
    pair = (src_group, dst_group)
    if not _insert(G, s, d, pair):
        return False
    _maybe_reciprocate(G, s, d, src_group, dst_group, reciprocity_p)
    if int_trans_p > 0 or ext_trans_p > 0:
        _apply_transitivity(G, s, d, src_group, pair, target,
                            int_trans_p, ext_trans_p, reciprocity_p)
    return True


def _effective_transitivity(transitivity_p, internal_p, external_p):
    """Negative per-side values fall back to the scalar (matches Rust)."""
    int_p = transitivity_p if internal_p < 0 else internal_p
    ext_p = transitivity_p if external_p < 0 else external_p
    return int_p, ext_p


# ── Phase A: community-based edge creation (one group pair) ────────────────

def establish_links(G, src_id, dst_id,
                    target_link_count, fraction, reciprocity_p, transitivity_p,
                    valid_communities=None, pa_scope="local",
                    bridge_probability=0, number_of_communities=1,
                    internal_transitivity_p=-1.0, external_transitivity_p=-1.0):
    """Phase A for one (src_id, dst_id) pair. Run over ALL pairs before any
    Phase B call. Returns True iff the target was reached in Phase A.

    Communities are iterated sequentially (shuffled once per pair) so each
    community exhausts a proportional quota before moving on, concentrating
    edges within communities and raising transitivity.
    """
    int_p, ext_p = _effective_transitivity(
        transitivity_p, internal_transitivity_p, external_transitivity_p)

    pair = (src_id, dst_id)
    if _links(G, pair) >= target_link_count:
        return True
    if not valid_communities:
        return False

    # Deduplicate (list may carry duplicates for weighting) and shuffle.
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

            if community_id not in src_node_cache:
                src_node_cache[community_id] = G.communities_to_nodes.get(
                    (community_id, src_id), [])
            src_nodes = src_node_cache[community_id]
            if not src_nodes:
                continue

            # Bridge or normal dst community.
            if (bridge_probability > 0 and number_of_communities > 1
                    and random.random() < bridge_probability):
                direction = random.choice([-1, 1])
                dst_community = (community_id + direction) % number_of_communities
            else:
                dst_community = community_id

            # Popularity pool for (dst_community, dst_group): a random
            # `fraction`-sized sample; PA below grows it with repeats.
            pool_key = (dst_community, dst_id)
            if pool_key not in G.popularity_pool:
                pool = list(G.communities_to_nodes.get(pool_key, []))
                if pool:
                    sample_size = min(len(pool), math.ceil(len(pool) * fraction))
                    random.shuffle(pool)
                    del pool[sample_size:]
                G.popularity_pool[pool_key] = pool
            if not G.popularity_pool[pool_key]:
                continue

            # Create up to `quota` edges within this community.
            created = 0
            for _attempt in range(quota * 3):
                if created >= quota or _links(G, pair) >= target_link_count:
                    break
                s = random.choice(src_nodes)
                d = random.choice(G.popularity_pool[pool_key])

                if not _create_edge(G, s, d, src_id, dst_id, target_link_count,
                                    reciprocity_p, int_p, ext_p):
                    continue
                created += 1

                # Preferential attachment: occasionally re-add d to pools so
                # it gets picked more often.
                if fraction != 1.0 and random.random() > fraction:
                    if pa_scope == "global":
                        for comm_id in range(number_of_communities):
                            if random.random() < fraction / number_of_communities:
                                global_key = (comm_id, dst_id)
                                if global_key in G.popularity_pool:
                                    G.popularity_pool[global_key].append(d)
                    elif random.random() > fraction:
                        G.popularity_pool[pool_key].append(d)
                        pool_nodes = G.communities_to_nodes.get(pool_key, [])
                        if pool_nodes:
                            G.popularity_pool[pool_key].append(random.choice(pool_nodes))

    return _links(G, pair) >= target_link_count


# ── Phase B: spatial ring search (one group pair) ──────────────────────────

def establish_links_phase_b(G, src_id, dst_id, target_link_count,
                            reciprocity_p, transitivity_p,
                            internal_transitivity_p=-1.0,
                            external_transitivity_p=-1.0):
    """Phase B for one (src_id, dst_id) pair still under budget: find nearest
    dst communities by centroid and pick a random node from each — spreading
    degree load across nodes rather than targeting edge-nearest ones.

    Run only after Phase A has completed for ALL pairs (matches the Rust
    two-pass structure). Requires G.node_coordinates; no-op without it.
    Returns True iff the target was reached.
    """
    int_p, ext_p = _effective_transitivity(
        transitivity_p, internal_transitivity_p, external_transitivity_p)

    pair = (src_id, dst_id)
    if _links(G, pair) >= target_link_count:
        return True

    node_coordinates = getattr(G, "node_coordinates", None)
    if node_coordinates is None:
        return False

    # Sorted arrays, cached on G (same content as the Rust precompute).
    if not hasattr(G, "_phase_b_src_sorted"):
        G._phase_b_src_sorted = {}       # group_id -> sorted [(theta, node)]
        G._phase_b_dst_comm_sorted = {}  # group_id -> sorted [(centroid, comm)]

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