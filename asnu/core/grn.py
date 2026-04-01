import numpy as np
import math
import random

def establish_links(G, src_id, dst_id,
                  target_link_count, fraction, reciprocity_p, transitivity_p,
                  valid_communities=None, pa_scope="local",
                  bridge_probability=0, number_of_communities=1):
    """
    Create edges between source and destination nodes with preferential attachment.

    Connects nodes from source and destination groups using a bounded preferential
    attachment model. Supports reciprocity and transitivity for realistic network
    structure.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with network data and metadata
    src_id : int
        Source group ID
    dst_id : int
        Destination group ID
    target_link_count : int
        Target number of edges to create
    fraction : float
        Preferential attachment parameter (0-1)
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)
    transitivity_p : float
        Probability of creating transitive edges (0-1)
    valid_communities : list, optional
        Communities shared by both groups (precomputed for efficiency)
    pa_scope : str, optional
        Scope of preferential attachment popularity:
        - "local": popularity stays within the community (default)
        - "global": popularity spreads to all communities
    bridge_probability : float, optional
        Probability of routing an edge to a neighboring community (±1,
        circular wrapping). Creates wide bridges between adjacent
        communities. Default 0 (no bridging).
    number_of_communities : int, optional
        Total number of communities (needed for circular wrapping).

    Returns
    -------
    bool
        True if target was met, False if exceeded
    """
    link_n_check = True
    attempts = 0
    max_attempts = target_link_count * 10

    # Get current link count for this group pair
    num_links = G.existing_num_links.get((src_id, dst_id), 0)

    # Check if already over target
    if num_links > target_link_count:
        link_n_check = False

    # Use precomputed communities for this group pair
    possible_communities = valid_communities

    if not possible_communities:
        return link_n_check

    # Cache for source node lists by community (created as needed)
    src_node_lists = {}

    # Preselect communities in batches for efficiency
    batch_size = 10000
    community_batch = np.random.choice(possible_communities, size=batch_size, replace=True)
    batch_idx = 0

    # Create edges until we reach the target
    while num_links < target_link_count and attempts < max_attempts:

        # Get next community from the batch
        community_id = community_batch[batch_idx]
        batch_idx += 1

        # Refill batch when exhausted
        if batch_idx >= batch_size:
            community_batch = np.random.choice(possible_communities, size=batch_size, replace=True)
            batch_idx = 0

        # Initialize node lists for this community on first use
        if community_id not in src_node_lists:
            # Get source nodes in this community
            src_node_lists[community_id] = G.communities_to_nodes[(community_id, src_id)]

        # Decide: bridge edge (dst from neighboring community) or normal edge
        if (bridge_probability > 0 and number_of_communities > 1
                and random.random() < bridge_probability):
            direction = random.choice([-1, 1])
            dst_community = (community_id + direction) % number_of_communities
        else:
            dst_community = community_id

        # Initialize global popularity pool for this community-group pair if needed
        pool_key = (dst_community, dst_id)
        if pool_key not in G.popularity_pool:
            dst_community_nodes = G.communities_to_nodes.get((dst_community, dst_id), [])
            if dst_community_nodes:
                sample_size = math.ceil(len(dst_community_nodes) * fraction)
                G.popularity_pool[pool_key] = list(np.random.choice(dst_community_nodes,
                                                                     size=sample_size,
                                                                     replace=False))
            else:
                G.popularity_pool[pool_key] = []

        # Skip if no destination nodes available
        if not G.popularity_pool[pool_key]:
            attempts += 1
            continue

        # Select random source and destination nodes from this community
        s = random.choice(src_node_lists[community_id])
        d_from_db = random.choice(G.popularity_pool[pool_key])

        # Add edge if valid (no self-loops, no duplicates)
        if s != d_from_db and not G.graph.has_edge(s, d_from_db):
            G.graph.add_edge(s, d_from_db)
            num_links += 1
            G.existing_num_links[(src_id, dst_id)] = num_links

            # Reciprocity
            if random.uniform(0,1) < reciprocity_p:
                if G.existing_num_links.get((dst_id, src_id), 0) < G.maximum_num_links.get((dst_id, src_id), 0) and not G.graph.has_edge(d_from_db, s):
                    G.graph.add_edge(d_from_db, s)
                    G.existing_num_links[(dst_id, src_id)] += 1
                    if (dst_id == src_id):
                        num_links += 1
                        G.existing_num_links[(src_id, dst_id)] = num_links

            # Preferential attachment: add popular nodes back to the pool
            if random.uniform(0,1) > fraction and fraction != 1:
                if pa_scope == "global":
                    # Spread popularity across communities, but scale probability
                    # to avoid N-fold amplification (where N = number of communities)
                    # Each community gets the node with probability 1/N, so expected
                    # total additions ≈ 1 (same as local mode)
                    for comm_id in range(G.number_of_communities):
                        if random.uniform(0, 1) < ((1.0 / G.number_of_communities) * (fraction)):
                            global_key = (comm_id, dst_id)
                            if global_key in G.popularity_pool:
                                G.popularity_pool[global_key].append(d_from_db)
                else:
                    # Local: only add to current community's pool
                    if random.uniform(0,1) > fraction:
                        G.popularity_pool[pool_key].append(d_from_db)
                        dst_random_community_node = np.random.choice(dst_community_nodes)
                        G.popularity_pool[pool_key].append(dst_random_community_node)

            # Add edges to neighbors (clustering effect)
            if transitivity_p < random.uniform(0,1):
                continue
            for n in G.graph.neighbors(d_from_db):
                if s == n:
                    continue
                n_id = G.nodes_to_group[n]
                if (src_id, n_id) in G.maximum_num_links:
                    if G.existing_num_links[(src_id, n_id)] < G.maximum_num_links[(src_id, n_id)]:
                        if not G.graph.has_edge(s, n):
                            G.graph.add_edge(s, n)
                            G.existing_num_links[(src_id, n_id)] += 1
                            # Also count toward main target if same destination group
                            if n_id == dst_id:
                                num_links += 1
                                G.existing_num_links[(src_id, dst_id)] = num_links
                            # Reciprocity
                            if random.uniform(0,1) < reciprocity_p:
                                if not G.graph.has_edge(n, s) and G.existing_num_links[(n_id, src_id)] < G.maximum_num_links[(n_id, src_id)]:
                                    G.graph.add_edge(n, s)
                                    G.existing_num_links[(n_id, src_id)] += 1
                                    if (n_id == src_id) & (src_id == dst_id):
                                        num_links += 1
                                        G.existing_num_links[(src_id, dst_id)] = num_links

        attempts += 1

    return link_n_check
