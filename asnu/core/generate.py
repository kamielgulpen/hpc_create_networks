"""
Network generation module for PyPleNet NetworkX.

This module provides functions to generate large-scale population networks
using NetworkX in-memory graph storage. It creates nodes from population data and
establishes edges based on interaction patterns, with support for scaling,
reciprocity, and preferential attachment.

This NetworkX-based implementation is significantly faster than file-based
approaches for graphs that fit in memory.

Functions
---------
init_nodes : Initialize nodes in the graph from population data
init_links : Initialize edges in the graph from interaction data
generate : Main function to generate a complete network

Examples
--------
>>> graph = generate('population.csv', 'interactions.xlsx',
...                  fraction=0.4, scale=0.1, reciprocity_p=0.2)
>>> print(f"Generated network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
"""
import os
import shutil
from collections import defaultdict

from asnu.core.utils import (find_nodes, read_file, desc_groups, stratified_allocate)
from asnu.core.grn import establish_links
from asnu.core.graph import NetworkXGraph
from asnu.core.community import (
    build_group_pair_to_communities_lookup,
    connect_all_within_communities,
    fill_unfulfilled_group_pairs,
    load_communities
)


def _compute_maximum_num_links(G, links_path, scale, src_suffix='_src',
                                dst_suffix='_dst', link_column='n', verbose=True):
    """
    Compute maximum link counts for all group pairs using stratified allocation.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with group_ids and group metadata already initialized
    links_path : str
        Path to interactions file (CSV or Excel)
    scale : float
        Population scaling factor
    src_suffix : str, optional
        Suffix for source group columns (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns (default '_dst')
    link_column : str, optional
        Name of column containing link counts (default 'n')
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    pandas.DataFrame
        The raw link data DataFrame for downstream use
    """
    import pandas as _pd
    import numpy as _np

    df_n_group_links = read_file(links_path)
    df_n_group_links = df_n_group_links.sort_values('n', ascending= True)

    print(df_n_group_links.n.sum())

    if verbose:
        print("Calculating link requirements...")

    G.maximum_num_links = {}

    src_cols = [c for c in df_n_group_links.columns if c.endswith(src_suffix)]
    dst_cols = [c for c in df_n_group_links.columns if c.endswith(dst_suffix)]
    src_attr_names = [c[:-len(src_suffix)] for c in src_cols]

    # Build a lookup table from G.attrs_to_group (one row per group)
    # and merge it with the interaction df — no hashing, no collision risk.
    dst_attr_names = [c[:-len(dst_suffix)] for c in dst_cols]
    _lookup = _pd.DataFrame(
        [{**dict(k), '_gid': gid} for k, gid in G.attrs_to_group.items()]
    )

    def _merge_gids(df, cols, attr_names, suffix):
        rename = {c: c[:-len(suffix)] for c in cols}
        merged = df[cols].rename(columns=rename).merge(_lookup, on=attr_names, how='left')
        return merged['_gid']

    src_id_series = _merge_gids(df_n_group_links, src_cols, src_attr_names, src_suffix)
    dst_id_series = _merge_gids(df_n_group_links, dst_cols, dst_attr_names, dst_suffix)

    # Aggregate duplicate (src, dst) pairs by summing counts before allocating.
    # The CSV can have multiple rows for the same group pair (e.g. different income
    # sub-groups that map to the same group ID after enrichment); dropping duplicates
    # would lose those counts.
    import numpy as _np
    mask_np = src_id_series.notna().values & dst_id_series.notna().values
    _src_raw = src_id_series.values[mask_np].astype(_np.int64)
    _dst_raw = dst_id_series.values[mask_np].astype(_np.int64)
    _n_raw   = df_n_group_links[link_column].values[mask_np].astype(_np.float64)

    # Sum counts per unique (src, dst) pair via pandas groupby
    _agg = _pd.DataFrame({'s': _src_raw, 'd': _dst_raw, 'n': _n_raw}) \
              .groupby(['s', 'd'], sort=False)['n'].sum()
    src_arr = _agg.index.get_level_values(0).values.astype(_np.int64)
    dst_arr = _agg.index.get_level_values(1).values.astype(_np.int64)
    n_arr   = _agg.values.astype(_np.float64)

    # Stratified allocation
    total_original = float(n_arr.sum())
    target_total   = int(scale * total_original)
    alloc_arr      = (n_arr * scale).astype(_np.int64)
    remainder      = target_total - int(alloc_arr.sum())
    if remainder > 0:
        idx = _np.argsort(-n_arr)
        alloc_arr[idx[:remainder]] += 1

    G.maximum_num_links = dict(zip(zip(src_arr.tolist(), dst_arr.tolist()), alloc_arr.tolist()))

    if verbose:
        total_links = int(sum(G.maximum_num_links.values()))
        print(f"Total requested links: {total_links} (target: {target_total})")

    # Return df + aggregated group ID arrays (one entry per unique pair, matching
    # G.maximum_num_links keys exactly) for reuse in edge creation.
    # Using the aggregated arrays avoids duplicate (s,d) entries that would cause
    # the Rust backend to process the same pair multiple times.
    return df_n_group_links, src_arr.astype(float), dst_arr.astype(float)


def init_nodes(G, pops_path, scale=1, pop_column='n'):
    """
    Initialize nodes from population data using stratified sampling.

    Uses stratified allocation to preserve demographic proportions while scaling.

    Parameters
    ----------
    G : NetworkXGraph
        Wrapper with G.graph (nx.DiGraph) and metadata
    pops_path : str
        Path to population file (CSV or Excel)
    scale : float, optional
        Population scaling factor (default 1)
    pop_column : str, optional
        Name of the column containing population counts (default 'n')
    """
    print(G.group_to_attrs.keys())
    group_desc_dict, characteristic_cols = desc_groups(pops_path, pop_column=pop_column)

    # Build (key, original_value) pairs for stratified allocation
    items = [(gid, info[pop_column]) for gid, info in group_desc_dict.items()]
    node_allocations = stratified_allocate(items, scale)
    # Create nodes
    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        G.group_to_attrs[group_id] = attrs
        n_nodes = node_allocations[group_id]
        G.group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))

        for _ in range(n_nodes):
            G.graph.add_node(node_id, **attrs)
            G.nodes_to_group[node_id] = group_id
            node_id += 1

    # Create attribute to group mapping
    for group_id, attrs in G.group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        G.attrs_to_group[attrs_key] = group_id

    # Initialize link tracking
    group_ids = list(G.group_to_attrs.keys())

    G.group_ids = group_ids
    G.existing_num_links = defaultdict(int)

def _setup_no_community_structure(G):
    """
    Create a single synthetic community containing all nodes.

    This allows the edge creation code to work unchanged when no community
    structure is desired — all nodes are placed in community 0.
    """
    import numpy as np

    n_groups = len(G.group_ids)
    G.number_of_communities = 1

    # Build probability matrix (normalized affinity) even without communities
    if G.maximum_num_links:
        rows_idx, cols_idx, vals = zip(*[(i, j, v) for (i, j), v in G.maximum_num_links.items() if v > 0]) or ([], [], [])
    else:
        rows_idx, cols_idx, vals = [], [], []
    from scipy.sparse import csr_matrix as _csr
    affinity_sp = _csr((list(vals), (list(rows_idx), list(cols_idx))), shape=(n_groups, n_groups), dtype=float) if vals else _csr((n_groups, n_groups), dtype=float)
    epsilon = 1e-5
    row_sums = np.asarray(affinity_sp.sum(axis=1)).flatten() + epsilon
    # Keep sparse if very large; todense() on a 70k×70k matrix would allocate 39 GB
    if n_groups > 500:
        G.probability_matrix = np.zeros((0, 0))  # not used in 1-community edge creation
    else:
        G.probability_matrix = np.asarray(affinity_sp.multiply(1.0 / row_sums[:, np.newaxis]).todense())

    # Single community (0) contains all groups and all nodes
    G.communities_to_groups[0] = list(G.group_ids)
    for group_id in G.group_ids:
        G.communities_to_nodes[(0, group_id)] = list(G.group_to_nodes[group_id])
    for node in G.graph.nodes:
        G.nodes_to_communities[node] = 0


def _run_edge_creation_python(G, links_path, fraction, reciprocity_p, transitivity_p,
                              verbose, src_suffix, dst_suffix, pa_scope,
                              bridge_probability=0, pre_seed_edges=None,
                              _links_df=None, _src_gids=None, _dst_gids=None):
    """Pure-Python fallback for edge creation."""
    import pandas as _pd

    warnings = []

    if _links_df is not None and _src_gids is not None and _dst_gids is not None:
        # Aggregated arrays from _compute_maximum_num_links — no NaN, no dups
        src_gid_list = _src_gids.astype(int).tolist()
        dst_gid_list = _dst_gids.astype(int).tolist()
    else:
        df_n_group_links = read_file(links_path)
        # df_n_group_links = df_n_group_links.sort_values('n', ascending= True)
        src_cols = [c for c in df_n_group_links.columns if c.endswith(src_suffix)]
        dst_cols = [c for c in df_n_group_links.columns if c.endswith(dst_suffix)]
        src_attr_names = [c[:-len(src_suffix)] for c in src_cols]
        dst_attr_names = [c[:-len(dst_suffix)] for c in dst_cols]
        _lk = _pd.DataFrame([{**dict(k), '_gid': gid} for k, gid in G.attrs_to_group.items()])
        def _mgid(df, cols, attr, sfx):
            return df[cols].rename(columns={c: c[:-len(sfx)] for c in cols}).merge(_lk, on=attr, how='left')['_gid']
        src_gid_list = _mgid(df_n_group_links, src_cols, src_attr_names, src_suffix).tolist()
        dst_gid_list = _mgid(df_n_group_links, dst_cols, dst_attr_names, dst_suffix).tolist()

    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    # Pre-compute (src_nodes, src_id, dst_nodes, dst_id) for every row
    row_lookups = [
        (
            (G.group_to_nodes[sg], sg) if sg is not None and not _pd.isna(sg) else None,
            (G.group_to_nodes[dg], dg) if dg is not None and not _pd.isna(dg) else None,
        )
        for sg, dg in zip(src_gid_list, dst_gid_list)
    ]

    total_rows = len(df_n_group_links)
    for idx, (src_lookup, dst_lookup) in enumerate(row_lookups):
        if verbose and ((idx + 1) % 500 == 0 or idx == 0 or idx == total_rows - 1):
            print(f"\rProcessing row {idx + 1} of {total_rows}", end="")

        if src_lookup is None or dst_lookup is None:
            continue
        src_nodes, src_id = src_lookup
        dst_nodes, dst_id = dst_lookup

        num_requested_links = G.maximum_num_links.get((src_id, dst_id), 0)

        if not src_nodes or not dst_nodes:
            continue

        valid_communities = group_pair_to_communities.get((src_id, dst_id), [])

        link_success = establish_links(G, src_id, dst_id,
                                       num_requested_links, fraction, reciprocity_p,
                                       transitivity_p, valid_communities, pa_scope,
                                       bridge_probability=bridge_probability,
                                       number_of_communities=G.number_of_communities)

        if not link_success:
            existing_links = G.existing_num_links.get((src_id, dst_id), 0)
            warnings.append(f"Groups ({src_id})-({dst_id}): {existing_links} exceeds target {num_requested_links}")

    if verbose:
        print()
        if warnings:
            print(f"\nWarnings ({len(warnings)} group pairs):")
            for warning in warnings[:10]:
                print(f"  {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")


def _run_edge_creation(G, links_path, fraction, reciprocity_p, transitivity_p,
                       verbose, src_suffix, dst_suffix, pa_scope,
                       bridge_probability=0, pre_seed_edges=None,
                       _links_df=None, _src_gids=None, _dst_gids=None):
    """
    Run the edge creation loop using the community structure already set on G.
    Tries Rust backend, falls back to Python.
    _links_df, _src_gids, _dst_gids: precomputed from _compute_maximum_num_links to avoid re-reading.
    """
    try:
        from asnu_rust import run_edge_creation as rust_edge_creation
    except ImportError:
        _run_edge_creation_python(G, links_path, fraction, reciprocity_p,
                                  transitivity_p, verbose, src_suffix, dst_suffix, pa_scope,
                                  bridge_probability=bridge_probability,
                                  pre_seed_edges=pre_seed_edges,
                                  _links_df=_links_df, _src_gids=_src_gids, _dst_gids=_dst_gids)
        return

    if verbose:
        print("Using Rust backend for edge creation...")

    import pandas as _pd
    import numpy as _np

    # _src_gids/_dst_gids are the aggregated arrays from _compute_maximum_num_links,
    # so they already correspond 1-to-1 with G.maximum_num_links keys (no NaN, no dups).
    if _links_df is not None and _src_gids is not None and _dst_gids is not None:
        sg_arr = _src_gids.astype(_np.int64)
        dg_arr = _dst_gids.astype(_np.int64)
    else:
        # Fallback: recompute from file using the same merge approach
        df_n_group_links = read_file(links_path)
        # df_n_group_links = df_n_group_links.sort_values("n", ascending=True)
        _src_cols = [c for c in df_n_group_links.columns if c.endswith(src_suffix)]
        _dst_cols = [c for c in df_n_group_links.columns if c.endswith(dst_suffix)]
        _src_attr = [c[:-len(src_suffix)] for c in _src_cols]
        _dst_attr = [c[:-len(dst_suffix)] for c in _dst_cols]
        _lk = _pd.DataFrame([{**dict(k), '_gid': gid} for k, gid in G.attrs_to_group.items()])
        def _mgid(df, cols, attr, sfx):
            return df[cols].rename(columns={c: c[:-len(sfx)] for c in cols}).merge(_lk, on=attr, how='left')['_gid']
        sg_s = _mgid(df_n_group_links, _src_cols, _src_attr, src_suffix)
        dg_s = _mgid(df_n_group_links, _dst_cols, _dst_attr, dst_suffix)
        # Aggregate duplicates (same as _compute_maximum_num_links)
        _agg = _pd.DataFrame({'s': sg_s, 'd': dg_s}).dropna()
        sg_arr = _agg['s'].values.astype(_np.int64)
        dg_arr = _agg['d'].values.astype(_np.int64)

    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    group_pairs = [
        (s, d, G.maximum_num_links.get((s, d), 0))
        for s, d in zip(sg_arr.tolist(), dg_arr.tolist())
        if G.group_to_nodes.get(s) and G.group_to_nodes.get(d)
    ]

    # Convert G data to plain dicts for Rust
    ctn = {(int(k[0]), int(k[1])): [int(n) for n in v]
           for k, v in G.communities_to_nodes.items()}
    ntg = {int(k): int(v) for k, v in G.nodes_to_group.items()}
    mnl = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}
    # For single community the lazy defaultdict has no .items(); build vcm from group_pairs
    if G.number_of_communities == 1:
        vcm = {(int(s), int(d)): [0] for s, d, _ in group_pairs}
    else:
        vcm = {(int(k[0]), int(k[1])): [int(c) for c in v]
               for k, v in group_pair_to_communities.items()}

    # Convert pre_seed_edges for Rust (list of (int, int) tuples or None)
    rust_pre_edges = None
    if pre_seed_edges:
        rust_pre_edges = [(int(u), int(v)) for u, v in pre_seed_edges]

    node_coords = getattr(G, 'node_coordinates', None)
    new_edges, link_counts = rust_edge_creation(
        group_pairs, vcm, mnl, ctn, ntg,
        fraction, reciprocity_p, transitivity_p,
        pa_scope, G.number_of_communities,
        bridge_probability,
        rust_pre_edges,
        node_coords,
    )

    # Apply edges to the NetworkX graph
    G.graph.add_edges_from(new_edges)
    for src, dst, count in link_counts:
        G.existing_num_links[(src, dst)] = count

    if verbose:
        print(f"\n  Created {len(new_edges)} edges")


def generate(pops_path, links_path, preferential_attachment, scale, reciprocity,
             transitivity, base_path="graph_data", verbose=True,
             pop_column='n', src_suffix='_src', dst_suffix='_dst', link_column='n',
             fill_unfulfilled=True, fully_connect_communities=False,
             pa_scope='local', community_file=None, bridge_probability=0,
             pre_seed_edges=None):
    """
    Generate a population-based network using NetworkX.

    Creates a network by first generating nodes from population data, then
    establishing edges based on interaction patterns. Supports preferential
    attachment, reciprocity, transitivity, and community structure.

    Community structure is provided via a pre-computed JSON file created by
    create_communities(). If no community file is given, edges are created
    without any community structure.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    preferential_attachment : float
        Preferential attachment strength (0-1)
    scale : float
        Population scaling factor
    reciprocity : float
        Probability of reciprocal edges (0-1)
    transitivity : float
        Probability of transitive edges (0-1)
    base_path : str, optional
        Directory for saving graph (default "graph_data")
    verbose : bool, optional
        Whether to print progress information
    pop_column : str, optional
        Column name for population counts in pops_path (default 'n')
    src_suffix : str, optional
        Suffix for source group columns in links_path (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns in links_path (default '_dst')
    link_column : str, optional
        Column name for link counts in links_path (default 'n')
    fill_unfulfilled : bool, optional
        Whether to fill unfulfilled group pairs after initial link creation (default True)
    fully_connect_communities : bool, optional
        Whether to fully connect all nodes within each community, bypassing normal
        link formation process (default False). Requires community_file.
    pa_scope : str, optional
        Scope of preferential attachment popularity (default 'local'):
        - 'local': popularity stays within the community (intra-community)
        - 'global': popularity spreads across all communities (inter-community)
    community_file : str, optional
        Path to a JSON file with pre-computed community assignments (default None).
        Created by create_communities(). If not provided, the network is generated
        without any community structure.
    bridge_probability : float, optional
        Probability of routing an edge to a neighboring community (±1, circular
        wrapping). Creates wide bridges between adjacent communities, modeling
        people participating in multiple foci. Default 0 (no bridging).

    Returns
    -------
    NetworkXGraph
        Generated network with graph data and metadata
    """
    if verbose:
        print("="*60)
        print("NETWORK GENERATION")
        print("="*60)
        print("\nStep 1: Creating nodes from population data...")

    # Prepare output directory
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    G = NetworkXGraph(base_path)

    # Create nodes with stratified allocation
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes")
        print("\nStep 2: Creating edges from interaction patterns...")

    # Compute maximum link targets; capture precomputed df + group ID arrays to avoid re-reading
    _links_df, _src_gids, _dst_gids = _compute_maximum_num_links(
        G, links_path, scale, src_suffix=src_suffix,
        dst_suffix=dst_suffix, link_column=link_column, verbose=verbose)

    # Invert preferential attachment for internal representation
    preferential_attachment_fraction = 1 - preferential_attachment

    if community_file is not None:
        # --- Load pre-computed communities from file ---
        if verbose:
            print("\nStep 2a: Loading communities from file...")

        load_communities(G, community_file)

        if verbose:
            print(f"  Loaded {G.number_of_communities} communities from {community_file}")

        # Pre-seed edges into the graph (for multiplex hierarchical generation)
        if pre_seed_edges:
            G.graph.add_edges_from(pre_seed_edges)
            for u, v in pre_seed_edges:
                src_g = G.nodes_to_group[u]
                dst_g = G.nodes_to_group[v]
                G.existing_num_links[(src_g, dst_g)] += 1
            if verbose:
                print(f"  Pre-seeded {len(pre_seed_edges)} edges into graph")

        if fully_connect_communities:
            if verbose:
                print("\nStep 2b: Fully connecting nodes within communities...")
            connect_all_within_communities(G, verbose=verbose)
        else:
            if verbose:
                print("\nStep 2b: Creating edges using community structure...")
            _run_edge_creation(G, links_path, preferential_attachment_fraction,
                               reciprocity, transitivity, verbose,
                               src_suffix, dst_suffix, pa_scope,
                               bridge_probability=bridge_probability,
                               pre_seed_edges=pre_seed_edges,
                               _links_df=_links_df, _src_gids=_src_gids, _dst_gids=_dst_gids)

            if fill_unfulfilled:
                if verbose:
                    print("\nStep 3: Filling remaining unfulfilled group pairs...")
                fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    else:
        # --- No community structure ---
        if verbose:
            print("\nNo community file given.")
            print("  Generating edges without community structure...")

        _setup_no_community_structure(G)

        _run_edge_creation(G, links_path, preferential_attachment_fraction,
                           reciprocity, transitivity, verbose,
                           src_suffix, dst_suffix, pa_scope,
                           bridge_probability=bridge_probability,
                           _links_df=_links_df, _src_gids=_src_gids, _dst_gids=_dst_gids)

        if fill_unfulfilled:
            if verbose:
                print("\nStep 3: Filling remaining unfulfilled group pairs...")
            fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    # Save to disk
    G.finalize()

    if verbose:
        # Calculate link fulfillment statistics
        total_requested = sum(G.maximum_num_links.values())
        total_created = G.graph.number_of_edges()
        fulfillment_rate = (total_created / total_requested * 100) if total_requested > 0 else 0

        # Count overfulfilled pairs
        overfulfilled = sum(1 for (src, dst) in G.maximum_num_links.keys()
                           if G.existing_num_links.get((src, dst), 0) > G.maximum_num_links[(src, dst)])

        print(f"\n{'='*60}")
        print(f"NETWORK GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Nodes: {G.graph.number_of_nodes()}")
        print(f"Edges: {G.graph.number_of_edges()}")
        print(f"\nLink Fulfillment:")
        print(f"  Requested: {total_requested}")
        print(f"  Created: {total_created}")
        print(f"  Difference: {total_created - total_requested:+d}")
        print(f"  Rate: {fulfillment_rate:.1f}%")
        if overfulfilled > 0:
            print(f"  Overfulfilled pairs: {overfulfilled}")
        print(f"\nSaved to: {base_path}")
        print(f"{'='*60}\n")

    return G