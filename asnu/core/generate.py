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
from asnu.core.grn import establish_links, establish_links_phase_b
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
    tuple
        (df_n_group_links, src_arr, dst_arr) — the raw link DataFrame plus the
        aggregated per-unique-pair group-ID arrays (one entry per key in
        G.maximum_num_links) for reuse in edge creation.
    """
    import pandas as _pd
    import numpy as _np

    df_n_group_links = read_file(links_path)
    df_n_group_links = df_n_group_links.sort_values(link_column, ascending=True)

    if verbose:
        print(f"Total input links: {df_n_group_links[link_column].sum()}")
        print("Calculating link requirements...")

    src_cols = [c for c in df_n_group_links.columns if c.endswith(src_suffix)]
    dst_cols = [c for c in df_n_group_links.columns if c.endswith(dst_suffix)]
    src_attr_names = [c[:-len(src_suffix)] for c in src_cols]
    dst_attr_names = [c[:-len(dst_suffix)] for c in dst_cols]

    # Build a lookup table from G.attrs_to_group (one row per group) and merge
    # it with the interaction df — no hashing, no collision risk.
    _lookup = _pd.DataFrame(
        [{**dict(k), '_gid': gid} for k, gid in G.attrs_to_group.items()]
    )

    def _merge_gids(cols, attr_names, suffix):
        rename = {c: c[:-len(suffix)] for c in cols}
        merged = df_n_group_links[cols].rename(columns=rename).merge(
            _lookup, on=attr_names, how='left')
        return merged['_gid']

    src_id_series = _merge_gids(src_cols, src_attr_names, src_suffix)
    dst_id_series = _merge_gids(dst_cols, dst_attr_names, dst_suffix)

    # Aggregate duplicate (src, dst) pairs by summing counts before allocating.
    # The CSV can have multiple rows for the same group pair (e.g. different
    # income sub-groups that map to the same group ID after enrichment);
    # dropping duplicates would lose those counts.
    mask = src_id_series.notna().values & dst_id_series.notna().values
    src_raw = src_id_series.values[mask].astype(_np.int64)
    dst_raw = dst_id_series.values[mask].astype(_np.int64)
    n_raw = df_n_group_links[link_column].values[mask].astype(_np.float64)

    _agg = (_pd.DataFrame({'s': src_raw, 'd': dst_raw, 'n': n_raw})
              .groupby(['s', 'd'], sort=False)['n'].sum())
    src_arr = _agg.index.get_level_values(0).values.astype(_np.int64)
    dst_arr = _agg.index.get_level_values(1).values.astype(_np.int64)
    n_arr = _agg.values.astype(_np.float64)

    # Stratified allocation of the scaled total.
    total_original = float(n_arr.sum())
    target_total = int(scale * total_original)
    alloc_arr = (n_arr * scale).astype(_np.int64)
    remainder = target_total - int(alloc_arr.sum())
    if remainder > 0:
        idx = _np.argsort(-n_arr)
        alloc_arr[idx[:remainder]] += 1

    G.maximum_num_links = dict(
        zip(zip(src_arr.tolist(), dst_arr.tolist()), alloc_arr.tolist()))

    if verbose:
        total_links = int(sum(G.maximum_num_links.values()))
        print(f"Total requested links: {total_links} (target: {target_total})")

    # Aggregated arrays match G.maximum_num_links keys exactly (no dups, no NaN),
    # so the backend never processes the same pair twice.
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
    group_desc_dict, characteristic_cols = desc_groups(pops_path, pop_column=pop_column)

    # Build (key, original_value) pairs for stratified allocation.
    items = [(gid, info[pop_column]) for gid, info in group_desc_dict.items()]
    node_allocations = stratified_allocate(items, scale)

    # Create nodes.
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

    # Create attribute -> group mapping.
    for group_id, attrs in G.group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        G.attrs_to_group[attrs_key] = group_id

    # Initialize link tracking.
    G.group_ids = list(G.group_to_attrs.keys())
    G.existing_num_links = defaultdict(int)


def _setup_no_community_structure(G):
    """
    Create a single synthetic community containing all nodes.

    This allows the edge creation code to work unchanged when no community
    structure is desired — all nodes are placed in community 0.
    """
    import numpy as np
    from scipy.sparse import csr_matrix as _csr

    n_groups = len(G.group_ids)
    G.number_of_communities = 1

    # Build probability matrix (normalized affinity) even without communities.
    if G.maximum_num_links:
        triples = [(i, j, v) for (i, j), v in G.maximum_num_links.items() if v > 0]
        rows_idx, cols_idx, vals = zip(*triples) if triples else ([], [], [])
    else:
        rows_idx, cols_idx, vals = [], [], []

    if vals:
        affinity_sp = _csr((list(vals), (list(rows_idx), list(cols_idx))),
                           shape=(n_groups, n_groups), dtype=float)
    else:
        affinity_sp = _csr((n_groups, n_groups), dtype=float)

    epsilon = 1e-5
    row_sums = np.asarray(affinity_sp.sum(axis=1)).flatten() + epsilon
    # Keep sparse if very large; todense() on a 70k×70k matrix would allocate 39 GB.
    if n_groups > 500:
        G.probability_matrix = np.zeros((0, 0))  # unused in 1-community edge creation
    else:
        G.probability_matrix = np.asarray(
            affinity_sp.multiply(1.0 / row_sums[:, np.newaxis]).todense())

    # Single community (0) contains all groups and all nodes.
    G.communities_to_groups[0] = list(G.group_ids)
    for group_id in G.group_ids:
        G.communities_to_nodes[(0, group_id)] = list(G.group_to_nodes[group_id])
    for node in G.graph.nodes:
        G.nodes_to_communities[node] = 0


def _build_group_pairs(G, links_path, src_suffix, dst_suffix,
                       _src_gids=None, _dst_gids=None):
    """
    Resolve the (src_group, dst_group) list to process.

    Prefers the aggregated arrays from _compute_maximum_num_links (no NaN, no
    duplicate pairs). Falls back to re-reading and merging the links file.
    Returns a list of (src_id, dst_id) integer tuples.
    """
    import pandas as _pd
    import numpy as _np

    if _src_gids is not None and _dst_gids is not None:
        return list(zip(_src_gids.astype(_np.int64).tolist(),
                        _dst_gids.astype(_np.int64).tolist()))

    df = read_file(links_path)
    src_cols = [c for c in df.columns if c.endswith(src_suffix)]
    dst_cols = [c for c in df.columns if c.endswith(dst_suffix)]
    src_attr = [c[:-len(src_suffix)] for c in src_cols]
    dst_attr = [c[:-len(dst_suffix)] for c in dst_cols]
    _lk = _pd.DataFrame([{**dict(k), '_gid': gid} for k, gid in G.attrs_to_group.items()])

    def _mgid(cols, attr, sfx):
        return (df[cols].rename(columns={c: c[:-len(sfx)] for c in cols})
                        .merge(_lk, on=attr, how='left')['_gid'])

    sg = _mgid(src_cols, src_attr, src_suffix)
    dg = _mgid(dst_cols, dst_attr, dst_suffix)
    _agg = _pd.DataFrame({'s': sg, 'd': dg}).dropna()
    return list(zip(_agg['s'].values.astype(_np.int64).tolist(),
                    _agg['d'].values.astype(_np.int64).tolist()))


def _run_edge_creation_python(G, links_path, fraction, reciprocity_p, transitivity_p,
                              verbose, src_suffix, dst_suffix, pa_scope,
                              bridge_probability=0, pre_seed_edges=None,
                              _links_df=None, _src_gids=None, _dst_gids=None,
                              internal_transitivity_p=-1.0, external_transitivity_p=-1.0):
    """
    Pure-Python fallback for edge creation.

    Mirrors the Rust two-pass structure: Phase A (community-based) runs over
    ALL group pairs first, then Phase B (spatial ring search) runs over all
    pairs still under budget. Running Phase B per-pair immediately after its
    own Phase A would give it a less complete graph than the Rust sees.
    """
    group_pairs = _build_group_pairs(G, links_path, src_suffix, dst_suffix,
                                     _src_gids, _dst_gids)
    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    # Keep only pairs whose endpoints actually have nodes, resolving each once.
    resolved = [
        (s, d, G.maximum_num_links.get((s, d), 0))
        for s, d in group_pairs
        if G.group_to_nodes.get(s) and G.group_to_nodes.get(d)
    ]
    total = len(resolved)

    def _valid_comms(s, d):
        if G.number_of_communities == 1:
            return [0]
        return group_pair_to_communities.get((s, d), [])

    # ── Phase A: community-based, all pairs ───────────────────────────────
    for idx, (src_id, dst_id, target) in enumerate(resolved):
        if verbose and ((idx + 1) % 500 == 0 or idx == 0 or idx == total - 1):
            print(f"\rPhase A: pair {idx + 1} of {total}", end="")
        establish_links(
            G, src_id, dst_id, target, fraction, reciprocity_p, transitivity_p,
            valid_communities=_valid_comms(src_id, dst_id), pa_scope=pa_scope,
            bridge_probability=bridge_probability,
            number_of_communities=G.number_of_communities,
            internal_transitivity_p=internal_transitivity_p,
            external_transitivity_p=external_transitivity_p,
        )
    if verbose:
        print()

    # ── Phase B: spatial ring search, all pairs still under budget ────────
    if getattr(G, 'node_coordinates', None) is not None:
        for idx, (src_id, dst_id, target) in enumerate(resolved):
            if verbose and ((idx + 1) % 500 == 0):
                print(f"\rPhase B: pair {idx + 1} of {total}", end="")
            establish_links_phase_b(
                G, src_id, dst_id, target, reciprocity_p, transitivity_p,
                internal_transitivity_p=internal_transitivity_p,
                external_transitivity_p=external_transitivity_p,
            )
        if verbose:
            print()

    # ── Overfulfilment warnings ───────────────────────────────────────────
    if verbose:
        warnings = [
            f"Groups ({s})-({d}): {G.existing_num_links.get((s, d), 0)} exceeds target {t}"
            for s, d, t in resolved
            if G.existing_num_links.get((s, d), 0) > t
        ]
        if warnings:
            print(f"\nWarnings ({len(warnings)} group pairs):")
            for warning in warnings[:10]:
                print(f"  {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")


def _run_edge_creation(G, links_path, fraction, reciprocity_p, transitivity_p,
                       verbose, src_suffix, dst_suffix, pa_scope,
                       bridge_probability=0, pre_seed_edges=None,
                       _links_df=None, _src_gids=None, _dst_gids=None,
                       internal_transitivity_p=-1.0, external_transitivity_p=-1.0):
    """
    Run the edge creation loop using the community structure already set on G.
    Tries the Rust backend, falls back to pure Python.

    internal_* / external_* control per-side (same-community vs cross-community)
    transitivity. Defaults preserve the previous behaviour exactly
    (transitivity inherits transitivity_p).
    """
    try:
        import x as d
        from asnu_rust import run_edge_creation as rust_edge_creation
    except ImportError:
        if verbose:
            print("Rust backend unavailable — using Python fallback for edge creation...")
        _run_edge_creation_python(
            G, links_path, fraction, reciprocity_p, transitivity_p, verbose,
            src_suffix, dst_suffix, pa_scope,
            bridge_probability=bridge_probability, pre_seed_edges=pre_seed_edges,
            _links_df=_links_df, _src_gids=_src_gids, _dst_gids=_dst_gids,
            internal_transitivity_p=internal_transitivity_p,
            external_transitivity_p=external_transitivity_p)
        return

    if verbose:
        print("Using Rust backend for edge creation...")

    group_pairs_ids = _build_group_pairs(G, links_path, src_suffix, dst_suffix,
                                         _src_gids, _dst_gids)
    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    group_pairs = [
        (s, d, G.maximum_num_links.get((s, d), 0))
        for s, d in group_pairs_ids
        if G.group_to_nodes.get(s) and G.group_to_nodes.get(d)
    ]

    ctn = {(int(k[0]), int(k[1])): [int(n) for n in v]
           for k, v in G.communities_to_nodes.items()}
    ntg = {int(k): int(v) for k, v in G.nodes_to_group.items()}
    mnl = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}

    if G.number_of_communities == 1:
        vcm = {(int(s), int(d)): [0] for s, d, _ in group_pairs}
    else:
        vcm = {(int(k[0]), int(k[1])): [int(c) for c in v]
               for k, v in group_pair_to_communities.items()}

    rust_pre_edges = ([(int(u), int(v)) for u, v in pre_seed_edges]
                      if pre_seed_edges else None)
    node_coords = getattr(G, 'node_coordinates', None)

    new_edges, link_counts = rust_edge_creation(
        group_pairs, vcm, mnl, ctn, ntg,
        fraction, reciprocity_p, transitivity_p,
        pa_scope, G.number_of_communities,
        bridge_probability,
        rust_pre_edges,
        node_coords,
        float(internal_transitivity_p),
        float(external_transitivity_p),
    )

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
             pre_seed_edges=None,
             internal_transitivity=-1.0, external_transitivity=-1.0):
    """
    Generate a population-based network using NetworkX.

    New params (Phase B internal/external control; defaults = previous behaviour):
        internal_transitivity, external_transitivity : float
            Triangle-closing probability for same-community (internal) vs
            different-community (external) closures. -1.0 → inherit `transitivity`.
    """
    if verbose:
        print("=" * 60)
        print("NETWORK GENERATION")
        print("=" * 60)
        print("\nStep 1: Creating nodes from population data...")

    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    G = NetworkXGraph(base_path)
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes")
        print("\nStep 2: Creating edges from interaction patterns...")

    _links_df, _src_gids, _dst_gids = _compute_maximum_num_links(
        G, links_path, scale, src_suffix=src_suffix,
        dst_suffix=dst_suffix, link_column=link_column, verbose=verbose)

    preferential_attachment_fraction = 1 - preferential_attachment

    # Bundle the per-side knobs once so both call sites stay identical.
    _edge_kwargs = dict(
        internal_transitivity_p=internal_transitivity,
        external_transitivity_p=external_transitivity,
    )

    if community_file is not None:
        if verbose:
            print("\nStep 2a: Loading communities from file...")
        load_communities(G, community_file)
        if verbose:
            print(f"  Loaded {G.number_of_communities} communities from {community_file}")

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
                               _links_df=_links_df, _src_gids=_src_gids, _dst_gids=_dst_gids,
                               **_edge_kwargs)

            if fill_unfulfilled:
                if verbose:
                    print("\nStep 3: Filling remaining unfulfilled group pairs...")
                fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    else:
        if verbose:
            print("\nNo community file given.")
            print("  Generating edges without community structure...")

        _setup_no_community_structure(G)

        _run_edge_creation(G, links_path, preferential_attachment_fraction,
                           reciprocity, transitivity, verbose,
                           src_suffix, dst_suffix, pa_scope,
                           bridge_probability=bridge_probability,
                           _links_df=_links_df, _src_gids=_src_gids, _dst_gids=_dst_gids,
                           **_edge_kwargs)

        if fill_unfulfilled:
            if verbose:
                print("\nStep 3: Filling remaining unfulfilled group pairs...")
            fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    G.finalize()

    if verbose:
        total_requested = sum(G.maximum_num_links.values())
        total_created = G.graph.number_of_edges()
        fulfillment_rate = (total_created / total_requested * 100) if total_requested > 0 else 0
        overfulfilled = sum(1 for (src, dst) in G.maximum_num_links.keys()
                            if G.existing_num_links.get((src, dst), 0) > G.maximum_num_links[(src, dst)])
        print(f"\n{'=' * 60}")
        print(f"NETWORK GENERATION COMPLETE")
        print(f"{'=' * 60}")
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
        print(f"{'=' * 60}\n")

    return G