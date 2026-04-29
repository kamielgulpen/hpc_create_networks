""" This module contains utility functions used during graph generation. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def stratified_allocate(items, scale):
    """
    Allocate integer counts from fractional scaled values, preserving the total.

    Each item gets floor(scale * original), then the remainder is distributed
    round-robin to the largest items to maintain the exact scaled total.

    Parameters
    ----------
    items : list of (key, original_value) tuples
        The items to allocate counts to, with their original values
    scale : float
        Scaling factor

    Returns
    -------
    dict
        Mapping from key to allocated integer count
    """
    total_original = sum(v for _, v in items)
    target_total = int(scale * total_original)

    allocations = {}
    allocated = 0
    for key, original in items:
        alloc = int(scale * original)
        allocations[key] = alloc
        allocated += alloc

    remainder = target_total - allocated
    if remainder > 0:
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            key = sorted_items[i % len(sorted_items)][0]
            allocations[key] += 1

    return allocations

def find_nodes(G, **attrs):
    """
    Finds the list of nodes in the graph associated that have attrs attributes.   
    Uses the predefined G.attrs_to_group and G.group_to_nodes dicts   
    (see graph.FileBasedGraph and generate.init_nodes())

    Parameters
    ----------
    G : FileBasedGraph instance

    Returns
    -------
    tuple (list, int)
        List contains all the node IDs
        int is the group ID
    """
    attrs_key = tuple(sorted(attrs.items()))
    group_id = G.attrs_to_group[attrs_key]
    if group_id is None:
        return []
    list_of_nodes = G.group_to_nodes[group_id]
    return list_of_nodes, group_id

def read_file(path):
    """ 
    CSV and XLSX file reader. Returns pandas dataframe.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format: {}".format(path))

def check_group_interactions(G, print_report=True):
    """
    Validate that actual edge counts between groups match the generation targets.

    Counts directed edges between every (src_group, dst_group) pair in the
    generated network and compares them to ``G.maximum_num_links``.

    Parameters
    ----------
    G : NetworkXGraph
        The generated graph, with ``nodes_to_group`` and ``maximum_num_links``
        populated.
    print_report : bool, optional
        If True (default), prints a formatted table of results.

    Returns
    -------
    dict
        Keys are (src_group, dst_group) tuples. Values are dicts with:
        ``actual``, ``target``, ``ratio`` (actual/target), ``diff`` (actual-target).
        Pairs with no target and no actual edges are omitted.
    """
    # Count actual edges per (src_group, dst_group)
    actual = {}
    for src, dst in G.graph.edges():
        sg = G.nodes_to_group.get(src)
        dg = G.nodes_to_group.get(dst)
        if sg is None or dg is None:
            continue
        key = (sg, dg)
        actual[key] = actual.get(key, 0) + 1

    # Union of all pairs that have a target or actual edges
    all_pairs = set(G.maximum_num_links.keys()) | set(actual.keys())

    results = {}
    for pair in sorted(all_pairs):
        tgt = G.maximum_num_links.get(pair, 0)
        act = actual.get(pair, 0)
        results[pair] = {
            'actual': act,
            'target': tgt,
            'diff': act - tgt,
            'ratio': (act / tgt) if tgt > 0 else float('inf'),
        }

    if print_report:
        def short_label(gid):
            attrs = G.group_to_attrs.get(gid) or G.group_to_attrs.get(str(gid))
            if not attrs:
                return str(gid)
            # Show only values, comma-separated (skip 'n' population key)
            vals = [str(v) for k, v in attrs.items() if k != 'n']
            return ', '.join(vals)

        W = 24  # column width for group labels
        header = f"  {'src_group':<{W}} {'dst_group':<{W}} {'target':>8} {'actual':>8} {'diff':>7} {'%':>6}"
        sep = "  " + "-" * (len(header) - 2)

        # Only show pairs that have a non-zero target or unexpected actual edges
        visible = {p: r for p, r in results.items() if r['target'] > 0 or r['actual'] > 0}
        n_ok  = sum(1 for r in visible.values() if abs(r['diff']) / max(r['target'], 1) < 0.05)
        n_bad = len(visible) - n_ok

        # print("\n" + "=" * len(header))
        # print(f"  Group interaction check  —  {len(visible)} pairs  |  {n_ok} OK  |  {n_bad} off by >5%")
        # print("=" * len(header))
        # print(header)
        # print(sep)

        total_target = total_actual = 0
        for (sg, dg), r in sorted(visible.items()):
            pct = 100 * r['actual'] / r['target'] if r['target'] > 0 else float('inf')
            flag = " OK" if abs(r['diff']) / max(r['target'], 1) < 0.05 else "!!!"
            src = short_label(sg)[:W]
            dst = short_label(dg)[:W]
            # print(f"  {src:<{W}} {dst:<{W}} {r['target']:>8} {r['actual']:>8} "
            #       f"{r['diff']:>+7} {pct:>5.1f}%  {flag}")
            total_target += r['target']
            total_actual += r['actual']

        overall_pct = 100 * total_actual / total_target if total_target > 0 else float('inf')
        # print(sep)
        label_col = f"  TOTAL ({len(visible)} pairs)"
        # print(f"{label_col:<{2 + W + 1 + W}} {total_target:>8} {total_actual:>8} "
        #       f"{total_actual - total_target:>+7} {overall_pct:>5.1f}%")
        # print("=" * len(header) + "\n")

    return results


def plot_group_interactions(results, G, scatter_path='group_scatter.png', bar_path='group_bar.png'):
    """
    Save two diagnostic plots for the output of check_group_interactions.

    Parameters
    ----------
    results : dict
        Return value of check_group_interactions.
    G : FileBasedGraph
        The generated graph (used to resolve group labels).
    scatter_path : str
        File path for the actual-vs-target scatter plot.
    bar_path : str
        File path for the per-pair ratio bar chart.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def short_label(gid):
        attrs = G.group_to_attrs.get(gid) or G.group_to_attrs.get(str(gid))
        if not attrs:
            return str(gid)
        vals = [str(v) for k, v in attrs.items() if k != 'n']
        return ', '.join(vals)

    pairs = sorted(results.keys())
    targets = np.array([results[p]['target'] for p in pairs])
    actuals = np.array([results[p]['actual'] for p in pairs])

    # --- Scatter: actual vs target ---
    fig1 = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig1)
    ax1 = fig1.add_subplot(111)
    ax1.scatter(targets, actuals, alpha=0.6, s=20)
    lim = max(targets.max(), actuals.max()) * 1.05
    ax1.plot([0, lim], [0, lim], 'r--', linewidth=1, label='actual = target')
    ax1.set_xlabel('Target')
    ax1.set_ylabel('Actual')
    ax1.set_title('Group pair edge counts: actual vs target')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(scatter_path, dpi=150)
    print(f"Saved scatter plot to {scatter_path}")

    # --- Heatmap: ratio per (src_group, dst_group) ---
    visible_pairs = {p: r for p, r in results.items() if r['target'] > 0 or r['actual'] > 0}
    src_groups = sorted({p[0] for p in visible_pairs})
    dst_groups = sorted({p[1] for p in visible_pairs})
    src_idx = {g: i for i, g in enumerate(src_groups)}
    dst_idx = {g: i for i, g in enumerate(dst_groups)}

    grid = np.full((len(src_groups), len(dst_groups)), np.nan)
    for (sg, dg), r in visible_pairs.items():
        ratio = r['ratio'] if r['target'] > 0 else 2.0
        grid[src_idx[sg], dst_idx[dg]] = ratio

    src_labels = [short_label(g) for g in src_groups]
    dst_labels = [short_label(g) for g in dst_groups]

    cell_size = 0.4
    fig_w = min(max(5, len(dst_groups) * cell_size + 3), 30)
    fig_h = min(max(4, len(src_groups) * cell_size + 2), 30)
    fig2 = Figure(figsize=(fig_w, fig_h))
    FigureCanvasAgg(fig2)
    ax2 = fig2.add_subplot(111)

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn_r  # red=over, yellow=near, green=under; reversed so green=1
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=2.0)
    im = ax2.imshow(grid, aspect='auto', cmap=cmap, norm=norm)
    fig2.colorbar(im, ax=ax2, label='actual / target', shrink=0.6)

    fs = max(5, min(9, int(200 / max(len(src_groups), len(dst_groups)))))
    ax2.set_xticks(np.arange(len(dst_groups)))
    ax2.set_xticklabels(dst_labels, rotation=90, fontsize=fs)
    ax2.set_yticks(np.arange(len(src_groups)))
    ax2.set_yticklabels(src_labels, fontsize=fs)
    ax2.set_xlabel('Destination group')
    ax2.set_ylabel('Source group')
    ax2.set_title('Group pair edge ratio heatmap (green=OK, red=over, white=under)')
    fig2.tight_layout()
    fig2.savefig(bar_path, dpi=100)
    print(f"Saved heatmap to {bar_path}")


def desc_groups(pops_path, pop_column = 'n'):
    """
    Reads the group sizes file. (csv or xlsx)
    All column headers in the file are considered as group characteristics except for pop_collumn.
    
    Parameters
    ----------
    pops_path : string
        The filepath for the group sizes file. Can be csv or xlsx.
    pop_column : string
        The name of the column that contains the population value.
    Returns
    -------
    tuple (dict, list)
        The dict contains the group IDs as keys and the sizes (populations) as value.   
        The list contains the names of the group characteristic collumns.
    """
    df_group_pops = read_file(pops_path)
    df_group_pops = df_group_pops.sort_values("n", ascending = False)
    # Identify characteristic columns (all except pop_column)
    characteristic_cols = [col for col in sorted(df_group_pops.columns) if col != pop_column]

    # Each group gets a unique ID (row number)
    group_populations = {
        idx: {**{col: row[col] for col in characteristic_cols}, pop_column: row[pop_column]}
        for idx, row in df_group_pops.iterrows()
    }

    return group_populations, characteristic_cols

