"""Utility functions used during graph generation."""

import pandas as pd


def stratified_allocate(items, scale):
    """Allocate integer counts from fractional scaled values, preserving the total.

    Each item gets floor(scale * original); the remainder is distributed
    round-robin to the largest items to hit the exact scaled total.

    items : list of (key, original_value) tuples
    scale : float scaling factor
    Returns {key: allocated integer count}.
    """
    target_total = int(scale * sum(v for _, v in items))

    allocations = {key: int(scale * original) for key, original in items}
    remainder = target_total - sum(allocations.values())

    if remainder > 0:
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            key = sorted_items[i % len(sorted_items)][0]
            allocations[key] += 1

    return allocations


def find_nodes(G, **attrs):
    """Find the node IDs whose attributes match `attrs`.

    Uses G.attrs_to_group and G.group_to_nodes (see graph / generate.init_nodes).
    Returns (list_of_node_ids, group_id).
    """
    attrs_key = tuple(sorted(attrs.items()))
    group_id = G.attrs_to_group[attrs_key]
    if group_id is None:
        return []
    return G.group_to_nodes[group_id], group_id


def read_file(path):
    """Read a CSV, XLSX, or Parquet file into a pandas DataFrame."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def desc_groups(pops_path, pop_column='n'):
    """Read the group-sizes file (csv/xlsx/parquet).

    Every column except `pop_column` is treated as a group characteristic. Each
    group gets a unique ID (its row number after sorting by population desc).

    Returns ({group_id: {characteristic_cols..., pop_column: size}},
             [characteristic column names]).
    """
    df = read_file(pops_path).sort_values(pop_column, ascending=False)
    characteristic_cols = [col for col in sorted(df.columns) if col != pop_column]

    group_populations = {
        idx: {**{col: row[col] for col in characteristic_cols}, pop_column: row[pop_column]}
        for idx, row in df.iterrows()
    }

    return group_populations, characteristic_cols