""" This module contains utility functions used during graph generation. """

import pandas as pd

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
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format: {}".format(path))


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