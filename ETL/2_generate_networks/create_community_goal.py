#!/usr/bin/env python3
"""
Sweep community size for etngrp_geslacht_lft_oplniv (all layers), recording the
refinement loss create_communities() returns for each LHS sample. Gives the
reference loss curve the other aggregations can target.

Loss depends only on fraction_of_communities + the pop/interaction data (not on
transitivity, a generate() parameter), so reusing the LHS samples characterises
it fully. Parallel across (layer, sample) pairs on one machine; results are
written to one parquet PER LAYER, with the layer in the filename. Run alongside
data_lake.py.
"""

import argparse
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import math                                          # new

import pandas as pd
from SALib.sample import latin

from asnu import create_communities
from run_parallel_generate_network import N_SAMPLES

import data_lake

# Must match the generation settings, or the loss isn't comparable.
SCALE               = 0.1
REFINE_SWAPS        = 1000000

N_SAMPLES          = int(N_SAMPLES/2)
RANDOM_SEED        = 42
PREF_ATTACHMENT    = 0
BRIDGE_PROBABILITY = 0.0
POP                = 861000 * SCALE


PROBLEM = {
    'num_vars': 2,
    'names':    ['n_communities', 'transitivity'],
    'bounds':   [[math.log10(1 / POP), math.log10(1.0)],   # log10 fraction
                 [0.0, 1.0]],                              # transitivity
}


AGG_LEVEL   = 'etngrp_geslacht_lft_oplniv'
WRITE_EVERY = 1  # checkpoint the per-layer parquet every N completed tasks


def get_or_create_samples() -> pd.DataFrame:
    if data_lake.samples_exist():
        return data_lake.read_samples()
    samples = latin.sample(PROBLEM, N_SAMPLES, seed=RANDOM_SEED)
    samples[:, 0] = 10.0 ** samples[:, 0]    
    
    df = pd.DataFrame(samples, columns=PROBLEM['names'])
    df.insert(0, 'sample_id', df.index)
    df.insert(3, 'optimize', 0)
    print(df.shape)
    df1 = pd.DataFrame(samples, columns=PROBLEM['names'])
    df1.insert(0, 'sample_id', df1.index + df.shape[0])
    df1.insert(3, 'optimize', 1)
    df = pd.concat([df, df1])
    data_lake.write_samples(df)
    print(f"Wrote {len(df)} samples to the data lake")
    return df


def loss_for_fraction(pops_path, links_path, fraction):
    """One create_communities() call -> its loss. create_communities returns
    (assignments, loss); we keep the loss. The community JSON is written to a
    temp file we discard."""
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
    try:
        return float(create_communities(
            pops_path, links_path,
            scale=SCALE,
            fraction_of_communities=fraction,
            output_path=tmp,
            refine_swaps=REFINE_SWAPS,
        )[1])
    finally:
        os.unlink(tmp)


def _worker(task):
    """Runs in a child process. Pure compute -- no writes to shared state.
    CSVs were materialized once by the parent; we only read the paths."""
    sample_id, layer, fraction, pops_path, links_path = task
    # loss = loss_for_fraction(pops_path, links_path, fraction)
    loss = 0.0
    return {'sample_id': sample_id, 'layer': layer,
            'n_communities': fraction, 'loss': loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    samples = get_or_create_samples()
    samples = samples[samples['optimize'] == 1] # Use only optimized samples to optimize community structure

    level_dir = data_lake.ROOT / 'aggregation_levels' / AGG_LEVEL
    layers = sorted(p.stem[len('interactions_'):]
                    for p in level_dir.glob('interactions_*.parquet'))

    out_dir = data_lake.ROOT / 'refine_loss_reference'
    out_dir.mkdir(parents=True, exist_ok=True)

    def layer_path(layer):
        # One file per layer, layer encoded in the name.
        return out_dir / f'{AGG_LEVEL}__{layer}.parquet'

    def flush(layer, layer_rows):
        (pd.DataFrame(layer_rows)
           .sort_values('sample_id')
           .to_parquet(layer_path(layer), index=False))

    # Materialize CSVs ONCE, into a temp dir that outlives the whole pool. The
    # pop file is shared; one links file per layer. Workers just read paths.
    with tempfile.TemporaryDirectory() as tmp_dir:
        pops_path = f'{tmp_dir}/pop.csv'
        pd.read_parquet(level_dir / 'population.parquet').to_csv(pops_path, index=False)

        tasks = []
        for layer in layers:
            links_path = f'{tmp_dir}/interactions_{layer}.csv'
            pd.read_parquet(level_dir / f'interactions_{layer}.parquet').to_csv(links_path, index=False)
            for _, row in samples.iterrows():
                tasks.append((int(row['sample_id']), layer,
                              float(row['n_communities']), pops_path, links_path))

        print(f"{len(tasks)} tasks ({len(layers)} layers x {len(samples)} samples), "
              f"{args.workers} workers")

        # Results grouped per layer so each writes to its own file. as_completed
        # returns out of order, hence the dict keyed by layer.
        rows_by_layer = defaultdict(list)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, t) for t in tasks]
            for done, fut in enumerate(as_completed(futures), 1):
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"  [{done}/{len(tasks)}] FAILED: {e!r}")
                    continue
                rows_by_layer[r['layer']].append(r)
                print(f"  [{done}/{len(tasks)}] {r['layer']}  "
                      f"sample {r['sample_id']:05d}  loss={r['loss']:.6g}")
                if done % WRITE_EVERY == 0:
                    flush(r['layer'], rows_by_layer[r['layer']])

        # Final flush -- every layer, once.
        for layer, layer_rows in rows_by_layer.items():
            flush(layer, layer_rows)

    total = sum(len(v) for v in rows_by_layer.values())
    print(f"\nDone. {total} rows across {len(rows_by_layer)} layer file(s) -> {out_dir}")


if __name__ == '__main__':
    main()