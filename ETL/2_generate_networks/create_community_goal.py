#!/usr/bin/env python3
"""
Sweep community size for etngrp_geslacht_lft_oplniv (all layers), recording the
refinement loss create_communities() returns for each LHS sample. Gives the
reference loss curve the other aggregations can target.

Loss depends only on fraction_of_communities + the pop/interaction data (not on
transitivity, a generate() parameter), so reusing the LHS samples characterises
it fully. Parallel across (layer, sample) pairs on one machine. Run alongside
data_lake.py.
"""

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from SALib.sample import latin

from asnu import create_communities
from run_parallel_generate_network import N_SAMPLES

import data_lake

# Must match the generation settings, or the loss isn't comparable.
SCALE               = 0.01
ISOLATION_THRESHOLD = 0.8
REFINE_SWAPS        = 1000000

SCALE           = 0.01
N_SAMPLES       = N_SAMPLES
RANDOM_SEED     = 42
PREF_ATTACHMENT = 0  # held fixed
BRIDGE_PROBABILITY = 0.0  # held fixed
POP = 861000 * SCALE

PROBLEM = {
    'num_vars': 2,
    'names':    ['n_communities', 'transitivity'],
    'bounds':   [[1/POP,   1.0],
                 [0.0, 1.0]
                 ]
}

AGG_LEVEL = 'etngrp_geslacht_lft_oplniv'

WRITE_EVERY = 25  # checkpoint the parquet every N completed tasks

def get_or_create_samples() -> pd.DataFrame:
    if data_lake.samples_exist():
        return data_lake.read_samples()
    samples = latin.sample(PROBLEM, N_SAMPLES, seed=RANDOM_SEED)
    print(samples)
    df = pd.DataFrame(samples, columns=PROBLEM['names'])
    df.insert(0, 'sample_id', df.index)
    df.insert(3, 'optimize', 0) 
    df1 = pd.DataFrame(samples, columns=PROBLEM['names'])
    df1.insert(0, 'sample_id', df.index)
    df1.insert(3, 'optimize', 1) 
    df = pd.concat([df,df1])
    data_lake.write_samples(df)
    print(f"Wrote {len(df)} samples to the data lake")
    return df


def loss_for_fraction(pops_path, links_path, fraction):
    """One create_communities() call -> its loss. The community JSON is written
    to a temp file we discard; only the returned loss matters."""
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
    try:
        return float(create_communities(
            pops_path, links_path,
            scale=SCALE,
            fraction_of_communities=fraction,
            output_path=tmp,
            isolation_threshold=ISOLATION_THRESHOLD,
            refine_swaps=REFINE_SWAPS,
        ))
    finally:
        os.unlink(tmp)


def _worker(task):
    """Runs in a child process. Pure compute -- no file writes to shared state.
    CSVs were materialized once by the parent; we only read the paths."""
    sample_id, layer, fraction, pops_path, links_path = task
    loss = loss_for_fraction(pops_path, links_path, fraction)
    return {'sample_id': sample_id, 'layer': layer,
            'n_communities': fraction, 'loss': loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    samples = get_or_create_samples() 
    level_dir = data_lake.ROOT / 'aggregation_levels' / AGG_LEVEL
    layers = sorted(p.stem[len('interactions_'):]
                    for p in level_dir.glob('interactions_*.parquet'))

    out_path = data_lake.ROOT / 'refine_loss_reference' / f'{AGG_LEVEL}.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

        rows = []
        # Parent owns all parquet writes -- workers never touch the output file.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, t) for t in tasks]
            for done, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                rows.append(r)
                print(f"  [{done}/{len(tasks)}] {r['layer']}  "
                      f"sample {r['sample_id']:05d}  loss={r['loss']:.6g}")
                if done % WRITE_EVERY == 0:
                    pd.DataFrame(rows).sort_values(['layer', 'sample_id']).to_parquet(out_path, index=False)

        pd.DataFrame(rows).sort_values(['layer', 'sample_id']).to_parquet(out_path, index=False)

    print(f"\nDone. {len(rows)} rows -> {out_path}")


if __name__ == '__main__':
    main()