#!/usr/bin/env python3
"""
Pure-Python parallel runner for the diffusion-simulation stage, replacing
the bash sweep script.

Runs `python seeding_experiments.py --network_id X` once per network in the
data lake, up to --max_parallel at a time. Each subprocess gets
NUMBA_NUM_THREADS / OMP_NUM_THREADS set to --cores_per_task, so
max_parallel x cores_per_task should be tuned to your total core count --
the numba kernel is itself parallel, and oversubscribing cores makes
everything slower, not faster.

Usage:
    python run_parallel_simulations.py
    python run_parallel_simulations.py --max_parallel 6 --cores_per_task 6
    python run_parallel_simulations.py --n_sims 100 --seeding neighbor_k --neighbor_k 20
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import pandas as pd

import data_lake

SCRIPT_DIR = Path(__file__).resolve().parent


def discover_network_ids() -> list[str]:
    net_dir = data_lake.ROOT / 'networks'
    if not net_dir.exists():
        return []
    return sorted(p.name for p in net_dir.iterdir()
                  if p.is_dir() and (p / 'edges.npz').exists())

def discover_network_aggregation() -> list[str]:
    net_dir = data_lake.ROOT / 'networks'
    if not net_dir.exists():
        return []   
    
    return sorted(
        (p.name.split("__")[1], p.name.split("__")[2], p.name) 
        for p in net_dir.iterdir() 
        if "__" in p.name
    )  

def discover_ks(aggregations: list[tuple[str, str, str]]) -> list[int]:
    ks = {}
    for aggregation in aggregations:
       df = pd.read_parquet(data_lake.ROOT / 'aggregation_levels' / aggregation[0]/ 'mixing_features.parquet')
       k = df[df["feature_name"] == f"{aggregation[1]}_degree_node_mean"]["feature_value"].iloc[0]
       ks[aggregation[2]] = int(k)
    return ks

def already_done(network_id: str, n_thresholds: int) -> bool:
    ie_dir = data_lake.ROOT / 'infection_events' / network_id
    return all((ie_dir / f'threshold_{i}.parquet').exists() for i in range(n_thresholds))


def run_task(network_id: str, ks, args, logs_dir: Path) -> tuple[str, int]:
    out_path = logs_dir / f"{network_id}.out"
    err_path = logs_dir / f"{network_id}.err"

    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(args.cores_per_task)
    env["OMP_NUM_THREADS"] = str(args.cores_per_task)
    env["PYTHONUNBUFFERED"] = "1"

    print(args, ks)
    cmd = [
        args.python, args.script,
        "--network_id", network_id,
        "--n_sims", str(args.n_sims),
        "--seeding", args.seeding,
        "--neighbor_k", ks,
        "--verbose", str(args.verbose),
    ]
    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=err_f,
                                 cwd=SCRIPT_DIR, env=env)
    return network_id, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--cores_per_task", type=int, default=6,
                        help="NUMBA/OMP threads per subprocess; "
                             "max_parallel x cores_per_task ~= total cores")
    parser.add_argument("--n_thresholds", type=int, default=1,
                        help="Must match SimulationConfig.n_thresholds -- "
                             "used only for the skip-if-done check")
    parser.add_argument("--n_sims", type=int, default=100)
    parser.add_argument("--seeding", type=str, default="neighbor_k",
                        choices=["random", "focal_neighbors", "neighbor_k"])
    parser.add_argument("--neighbor_k", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--script", default="seeding_experiments.py")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    logs_dir = data_lake.ROOT/ "logs"
    logs_dir.mkdir(exist_ok=True)
    logs_dir = data_lake.ROOT/ "logs" / "logs_simulations"
    logs_dir.mkdir(exist_ok=True)

    all_ids = discover_network_ids()
    aggregations = discover_network_aggregation()
    ks = discover_ks(aggregations)

    if not all_ids:
        print(f"No networks with edges.npz found under {data_lake.ROOT / 'networks'}")
        return

    todo = [nid for nid in all_ids if not already_done(nid, args.n_thresholds)]
    skipped = len(all_ids) - len(todo)

    print(f"Found {len(all_ids)} networks, {skipped} already done, {len(todo)} to simulate")
    print(f"{args.max_parallel} tasks in parallel, {args.cores_per_task} numba threads each")
    print(f"Starting at {datetime.now()}")

    if not todo:
        print("Nothing to do.")
        return

    completed = 0
    failures = []
  
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(run_task, nid, str(ks[nid]), args, logs_dir): nid for nid in todo}
        try:
            for future in as_completed(futures):
                network_id, returncode = future.result()
                completed += 1
                if returncode != 0:
                    failures.append(network_id)
                if completed % 10 == 0 or completed == len(todo):
                    print(f"Progress: {completed}/{len(todo)} finished at {datetime.now()}")
        except KeyboardInterrupt:
            print(f"\nInterrupted after {completed}/{len(todo)} -- "
                  f"cancelling not-yet-started tasks...")
            for f in futures:
                f.cancel()
            raise

    print(f"\nAll tasks completed at {datetime.now()}")
    print(f"Skipped (already done): {skipped}")
    print(f"Results: {completed - len(failures)} ok, {len(failures)} errors out of {completed}")
    if failures:
        print(f"Failed network_ids ({len(failures)}): {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()