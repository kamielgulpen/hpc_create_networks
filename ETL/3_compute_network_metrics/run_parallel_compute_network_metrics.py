#!/usr/bin/env python3
"""
Pure-Python parallel runner for the network-metrics stage, replacing
run_compute_metrics_locally.sh.

Runs `python compute_metrics.py --network_id X` once per network in the
data lake that has an edges.npz, up to --max_parallel at a time, logging
each task's stdout/stderr to its own file under logs_metrics/. Same
concurrency model as run_parallel_generation.py: every task is a fully
separate OS process -- important here specifically, since a single huge
network's igraph computation crashing or exhausting memory shouldn't take
the whole batch down with it.

With `--only assortativity`, runs the assortativity backfill instead
(`compute_network_metrics.py --only assortativity`): only networks that
don't yet have assortativity stats are processed, and nothing else in
their network_stats is recomputed. Logs go to logs/metrics_assortativity.

Usage:
    python run_parallel_metrics.py
    python run_parallel_metrics.py --max_parallel 5
    python run_parallel_metrics.py --only assortativity --max_parallel 10
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import sys
LAKE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(1, str(LAKE_DIR))
import data_lake

SCRIPT_DIR = Path(__file__).resolve().parent

# Must match ASSORT_DONE_KEY in compute_network_metrics.py
ASSORT_DONE_KEY = "assortativity_out_in_pearson"


def discover_network_ids() -> list[str]:
    net_dir = data_lake.ROOT / 'networks'
    if not net_dir.exists():
        return []
    return sorted(p.name for p in net_dir.iterdir() if p.is_dir() and "ncom" in p.name and (p / 'edges.npz').exists())


def _stat_names(network_id: str) -> pd.DataFrame | None:
    stats_path = data_lake.ROOT / 'networks' / network_id / 'network_stats.parquet'
    if not stats_path.exists():
        return None
    return pd.read_parquet(stats_path)


def already_done(network_id: str) -> bool:
    df = _stat_names(network_id)
    if df is None:
        return False
    if not 'global_clustering' in set(df['stat_name']): return False
    row = df[df['stat_name'] == 'global_clustering']
    if row.empty:
        return False
    return row['stat_value'].notna().any() 


def assortativity_done(network_id: str) -> bool:
    # Presence only: nan is a legitimate value (constant degrees).
    df = _stat_names(network_id)
    return df is not None and ASSORT_DONE_KEY in set(df['stat_name'])


def run_task(network_id: str, script: str, python: str, logs_dir: Path,
             only: str | None = None) -> tuple[str, int]:
    out_path = logs_dir / f"{network_id}.out"
    err_path = logs_dir / f"{network_id}.err"
    cmd = [python, script, "--network_id", network_id]
    if only:
        cmd += ["--only", only]
    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=err_f, cwd=SCRIPT_DIR)
    return network_id, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--script", default="compute_network_metrics.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", choices=["assortativity"], default=None,
                        help="Backfill only this group of stats onto networks missing it.")
    args = parser.parse_args()

    logs_dir = data_lake.ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    logs_dir = logs_dir / ("metrics" if args.only is None else f"metrics_{args.only}")
    logs_dir.mkdir(exist_ok=True)

    all_network_ids = discover_network_ids()
    if not all_network_ids:
        print(f"No networks with edges.npz found under {data_lake.ROOT / 'networks'}")
        return

    done_check = assortativity_done if args.only == "assortativity" else already_done
    todo = [nid for nid in all_network_ids if not done_check(nid)]
    skipped = len(all_network_ids) - len(todo)

    mode = "full metrics" if args.only is None else f"{args.only} only"
    print(f"Mode: {mode}")
    print(f"Found {len(all_network_ids)} networks, {skipped} already done, {len(todo)} to process")
    print(f"Max parallel workers: {args.max_parallel}")
    print(f"Starting at {datetime.now()}")

    if not todo:
        print("Nothing to do.")
        return

    completed = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(run_task, nid, args.script, args.python, logs_dir, args.only): nid
            for nid in todo
        }
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
    print(f"Check logs in {logs_dir} for details")

    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()