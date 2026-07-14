#!/usr/bin/env python3
"""
Pure-Python equivalent of generate_networks.sh.

Runs `python PAWN_analysis.py --task_id N` once per task, up to
--max_parallel at a time, logging each task's stdout/stderr to its own
file under logs/. Same concurrency model as the bash version: every task
is a fully separate OS process touching only its own files (its own log
files, and -- inside PAWN_analysis.py -- only its own network_id/run_id
in the data lake), so this is safe to run with many workers. The
per-task `if edges_file.exists(): skip` check inside PAWN_analysis.py
already makes individual tasks idempotent; this launcher doesn't need to
duplicate that.

Usage:
    python run_parallel_generation.py
    python run_parallel_generation.py --tasks 750 --max_parallel 8
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import data_lake

N_SAMPLES = 300
SCRIPT_DIR = Path(__file__).resolve().parent


def run_task(task_id: int, script: str, python: str, logs_dir: Path) -> tuple[int, int]:
    """Run one task_id as a subprocess; return (task_id, returncode)."""
    out_path = logs_dir / f"task_{task_id}.out"
    err_path = logs_dir / f"task_{task_id}.err"
    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        result = subprocess.run(
            [python, script, "--task_id", str(task_id)],
            stdout=out_f, stderr=err_f, cwd=SCRIPT_DIR,
        )
    return task_id, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=N_SAMPLES)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--script", default="generate_networks.py")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()


    logs_dir = data_lake.ROOT/ "logs"
    logs_dir.mkdir(exist_ok=True)

    logs_dir = data_lake.ROOT/ "logs" / "logs_generation"
    logs_dir.mkdir(exist_ok=True)

    print(f"Starting {args.tasks} tasks at {datetime.now()}")

    completed = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(run_task, task_id, args.script, args.python, logs_dir): task_id
            for task_id in range(args.tasks)
        }
        try:
            for future in as_completed(futures):
                task_id, returncode = future.result()
                completed += 1
                if returncode != 0:
                    failures.append(task_id)
                if completed % 5 == 0 or completed == args.tasks:
                    print(f"Progress: {completed}/{args.tasks} finished "
                          f"(task {task_id}, rc={returncode})")
        except KeyboardInterrupt:
            print(f"\nInterrupted after {completed}/{args.tasks} -- "
                  f"cancelling not-yet-started tasks...")
            for f in futures:
                f.cancel()
            raise

    print(f"All tasks completed at {datetime.now()}")
    print(f"Results: {completed}/{args.tasks} tasks finished, {len(failures)} failed")
    if failures:
        print(f"Failed task_ids ({len(failures)}): {sorted(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()