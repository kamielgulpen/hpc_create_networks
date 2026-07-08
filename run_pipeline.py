#!/usr/bin/env python3
"""
Runs the three-part analysis-table pipeline sequentially:

    1. compute_ignition_probability.py
    2. build_networks_table.py
    3. join_tables.py

Each stage runs as its own subprocess (fresh memory, isolated from
whatever the previous stage was doing -- these can be memory-heavy on
large data, so no leftover state carries over). Output is streamed live
to the console as it happens AND saved to logs_pipeline/<stage>.log, so
you get real-time progress (e.g. compute_ignition_probability's batch
progress prints) without losing the record afterward.

Stops on the first failure by default, since each stage depends on files
the previous one wrote -- there's nothing useful a later stage can do if
an earlier one didn't finish.

Usage:
    python run_pipeline.py
    python run_pipeline.py --from-stage 2   # skip stage 1, e.g. to retry
                                              # after fixing stage 2 without
                                              # re-running the slow scan
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs_pipeline"

STAGES = [
    "ETL/2_generate_networks/run_parallel_generate_network.py",
    "ETL/3_compute_network_metrics/run_parallel_compute_network_metrics.py",
    "ETL/4_seeding_experiment/run_parallel_seeding_experiments.py",
    "ETL/5_build_analysis_table/build_networks_table.py"
]


def run_stage(script: str, python: str) -> int:
    log_path = LOGS_DIR / f"{Path(script).stem}.log"
    print(f"\n=== Running {script} (log: {log_path}) ===", flush=True)

    with open(log_path, "w") as log_f:
        process = subprocess.Popen(
            [python, script],
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge, so the log is in true chronological order
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        process.wait()

    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", type=int, default=1,
                        help="1-indexed stage to start from (skip earlier ones)")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True)

    stages_to_run = STAGES[args.from_stage - 1:]
    if not stages_to_run:
        print(f"--from-stage {args.from_stage} is past the last stage ({len(STAGES)}). Nothing to do.")
        return

    print(f"Starting pipeline at {datetime.now()} "
          f"({len(stages_to_run)}/{len(STAGES)} stage(s): {stages_to_run})")

    for script in stages_to_run:
        t0 = datetime.now()
        returncode = run_stage(script, args.python)
        elapsed = (datetime.now() - t0).total_seconds()

        if returncode != 0:
            print(f"\n[FAILED] {script} exited with code {returncode} after {elapsed:.1f}s")
            print(f"See {LOGS_DIR / (Path(script).stem + '.log')} for the full output.")
            print("Stopping -- later stages depend on this one's output.")
            sys.exit(1)

        print(f"[OK] {script} finished in {elapsed:.1f}s")

    print(f"\nPipeline completed at {datetime.now()}")


if __name__ == "__main__":
    main()