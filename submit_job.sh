#!/bin/bash
# =============================================================================
# SLURM Array Job: Enriched Network Generation
#
# Parameter space: 10 pref_attachment x 10 n_communities = 100 combinations
# Each array task generates all enriched pairs from Data/enriched/aggregated/
# for one (pref_attachment, n_communities) combination.
#
# Workflow:
#   1. sbatch submit_job.sh
# =============================================================================

#SBATCH --job-name=create_networks
#SBATCH --array=0-99               # 100 tasks (0-indexed), one per parameter combo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00            # adjust per your cluster limits
#SBATCH --output=logs/create_%A_%a.out
#SBATCH --error=logs/create_%A_%a.err

# --- Environment setup -------------------------------------------------------
# Adjust the lines below to match your cluster's module system and venv path.

# module load python/3.11          # adjust version if needed
# source .venv/bin/activate

# -----------------------------------------------------------------------------

mkdir -p logs

echo "Starting task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"

python run_task.py --task_id "${SLURM_ARRAY_TASK_ID}"

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"
