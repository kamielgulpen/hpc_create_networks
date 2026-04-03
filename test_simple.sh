#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

echo "Script started"
pwd
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}"
echo "Changed to: $(pwd)"
ls -la
echo "Script finished"
