#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

echo "Hello from SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
date
sleep 10
echo "Done!"
