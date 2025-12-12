#!/bin/bash
#SBATCH --job-name=active_learning
#SBATCH --output=/das/home/barbaf_l/smartTT/examples/logs/active_learning_%j.out
#SBATCH --error=/das/home/barbaf_l/smartTT/examples/logs/active_learning_%j.err
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:1

# Active Learning Pipeline - SLURM Launcher
# ========================================================
# This script runs the active learning example on SLURM
# 
# Usage:
#   sbatch run_active_learning.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/active_learning_JOBID.out

# Create logs directory if it doesn't exist
mkdir -p /das/home/barbaf_l/smartTT/examples/logs

# Set up environment
echo "=================================================="
echo "Active Learning Pipeline Starting"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=================================================="

# Change to the smartTT directory
cd /das/home/barbaf_l/smartTT

module load anaconda/2019.07

# Activate conda environment if needed (uncomment and modify as needed)
# source /path/to/conda/etc/profile.d/conda.sh
conda activate /das/work/units/pem/p20639/envs/p20639

# Run the active learning example
echo "Running active learning example..."
python examples/active_learning_example.py

# Check exit status
EXIT_STATUS=$?

echo "=================================================="
echo "Active Learning Pipeline Complete"
echo "End time: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "=================================================="

exit $EXIT_STATUS
