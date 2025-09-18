#!/bin/bash
# Job name:
#SBATCH --job-name=SAXS-TT
#
#SBATCH --partition=hour
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Wall clock limit:
#SBATCH --time=00:55:00
#
## Command(s) to run (example):

source /das/home/carlse_m/.bashrc
conda activate mumott_dev
cd /das/work/units/pem/p20639/Mads/frog
python recon_qbin.py $1
