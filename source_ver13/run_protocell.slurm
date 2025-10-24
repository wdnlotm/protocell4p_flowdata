#!/bin/bash
#SBATCH --job-name=protocell_rh2
#SBATCH --nodes=1
#SBATCH --output=autput_%x_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=72:00:00
#SBATCH -A iprime

module load gcc openmpi python
source your python virtual environment
cd your source directory
python python run_ptorocell_cytof_ver013.py

# sbatch run_protocell.slurm
