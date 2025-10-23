#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=72:00:00
#SBATCH --output=protocell_runner.out
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A iprime
#SBATCH --cpus-per-task=1
#SBATCH --array=1-8


module load anaconda
conda activate /scratch/mbt8hz/python_environment/pytorch_py3_10/
python runner.py
