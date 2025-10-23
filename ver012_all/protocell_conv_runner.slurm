#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=protocell_runner.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=72:00:00
#SBATCH -A iprime


module load anaconda
conda activate /scratch/mbt8hz/python_environment/pytorch_py3_10/
jupyter nbconvert --to python Protocell_cytof_ver010.ipynb
python Protocell_cytof_ver010.ipynb.py
