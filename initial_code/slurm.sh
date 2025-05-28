#!/bin/bash
#SBATCH --cluster=whale
#SBATCH --partition=long
#SBATCH --account=researchers
#SBATCH --job-name=quick_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00-20:00
#SBATCH --output=/home/nico/logs/%j.out
#SBATCH --error=/home/nico/logs/%j.err

#python3 train.py --uniform_masking --seed 0 --num_epochs 30 --num_samples 100000
python3 train_full.py --uniform_masking --seed 0 --num_epochs 30 --num_samples 100000 > OUT