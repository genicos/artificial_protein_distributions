#!/bin/bash
#SBATCH --cluster=whale
#SBATCH --partition=long
#SBATCH --account=researchers
#SBATCH --job-name=quick_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=00-01:00
#SBATCH --output=/home/nico/logs/%j.out
#SBATCH --error=/home/nico/logs/%j.err



python3 main.py --seed 0 --num_epochs 6 --num_samples 100000 1> OUT 2> ERR
