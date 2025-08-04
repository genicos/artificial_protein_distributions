import torch
from torch.optim import Adam
from data import get_loaders
from data_generation_energy import generate_energy_distribution, sequence_log_likelihood, sample_sequences, save_energy_distribution, load_energy_distribution, get_noised_fitness
import numpy as np
from model import MaskedLanguageModel
import sys
import torch.nn.functional as F
import argparse
import random
import math
import itertools
import matplotlib.pyplot as plt
import os

from train import train
from validation import single_sequence_scoring






def main():
    
    print("Starting main", flush=True)
    exit()

    parser = argparse.ArgumentParser(description="Train Masked Language Model on Energy Distribution")

    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=10000,
                      help='Number of samples to train on')
    
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU num to use')

    parser.add_argument('--distribution_parameter', type=float, default=3.0,
                      help='Parameter for the energy distribution')

    args = parser.parse_args()

    # Set all random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set CUDA random seed if using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create a generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)
    

    #Define protein data distribution
    L = 10
    alphabet_size = 5

    energy_params = generate_energy_distribution( #Energy distribution is deterministic with seed
        L=L,
        alphabet_size=alphabet_size,
        upper_range=args.distribution_parameter
    )

    train_ratio = 0.9


    validation_set = sample_sequences(energy_params, 100)
    validation_set_log_p = [sequence_log_likelihood(seq, energy_params) for seq in validation_set]


    # Train model
    model = train(energy_params,
                num_epochs=args.num_epochs,
                num_samples=args.num_samples,
                gpu=args.gpu,
                generator=g,
                train_ratio=train_ratio)






    print("Single Sequence Scoring", flush=True)
    results = single_sequence_scoring(model, energy_params, args, num_samples=1000)
    print("Finished Single Sequence Scoring", flush=True)
    gen_ll_mean_square_error = np.mean((results['truth'] - results['scores']['generation_likelihood'])**2)
    gen_ll_greedy_mean_square_error = np.mean((results['truth'] - results['scores']['generation_likelihood_greedy_path'])**2)
    pl_mean_square_error = np.mean((results['truth'] - results['scores']['pseudo_likelihood'])**2)
    print(f"Generation Likelihood Mean Square Error: {gen_ll_mean_square_error:.4f}")
    print(f"Greedy Generation Likelihood Mean Square Error: {gen_ll_greedy_mean_square_error:.4f}")
    print(f"Pseudo Likelihood Mean Square Error: {pl_mean_square_error:.4f}")
    print(f"Generation Likelihood Correlation: {results['correlations']['generation_likelihood']:.4f}")
    print(f"Greedy Generation Likelihood Correlation: {results['correlations']['generation_likelihood_greedy_path']:.4f}")
    print(f"Pseudo Likelihood Correlation: {results['correlations']['pseudo_likelihood']:.4f}")

    #Create subplot of scatter plot of generation likelihood and pseudo likelihood vs truth
    plt.figure(figsize=(20, 5))
    
    # Get min and max for consistent y = x line
    min_val = min(min(results['truth']), min(results['scores']['generation_likelihood']),
                min(results['scores']['generation_likelihood_greedy_path']),
                min(results['scores']['pseudo_likelihood']))
    max_val = max(max(results['truth']), max(results['scores']['generation_likelihood']),
                max(results['scores']['generation_likelihood_greedy_path']),
                max(results['scores']['pseudo_likelihood']))

    # Create the plots
    plt.figure(figsize=(20, 5))

    # Subplot 1
    plt.subplot(1, 3, 1)
    plt.scatter(results['truth'], results['scores']['generation_likelihood'], label=f"Generation Likelihood corr: {results['correlations']['generation_likelihood']:.4f}", s=1)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')  # Dashed y = x line
    plt.xlabel('Truth')
    plt.ylabel('Score')
    plt.legend()

    # Subplot 2
    plt.subplot(1, 3, 2)
    plt.scatter(results['truth'], results['scores']['generation_likelihood_greedy_path'], label=f"Greedy Generation Likelihood corr: {results['correlations']['generation_likelihood_greedy_path']:.4f}", s=1)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    plt.xlabel('Truth')
    plt.ylabel('Score')
    plt.legend()

    # Subplot 3
    plt.subplot(1, 3, 3)
    plt.scatter(results['truth'], results['scores']['pseudo_likelihood'], label=f"Pseudo Likelihood corr: {results['correlations']['pseudo_likelihood']:.4f}", s=1)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    plt.xlabel('Truth')
    plt.ylabel('Score')
    plt.legend()

    # Save figure
    plt.savefig('single_sequence_scoring.png')


if __name__ == "__main__":
    main()