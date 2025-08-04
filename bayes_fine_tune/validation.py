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




def pseudo_likelihood(model, sequence, alphabet_size, device):
    """Calculate pseudo-likelihood"""
    seq_tensor = torch.tensor(sequence).to(device)
    pl_score = 0.0
        
    for pos in range(len(sequence)):
        masked = seq_tensor.clone()
        masked[pos] = alphabet_size
            
        with torch.no_grad():
            logits = model(masked.unsqueeze(0))[0, pos]
            
        pl_score += F.log_softmax(logits, dim=-1)[sequence[pos]].item()
    
    return pl_score
    

def generation_likelihood(model, sequence, alphabet_size, device, paths=6):
    """Calculate generation likelihood through progressive unmasking"""
    seq_len = len(sequence)
    total_log_prob = 0.0
    seq_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
    
    for _ in range(paths):
        current = torch.full((seq_len,), alphabet_size, device=device)
        remaining_pos = list(range(seq_len))
        path_log_prob = 0.0
        
        while remaining_pos:
            # Randomly select next position to unmask
            idx = np.random.randint(len(remaining_pos))
            pos = remaining_pos.pop(idx)
            
            # Get model predictions
            with torch.no_grad():
                logits = model(current.unsqueeze(0))[0, pos]
            
            # Add log probability of correct amino acid
            path_log_prob += F.log_softmax(logits, dim=-1)[seq_tensor[pos]].item()
            
            # Update current sequence with revealed amino acid
            current[pos] = seq_tensor[pos]
        
        total_log_prob += path_log_prob
    
    return total_log_prob / paths  # Average across paths


def generation_likelihood_all_paths(model, sequence, alphabet_size, device):
    """Calculate generation likelihood through all possible paths"""
    seq_len = len(sequence)
    total_log_prob = 0.0
    seq_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
    
    paths = list(itertools.permutations(range(seq_len)))
    for path in paths:
        current = torch.full((seq_len,), alphabet_size, device=device)
        path_log_prob = 0.0
        
        for i in range(seq_len):
            pos = path[i]
            
            with torch.no_grad():
                logits = model(current.unsqueeze(0))[0, pos]
            
            path_log_prob += F.log_softmax(logits, dim=-1)[seq_tensor[pos]].item()
            
            current[pos] = seq_tensor[pos]
            
        total_log_prob += path_log_prob
        
    return total_log_prob / len(paths)
            
            
            




def generation_likelihood_greedy_path(model, sequence, alphabet_size, device):
    """Calculate generation likelihood through greedy unmasking"""
    seq_len = len(sequence)
    total_log_prob = 0.0
    seq_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
    
    current = torch.full((seq_len,), alphabet_size, device=device)
    remaining_pos = list(range(seq_len))
    path_log_prob = 0.0
    
    while remaining_pos:
        # Get model predictions
        with torch.no_grad():
            log_probs = F.log_softmax(model(current.unsqueeze(0))[0], dim=-1)
        
        # Select position with highest log probability of correct amino acid
        max_log_prob = -float('inf')
        for p in remaining_pos:
            if log_probs[p, seq_tensor[p]] > max_log_prob:
                max_log_prob = log_probs[p, seq_tensor[p]]
                pos = p
        remaining_pos.remove(pos)
        
        # Add log probability of correct amino acid
        path_log_prob += max_log_prob.item()
        
        # Update current sequence with revealed amino acid
        current[pos] = seq_tensor[pos]

    return path_log_prob










def single_sequence_scoring(model, energy_params, args, num_samples=100):
    """
    Score individual sequences using model-based and analytic (true) likelihoods.
    Returns:
        Dict containing:
        - 'truth': Ground truth log likelihoods
        - 'scores': Dict of scores for each method
        - 'correlations': Pearson correlations for each method
    """
    device = next(model.parameters()).device
    alphabet_size = energy_params['alphabet_size']
    results = {
        'truth': [],
        'scores': {
            'generation_likelihood': [],
            'generation_likelihood_greedy_path': [],
            'pseudo_likelihood': []
        }
    }

    samples = sample_sequences(energy_params, num_samples)
    
    

    for i in range(num_samples):
        if i % 10 == 0:
            print(i, file=sys.stderr)
        seq = np.array(samples[i])

        # Analytic log likelihood
        log_p = sequence_log_likelihood(seq, energy_params)


        fitness = log_p
        
        results['truth'].append(fitness)

        # Generation likelihood score
        gen_score = generation_likelihood(model, seq, alphabet_size, device, paths=10)
        results['scores']['generation_likelihood'].append(gen_score)

        # Greedy generation likelihood score
        gen_score = generation_likelihood_greedy_path(model, seq, alphabet_size, device)
        results['scores']['generation_likelihood_greedy_path'].append(gen_score)

        # Pseudo likelihood score
        pl_score = pseudo_likelihood(model, seq, alphabet_size, device)
        results['scores']['pseudo_likelihood'].append(pl_score)

    # Calculate correlations and MSE
    truth = np.array(results['truth'])
    correlations = {}
    MSE = {}
    for method in results['scores']:
        pred = np.array(results['scores'][method])
        correlations[method] = np.corrcoef(truth, pred)[0, 1]
        MSE[method] = np.mean((truth - pred) ** 2)

    # Print results
    print("SingleSeqScoring", "seed="+str(args.seed), file=sys.stdout, sep='\t', end='\t', flush=True)
    for method in correlations.keys():
        corr = correlations[method]
        mse = MSE[method]
        print(f"{method}\t{corr:.4f}\t{mse:.4f}", end='\t', file=sys.stdout)
    print(flush=True, file=sys.stdout)

    return {
        'truth': truth,
        'scores': results['scores'],
        'correlations': correlations
    }