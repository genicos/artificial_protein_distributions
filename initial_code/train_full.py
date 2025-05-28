import torch
from torch.optim import Adam
from data import get_loaders
from data_generation_energy import generate_energy_distribution, sequence_log_likelihood, sample_sequences
import numpy as np
from model import MaskedLanguageModel
import sys
import torch.nn.functional as F
import argparse
import random
import math
import itertools
import matplotlib.pyplot as plt


def train(energy_params, model_type='mlm', num_epochs=30, num_samples=10000, batch_size=256,
          mlm_embedding_dim=128, mlm_hidden_dim=256, mlm_num_layers=12, mlm_test_prob=0.15,
          mlm_mask_token_ratio=0.8, mlm_random_token_ratio=0.1, gpu=0, uniform_masking=False):
    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
    alphabet_size = energy_params['alphabet_size']

    if model_type == 'markov':
        raise NotImplementedError("nope")
    elif model_type == 'mlm':
        model = MaskedLanguageModel(
            alphabet_size=alphabet_size,
            embedding_dim=mlm_embedding_dim,
            hidden_dim=mlm_hidden_dim,
            num_layers=mlm_num_layers
        ).to(device)
    else:
        raise ValueError("Invalid model_type. Choose 'markov' or 'mlm'.")

    optimizer = Adam(model.parameters(), lr=0.001)
    
    print("Simulating Data", file=sys.stderr)
    # Get data loaders with BERT-style masking
    train_loader, val_loader = get_loaders(
        energy_params=energy_params,
        batch_size=batch_size,
        num_samples=num_samples,
        mlm=(model_type == 'mlm'),
        test_prob=mlm_test_prob,
        mask_token_ratio=mlm_mask_token_ratio,
        random_token_ratio=mlm_random_token_ratio,
        uniform_masking=uniform_masking
    )
    print("Beggining Training", file=sys.stderr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        total_tokens = 0
        test_tokens = 0
        mask_tokens_giv_test_token = 0
        same_token_giv_test_token = 0
        
        for batch in train_loader:
            if model_type == 'markov':
                pass
            elif model_type == 'mlm':
                masked_sequences, labels = batch
                masked_sequences, labels = masked_sequences.to(device), labels.to(device)

                """
                print()
                print(labels[0])
                print(masked_sequences[0])
                print()
                for i in range(len(masked_sequences)):
                    for j in range(len(masked_sequences[0])):
                        total_tokens += 1
                        if labels[i][j] != -1:
                            test_tokens += 1

                            if masked_sequences[i][j] == 5:
                                mask_tokens_giv_test_token += 1
                            elif masked_sequences[i][j]==labels[i][j]:
                                same_token_giv_test_token += 1
                """

                optimizer.zero_grad()
                logits = model(masked_sequences)
                # Reshape for cross entropy: (batch*seq_len, alphabet_size) vs (batch*seq_len)
                loss = F.cross_entropy(
                    logits.view(-1, alphabet_size),
                    labels.view(-1),
                    ignore_index=-1  # Ignore non-masked positions
                )
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'markov':
                    break
                elif model_type == 'mlm':
                    masked_sequences, labels = batch
                    masked_sequences, labels = masked_sequences.to(device), labels.to(device)
                    logits = model(masked_sequences)
                    loss = F.cross_entropy(
                        logits.view(-1, alphabet_size),
                        labels.view(-1),
                        ignore_index=-1
                    )
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}", 
              file=sys.stderr)
    
    return model

def validate_mlm(model, energy_params, args, num_samples=100):
    """Validate MLM by comparing predictions to true distributions"""
    device = next(model.parameters()).device
    alphabet_size = energy_params['alphabet_size']
    L = energy_params['L']
    
    model.eval()
    with torch.no_grad():
        # Track prediction accuracy
        correct = 0
        total_masked = 0
        
        # Track distribution distances
        kl_divergences = []

        sequences = sample_sequences(energy_params, num_samples)
        
        for i in range(num_samples):
            # Generate test sequence
            seq = np.array(sequences[i])
            pos = np.random.choice(len(seq), size=1, replace=False)[0]
            
            # Create masked input
            masked_seq = torch.tensor(seq).clone().unsqueeze(0).to(device)
            masked_seq[0, pos] = alphabet_size  # Use alphabet_size as mask token
            
            # Get model prediction
            logits = model(masked_seq)
            pred_probs = F.softmax(logits[0, pos], dim=-1).cpu().numpy()
            
            # Calculate true distribution
            true_probs = np.zeros(alphabet_size)
            for s in range(alphabet_size):
                temp_seq = seq.copy()
                temp_seq[pos] = s
                true_probs[s] = math.exp(sequence_log_likelihood(temp_seq, energy_params))
            true_probs /= true_probs.sum()
            
            # Calculate accuracy
            pred_token = pred_probs.argmax()
            if pred_token == seq[pos]:
                correct += 1
            total_masked += 1
            
            # Calculate KL divergence
            kl = np.sum(true_probs * np.log(true_probs / (pred_probs + 1e-10)))
            kl_divergences.append(kl)
        
        accuracy = correct / total_masked
        avg_kl = np.mean(kl_divergences)
        
        
        #print(f"Average KL Divergence: {avg_kl:.4f}", file=sys.stdout)
        print("Results", avg_kl, accuracy, args.mlm_test_prob, args.mlm_mask_token_ratio, 
            args.mlm_random_token_ratio, args.num_epochs, args.num_samples, args.seed, file=sys.stdout, sep='\t')















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
        results['truth'].append(log_p)

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





def mutate_sequence(sequence, num_mutations, alphabet_size):
    """Apply random mutations to a sequence"""
    sequence = sequence.copy()
    positions = np.random.choice(len(sequence), size=num_mutations, replace=False)
    for pos in positions:
        current = sequence[pos]
        # Sample different amino acid
        choices = [aa for aa in range(alphabet_size) if aa != current]
        sequence[pos] = np.random.choice(choices)
    return sequence, positions


def assay_scoring(model, energy_params, args, mutations=2, num_samples=100):
    """
    Compare multiple scoring methods against analytic likelihood ratios.
    
    Returns:
        Dict containing:
        - 'truth': Ground truth log likelihood ratios
        - 'scores': Dict of scores for each method
        - 'correlations': Pearson correlations for each method
    """
    device = next(model.parameters()).device
    alphabet_size = energy_params['alphabet_size']
    results = {
        'truth': [],
        'scores': {
            'path_chaining': [],
            'masked_marginal': [],
            'wildtype_context': [],
            'mutant_context': [],
            'pseudo_likelihood': [],
            'generation_likelihood': [],
            'shared_decoding': []
        }
    }

    samples = sample_sequences(energy_params, num_samples)
    
    for i in range(num_samples):
        
        if i % 10 == 0:
            #print(i, file=sys.stderr)
            pass

        # Generate original and mutated sequences
        original = np.array(samples[i])
        mutated, mut_positions = mutate_sequence(original, mutations, alphabet_size)
        
        # Calculate analytic log likelihood ratio
        log_p_original = sequence_log_likelihood(original, energy_params)
        log_p_mutated = sequence_log_likelihood(mutated, energy_params)
        results['truth'].append(log_p_mutated - log_p_original)

        # Generation likelihood scores
        gen_original = generation_likelihood(model, original, alphabet_size, device)
        gen_mutated = generation_likelihood(model, mutated, alphabet_size, device)
        results['scores']['generation_likelihood'].append(gen_mutated - gen_original)
        
        # Calculate all model scores
        results['scores']['path_chaining'].append(
            score_hamming_path(model, original, mutated, mut_positions, alphabet_size, device)
        )
        results['scores']['masked_marginal'].append(
            masked_marginal_score(model, original, mutated, mut_positions, alphabet_size, device)
        )
        results['scores']['wildtype_context'].append(
            masked_marginal_wildtype_context_score(model, original, mutated, mut_positions, alphabet_size, device)
        )
        results['scores']['mutant_context'].append(
            masked_marginal_mutant_context_score(model, original, mutated, mut_positions, alphabet_size, device)
        )
        results['scores']['pseudo_likelihood'].append(
            pseudo_likelihood_score(model, original, mutated, alphabet_size, device)
        )
        results['scores']['shared_decoding'].append(
            shared_decoding_score(model, original, mutated, alphabet_size, device)
        )
    
    # Calculate correlations
    truth = np.array(results['truth'])
    correlations = {}
    MSE = {}
    for method in results['scores']:
        pred = np.array(results['scores'][method])
        #for i in range(len(truth)):
        #    print(method,float(truth[i]), float(pred[i]))
        correlations[method] = np.corrcoef(truth, pred)[0, 1]
        MSE[method] = np.mean((truth - pred) ** 2)

    
    # Print results

    print("EnergyDist", "M="+str(mutations),"seed="+str(args.seed), file=sys.stdout, sep='\t', end='\t', flush=True)
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

# New scoring functions
def masked_marginal_score(model, original, mutated, positions, alphabet_size, device):
    """Score all mutations simultaneously with full masking"""
    masked = torch.tensor(original).to(device)
    masked[positions] = alphabet_size
    
    with torch.no_grad():
        logits = model(masked.unsqueeze(0))[0]
    
    score = 0.0
    for pos in positions:
        score += (logits[pos, mutated[pos]] - logits[pos, original[pos]]).item()
    return score

def masked_marginal_wildtype_context_score(model, original, mutated, positions, alphabet_size, device):
    """Score each mutation in wildtype context"""
    score = 0.0
    for pos in positions:
        # Create wildtype-masked sequence
        masked = torch.tensor(original).to(device)
        masked[pos] = alphabet_size
        
        with torch.no_grad():
            logits = model(masked.unsqueeze(0))[0, pos]
        
        score += (logits[mutated[pos]] - logits[original[pos]]).item()
    return score

def masked_marginal_mutant_context_score(model, original, mutated, positions, alphabet_size, device):
    """Score each mutation in mutant context"""
    current = original.copy()
    score = 0.0
    
    for pos in positions:
        # Create mutant-masked sequence
        masked = torch.tensor(mutated).to(device)
        masked[pos] = alphabet_size
        
        with torch.no_grad():
            logits = model(masked.unsqueeze(0))[0, pos]
        
        score += (logits[mutated[pos]] - logits[original[pos]]).item()
    return score

def pseudo_likelihood_score(model, original, mutated, alphabet_size, device):
    """Calculate pseudo-likelihood ratio"""
    def pl(sequence):
        seq_tensor = torch.tensor(sequence).to(device)
        pl_score = 0.0
        
        for pos in range(len(sequence)):
            masked = seq_tensor.clone()
            masked[pos] = alphabet_size
            
            with torch.no_grad():
                logits = model(masked.unsqueeze(0))[0, pos]
            
            pl_score += logits[sequence[pos]].item()
        return pl_score
    
    return pl(mutated) - pl(original)


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



def shared_decoding_score(model, original, mutated, alphabet_size, device, paths=6):
    """Compare sequences via shared masked decoding from common ancestor"""
    # Identify differing positions
    diff_positions = [i for i, (o, m) in enumerate(zip(original, mutated)) if o != m]
    k = len(diff_positions)
    if k == 0:
        return 0.0  # Identical sequences
    
    # Create common sequence with shared positions revealed
    common_seq = torch.full((len(original),), alphabet_size, device=device)
    for i in range(len(original)):
        if original[i] == mutated[i]:
            common_seq[i] = original[i]
    

    total_paths = math.factorial(k)
    
    permutations = []
    if total_paths <= paths:
        permutations = list(itertools.permutations(diff_positions))
    else:
        # Sample without replacement up to paths
        seen = set()
        while len(permutations) < paths:
            perm = tuple(np.random.permutation(diff_positions).tolist())
            if perm not in seen:
                seen.add(perm)
                permutations.append(perm)
    
    paths = len(permutations)
    
    total_score = 0.0
    for perm in permutations:
        
        # Decode towards original
        current_o = common_seq.clone()
        log_prob_o = 0.0
        
        # Decode towards mutated
        current_m = common_seq.clone()
        log_prob_m = 0.0
        
        for pos in perm:
            # Decode original direction
            masked_o = current_o.clone()
            masked_o[pos] = alphabet_size
            with torch.no_grad():
                logits_o = model(masked_o.unsqueeze(0))[0, pos]
            log_prob_o += F.log_softmax(logits_o, dim=-1)[original[pos]].item()
            current_o[pos] = original[pos]
            
            # Decode mutated direction
            masked_m = current_m.clone()
            masked_m[pos] = alphabet_size
            with torch.no_grad():
                logits_m = model(masked_m.unsqueeze(0))[0, pos]
            log_prob_m += F.log_softmax(logits_m, dim=-1)[mutated[pos]].item()
            current_m[pos] = mutated[pos]
        
        total_score += (log_prob_m - log_prob_o)
    
    return total_score / paths







def score_single_mutation(model, original, mutated, pos, alphabet_size, device):
    """Score single mutation by masking the differing position"""
    # Create masked sequence
    masked = torch.tensor(original).to(device)
    masked[pos] = alphabet_size  # Mask token
    
    # Get model predictions
    with torch.no_grad():
        logits = model(masked.unsqueeze(0))
        log_probs = F.log_softmax(logits[0, pos], dim=-1)
    
    # Calculate log ratio
    return (log_probs[mutated[pos]] - log_probs[original[pos]]).cpu().item()



def score_hamming_path(model, original, mutated, positions, alphabet_size, device, paths=6):
    """Score multiple mutations by sampling multiple Hamming paths"""
    k = len(positions)
    if k == 0:
        return 0.0  # No mutations to score
    
    # Calculate total possible paths (k!)
    total_paths = math.factorial(k)
    permutations = []
    
    if total_paths <= paths:
        # Use all possible paths
        permutations = list(itertools.permutations(positions))
    else:
        # Sample random unique paths
        seen = set()
        while len(permutations) < paths:
            perm = tuple(np.random.permutation(positions).tolist())
            if perm not in seen:
                seen.add(perm)
                permutations.append(perm)
    
    total_score = 0.0
    for perm in permutations:
        current = original.copy()
        path_score = 0.0
        
        for pos in perm:
            intermediate = current.copy()
            intermediate[pos] = mutated[pos]
            
            # Score this mutation step
            step_score = score_single_mutation(
                model, current, intermediate, pos, 
                alphabet_size, device
            )
            path_score += step_score
            current = intermediate
            
        total_score += path_score
    
    return total_score / len(permutations)  # Average across paths






import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, gaussian_kde
import os

def format_tick_labels(sorted_aas, original_aa):
    # Return list of labels, bolding the original
    return [f"{aa}" for aa in sorted_aas]

def visualize_two_site_spectrum(model, energy_params, images_dir=None):
    alphabet_size = energy_params['alphabet_size']
    all_two_site_amino_acid_combinations = list(itertools.product(range(alphabet_size), repeat=2))

    original_sequence = sample_sequences(energy_params, 1)[0]
    L = len(original_sequence)
    device = next(model.parameters()).device

    # Randomly pick two positions
    pos1 = np.random.randint(0, L)
    pos2 = np.random.randint(0, L)
    while pos2 == pos1:
        pos2 = np.random.randint(0, L)

    original_aa1 = original_sequence[pos1]
    original_aa2 = original_sequence[pos2]

    # Collect all data into a DataFrame
    data = []
    baseline_log_p = sequence_log_likelihood(original_sequence, energy_params)
    baseline_generation_likelihood = generation_likelihood(model, original_sequence, alphabet_size, device)
    baseline_generation_likelihood_greedy_path = generation_likelihood_greedy_path(model, original_sequence, alphabet_size, device)
    baseline_pseudo_likelihood = pseudo_likelihood(model, original_sequence, alphabet_size, device)

    for aa1, aa2 in all_two_site_amino_acid_combinations:
        mutant = original_sequence.copy()
        mutant[pos1] = aa1
        mutant[pos2] = aa2

        log_p_mutant = sequence_log_likelihood(mutant, energy_params)
        truth = log_p_mutant - baseline_log_p
        gen_ll = generation_likelihood(model, mutant, alphabet_size, device) - baseline_generation_likelihood
        gen_ll_greedy = generation_likelihood_greedy_path(model, mutant, alphabet_size, device) - baseline_generation_likelihood_greedy_path
        pl = pseudo_likelihood(model, mutant, alphabet_size, device) - baseline_pseudo_likelihood
        pc = score_hamming_path(model, original_sequence, mutant, [pos1, pos2], alphabet_size, device)
        mm = masked_marginal_score(model, original_sequence, mutant, [pos1, pos2], alphabet_size, device)
        wtmm = masked_marginal_wildtype_context_score(model, original_sequence, mutant, [pos1, pos2], alphabet_size, device)
        sd = shared_decoding_score(model, original_sequence, mutant, alphabet_size, device)

        data.append({
            'mut_aa1': aa1,
            'mut_aa2': aa2,
            'truth': truth,
            'SD': sd,
            'MM': mm,
            'WTMM': wtmm,
            'PC': pc,
            'generation_likelihood': gen_ll,
            'generation_likelihood_greedy_path': gen_ll_greedy,
            'pseudo_likelihood': pl
        })

    df = pd.DataFrame(data)

    # Sorting
    avg_score_pos1 = df.groupby('mut_aa1')['truth'].mean()
    avg_score_pos2 = df.groupby('mut_aa2')['truth'].mean()
    sorted_aa_pos1 = avg_score_pos1.sort_values().index.tolist()
    sorted_aa_pos2 = avg_score_pos2.sort_values().index.tolist()

    # Scores to plot
    scores = [
        ('truth', 'Analytic ΔlogP'),
        ('SD', 'Shared Decoding'),
        ('MM', 'Masked Marginal'),
        ('WTMM', 'Wildtype Context'),
        ('PC', 'Path Chaining'),
        ('generation_likelihood', 'Gen Likelihood'),
        ('generation_likelihood_greedy_path', 'Greedy Gen Likelihood'),
        ('pseudo_likelihood', 'Pseudo Likelihood')
    ]
    fig, axes = plt.subplots(3, len(scores), figsize=(5*len(scores), 15), gridspec_kw={'height_ratios': [3, 2, 0.5]})

    for i, (score_key, score_label) in enumerate(scores):
        # Row 1: Heatmap
        pivot = df.pivot_table(index='mut_aa1', columns='mut_aa2', values=score_key, aggfunc='first')
        pivot = pivot.reindex(index=sorted_aa_pos1, columns=sorted_aa_pos2)
        sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': score_label}, ax=axes[0, i], square=True)
        axes[0, i].set_title(score_label)
        axes[0, i].set_xlabel('Mutated AA at Position 2')
        axes[0, i].set_ylabel('Mutated AA at Position 1')
        axes[0, i].set_xticks([x + 0.5 for x in range(len(sorted_aa_pos2))])
        axes[0, i].set_yticks([y + 0.5 for y in range(len(sorted_aa_pos1))])
        tick_labels_x = format_tick_labels(sorted_aa_pos2, original_aa2)
        tick_labels_y = format_tick_labels(sorted_aa_pos1, original_aa1)
        axes[0, i].set_xticklabels(tick_labels_x, rotation=90)
        axes[0, i].set_yticklabels(tick_labels_y, rotation=0)
        # Bold original amino acids
        for j, label in enumerate(axes[0, i].get_xticklabels()):
            if sorted_aa_pos2[j] == original_aa2:
                label.set_fontweight('bold')
        for j, label in enumerate(axes[0, i].get_yticklabels()):
            if sorted_aa_pos1[j] == original_aa1:
                label.set_fontweight('bold')

        # Row 2: Density or scatter
        if score_key == 'truth':
            x_vals = df[score_key].values
            try:
                kde = gaussian_kde(x_vals)
                x_grid = np.linspace(min(x_vals), max(x_vals), 200)
                density = kde(x_grid)
            except Exception:
                jitter = np.random.normal(0, 1e-10, size=len(x_vals))
                kde = gaussian_kde(x_vals + jitter)
                x_grid = np.linspace(min(x_vals), max(x_vals), 200)
                density = kde(x_grid)
            axes[1, i].plot(x_grid, density, color='blue')
            axes[1, i].fill_between(x_grid, density, alpha=0.3, color='blue')
            axes[1, i].set_title('Density of Analytic ΔlogP')
            axes[1, i].set_xlabel('Analytic ΔlogP')
            axes[1, i].set_ylabel('Density')
        else:
            x_vals = df['truth'].values
            y_vals = df[score_key].values
            axes[1, i].scatter(x_vals, y_vals, alpha=0.6, s=10)
            axes[1, i].set_title(f'Scatter: {score_label} vs Analytic ΔlogP')
            axes[1, i].set_xlabel('Analytic ΔlogP')
            axes[1, i].set_ylabel(score_label)

        # Row 3: Spearman correlation
        if score_key == 'truth':
            corr = 1.0
            axes[2, i].text(0.5, 0.5, f'Spearman r = {corr:.4f}', ha='center', va='center', fontsize=12)
            axes[2, i].axis('off')
        else:
            x_vals = df['truth'].values
            y_vals = df[score_key].values
            corr, pval = spearmanr(x_vals, y_vals)
            top_n = 10
            top_n_subset = df.nlargest(top_n, score_key)
            avg_truth_top_n = top_n_subset['truth'].mean()
            text = f'Spearman r = {corr:.4f}\\nAvg Analytic ΔlogP (top {top_n}) = {avg_truth_top_n:.4f}'
            axes[2, i].text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
            axes[2, i].axis('off')

    plt.suptitle(f"{original_sequence} \\nPositions {pos1}-{pos2}\\nSorted by: Analytic ΔlogP")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if images_dir is None:
        plt.savefig('two_site_spectrum.png')
    else:
        os.makedirs(images_dir, exist_ok=True)
        plt.savefig(os.path.join(images_dir, f"two_site_spectrum_{original_sequence}_{pos1}-{pos2}.png"))
    plt.close()
    
















def main():
    
    
    parser = argparse.ArgumentParser(description="Train Masked Language Model on Energy Distribution")
    parser.add_argument('--model_type', type=str, default='mlm', choices=['mlm'],
                      help='Type of model to train')
    parser.add_argument('--mlm_embedding_dim', type=int, default=128,
                      help='Embedding dimension for MLM')
    parser.add_argument('--mlm_hidden_dim', type=int, default=256,
                      help='Hidden dimension for MLM')
    parser.add_argument('--mlm_num_layers', type=int, default=12,
                      help='Number of layers for MLM')
    
    parser.add_argument('--uniform_masking', action='store_true',
                      help='Use uniform random number of masks (0 to L)')
    
    parser.add_argument('--mlm_test_prob', type=float, default=0.15,
                      help='Probability of masking a token')
    parser.add_argument('--mlm_mask_token_ratio', type=float, default=1.0,
                      help='Fraction of masked tokens to replace with [MASK]')
    parser.add_argument('--mlm_random_token_ratio', type=float, default=0.0,
                      help='Fraction of masked tokens to replace randomly')
    
    parser.add_argument('--num_epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=10000,
                      help='Number of samples to train on')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed')
    
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU num to use')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Generate Energy dist
    L = 5
    alphabet_size = 10
    energy_params = generate_energy_distribution(
        L=L,
        alphabet_size=alphabet_size,
        upper_range=3
    )

    validation_set = sample_sequences(energy_params, 100)
    validation_set_log_p = [sequence_log_likelihood(seq, energy_params) for seq in validation_set]
    
    # Train model
    model = train(energy_params, model_type=args.model_type,
                 num_epochs=args.num_epochs,
                 num_samples=args.num_samples,
                 mlm_embedding_dim=args.mlm_embedding_dim,
                 mlm_hidden_dim=args.mlm_hidden_dim,
                 mlm_num_layers=args.mlm_num_layers,
                 mlm_test_prob=args.mlm_test_prob,
                 mlm_mask_token_ratio=args.mlm_mask_token_ratio,
                 mlm_random_token_ratio=args.mlm_random_token_ratio,
                 gpu=args.gpu,
                 uniform_masking=args.uniform_masking)

    
    # Validation
    if args.model_type == 'markov':
        exit()
    else:

        for i in range(100):
            visualize_two_site_spectrum(model, energy_params, images_dir=f"images")

        exit()

        #validate_mlm(model, energy_params, args, num_samples=2)
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
        plt.scatter(results['truth'], results['scores']['generation_likelihood'], label='Generation Likelihood', s=1)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')  # Dashed y = x line
        plt.xlabel('Truth')
        plt.ylabel('Score')
        plt.legend()

        # Subplot 2
        plt.subplot(1, 3, 2)
        plt.scatter(results['truth'], results['scores']['generation_likelihood_greedy_path'], label='Greedy Generation Likelihood', s=1)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        plt.xlabel('Truth')
        plt.ylabel('Score')
        plt.legend()

        # Subplot 3
        plt.subplot(1, 3, 3)
        plt.scatter(results['truth'], results['scores']['pseudo_likelihood'], label='Pseudo Likelihood', s=1)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        plt.xlabel('Truth')
        plt.ylabel('Score')
        plt.legend()

        # Save figure
        plt.savefig('single_sequence_scoring.png')

        #for m in range(L):
            #assay_scoring(model, energy_params, args, mutations=m+1, num_samples=1000)


if __name__ == "__main__":
    main()
