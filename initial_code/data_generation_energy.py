import numpy as np
from collections import defaultdict
import itertools
from scipy.special import logsumexp
import time

def seconds_since_last_call():
    now = time.perf_counter()
    if not hasattr(seconds_since_last_call, "last_time"):
        seconds_since_last_call.last_time = now
        return None  # or 0, if you prefer
    elapsed = now - seconds_since_last_call.last_time
    seconds_since_last_call.last_time = now
    return elapsed
seconds_since_last_call()


def generate_energy_distribution(L, alphabet_size, upper_range=3):
    # Generate random effects arrays
    order_1_effects = np.random.uniform(1, upper_range, (L, alphabet_size))
    order_2_effects = np.random.uniform(1, upper_range, (L, L, alphabet_size, alphabet_size))
    
    # Generate all sequences vectorized
    grid = np.meshgrid(*[np.arange(alphabet_size)]*L, indexing='ij')
    sequences = np.stack(grid, axis=-1).reshape(-1, L)  # Shape (N_sequences, L)
    
    # Calculate order 1 contributions
    order1_contrib = order_1_effects[np.arange(L)[:, None], sequences.T].sum(axis=0)
    
    # Calculate order 2 contributions
    order2_contrib = np.zeros(len(sequences))
    for i in range(L):
        for j in range(i+1, L):
            a, b = sequences[:, i], sequences[:, j]
            order2_contrib += order_2_effects[i, j, a, b]
    
    # Compute probabilities
    total_log_likelihood = order1_contrib + order2_contrib
    log_Z = logsumexp(total_log_likelihood)
    probabilities = np.exp(total_log_likelihood - log_Z)
    
    return {
        'order_1_effects': order_1_effects,
        'order_2_effects': order_2_effects,
        'log_Z': log_Z,
        'all_sequences': sequences,
        'probabilities': probabilities,
        'upper_range': float(upper_range),
        'L':L , 
        'alphabet_size':alphabet_size
    }


def sequence_log_likelihood(sequence, energy_dist):

    L = energy_dist['L']

    unnormalized_log_likelihood = 0

    #order 1 effects
    for i in range(L):
        unnormalized_log_likelihood += energy_dist["order_1_effects"][i][sequence[i]]
    
    #order 2 effects
    for i in range(L):
        for j in range(i+1, L):
            unnormalized_log_likelihood += energy_dist["order_2_effects"][i][j][sequence[i]][sequence[j]]
    
    return unnormalized_log_likelihood - energy_dist["log_Z"]



def sample_sequences(energy_dist, N=10000):
    """Exact sampling via inverse transform sampling"""
    # Precomputed data from energy distribution
    sequences = energy_dist['all_sequences']
    probs = energy_dist['probabilities']
    
    # Create CDF and sort both CDF and sequences
    cdf = np.cumsum(probs)
    sorted_indices = np.argsort(cdf)
    sorted_cdf = cdf[sorted_indices]
    sorted_sequences = sequences[sorted_indices]
    
    # Generate uniform samples and find positions
    u = np.random.uniform(0, 1, N)
    idx = np.searchsorted(sorted_cdf, u)
    
    return sorted_sequences[idx]



if __name__ == "__main__":
    np.random.seed(0)

    L=10
    alphabet_size=5

    seconds_since_last_call()
    # Generate chain with higher sparsity for clearer patterns
    ed = generate_energy_distribution(L=L, alphabet_size=alphabet_size, upper_range=3)
    
    print("Seconds to generate", seconds_since_last_call())
    print(ed["order_1_effects"])
    print(ed["order_2_effects"])
    print()
    print()
    print(ed["log_Z"])



    i = 0
    for sequence in itertools.product(range(alphabet_size), repeat=L):
        print(sequence)
        print(sequence_log_likelihood(sequence, ed))
        i += 1
        if i > 100:
            break
    
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    samples = sample_sequences(ed, N=10000)
    for sequence in samples:
        print(sequence)
        print(sequence_log_likelihood(sequence, ed))
        
    
