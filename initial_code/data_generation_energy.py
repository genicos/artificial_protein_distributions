import numpy as np
from collections import defaultdict
import itertools
from scipy.special import logsumexp
import time
import pickle
import zlib
from scipy.stats import norm

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


#Fitness is the log likelihood of the sequence + noise
# This noise is deterministic with seed + sequence 
def get_noised_fitness(sequence, energy_dist, noise_level=0.1):
    fitness = sequence_log_likelihood(sequence, energy_dist)
    if noise_level == 0:
        return fitness
    sequence_hash_between_0_and_1 = zlib.crc32(sequence.tobytes()) / 2**32
    noise = norm.ppf(sequence_hash_between_0_and_1, 0, noise_level)
    return fitness + noise


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
    # Use right=True to handle edge cases better
    idx = np.searchsorted(sorted_cdf, u, side='right')
    # Ensure we don't go out of bounds
    idx = np.clip(idx, 0, len(sorted_sequences) - 1)
    
    return sorted_sequences[idx]



def sort_sequences(sequences):
    sort_indices = np.lexsort(sequences[:, ::-1].T)
    return sequences[sort_indices]



def save_energy_distribution(energy_dist, filename):
    with open(filename, 'wb') as f:
        pickle.dump(energy_dist, f)

def load_energy_distribution(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



def see_all_sequences_compare(energy_dist, number_of_samples=100000):

    L = energy_dist['L']
    alphabet_size = energy_dist['alphabet_size']

    samples = sample_sequences(energy_dist, N=number_of_samples)

    #sort sequences properly
    samples = sort_sequences(samples)
    

    #list through all possible sequences
    total_appearance_count = 0
    total_expected_count_stored_probabilities = 0
    total_expected_count_stored_log_likelihood = 0
    for sequence in itertools.product(range(alphabet_size), repeat=L):
        print(sequence, end=" ")
        #np array 
        seq_idx = np.where(np.all(energy_dist["all_sequences"] == sequence, axis=1))[0]
        print(seq_idx)

        appearance_count = np.sum(np.sum(samples == sequence, axis=1) == L)
        
        expected_count_stored_probabilities = energy_dist["probabilities"][seq_idx] * number_of_samples
        expected_count_stored_log_likelihood = np.exp(sequence_log_likelihood(sequence, energy_dist)) * number_of_samples
        print(f"Appearance count: {appearance_count}, Expected count (stored probabilities): {expected_count_stored_probabilities}, Expected count (stored log likelihood): {expected_count_stored_log_likelihood}")
        total_appearance_count += appearance_count
        total_expected_count_stored_probabilities += expected_count_stored_probabilities
        total_expected_count_stored_log_likelihood += expected_count_stored_log_likelihood
        
    print(f"Total appearance count: {total_appearance_count}")
    print(f"Total expected count (stored probabilities): {total_expected_count_stored_probabilities}")
    print(f"Total expected count (stored log likelihood): {total_expected_count_stored_log_likelihood}")

if __name__ == "__main__":
    np.random.seed(0)

    L=5
    alphabet_size=5

    seconds_since_last_call()
    # Generate chain with higher sparsity for clearer patterns
    ed = generate_energy_distribution(L=L, alphabet_size=alphabet_size, upper_range=3)
    #usually upper range is 3
    
    print("Seconds to generate", seconds_since_last_call())
    print(ed["order_1_effects"])
    print(ed["order_2_effects"])
    print()
    print()
    print(ed["log_Z"])

    #print("\nTesting sampling coherence...") #LLM nonsense dont listen to it
    #test_results = test_sampling_coherence(ed)#
    #print("Test results:", test_results)

    print("Checking noise consistency...")
    """
    print(sequence_log_likelihood(np.array([0,0,0,0,0]), ed))
    print(sequence_log_likelihood(np.array([0,0,0,0,1]), ed))
    print(get_noised_fitness(np.array([0,0,0,0,0]), ed, noise_level=0.1))
    print(get_noised_fitness(np.array([0,0,0,0,0]), ed, noise_level=0.1))
    print(get_noised_fitness(np.array([0,0,0,0,1]), ed, noise_level=0.1))
    print(get_noised_fitness(np.array([0,0,0,0,1]), ed, noise_level=0.1))
    """
    print("Checking noise consistency done")

    print("Checking all sequences")
    see_all_sequences_compare(ed, number_of_samples=100000)
    print("Checking all sequences done")

    samples = sample_sequences(ed, N=100000)
    num_in_range1 = 0
    num_in_range2 = 0
    for sequence in samples:
        #print(sequence)
        #print(sequence_log_likelihood(sequence, ed))
        if sequence_log_likelihood(sequence, ed) > -10.5 and sequence_log_likelihood(sequence, ed) < -10:
            num_in_range1 += 1
        if sequence_log_likelihood(sequence, ed) > -12.5 and sequence_log_likelihood(sequence, ed) < -12:
            num_in_range2 += 1
    
    print("BBBBBBB")
    
    print(num_in_range1)
    print(num_in_range2)
    
    
    
