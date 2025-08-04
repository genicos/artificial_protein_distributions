from data_generation_energy import generate_energy_distribution, sample_sequences
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, energy_params, sequences=None, num_samples=10000, test_prob=0.15, mlm=True,
                 mask_token_ratio=0.8, random_token_ratio=0.1, uniform_masking=False):
        self.sequences = sequences if sequences is not None else sample_sequences(energy_params, num_samples)
        self.alphabet_size = energy_params['alphabet_size']
        self.test_prob = test_prob
        self.mlm = mlm
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio
        self.unchanged_ratio = 1 - mask_token_ratio - random_token_ratio
        self.uniform_masking = uniform_masking
        
        assert 0 <= mask_token_ratio <= 1, "Mask token ratio must be between 0-1"
        assert 0 <= random_token_ratio <= 1, "Random token ratio must be between 0-1"
        assert self.unchanged_ratio >= -1e-9, "Token ratios sum cannot exceed 1"

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.mlm:

            masked_sequence = sequence.clone()
            labels = sequence.clone()

            if self.uniform_masking:
                test_prob = np.random.random()
            else:
                test_prob = self.test_prob
            
            # Create initial mask using test_prob
            test = torch.rand(len(sequence)) < test_prob
            test_indices = torch.where(test)[0]
            num_tested = len(test_indices)

            if num_tested > 0:
                # Create probability distribution for mask/random/unchanged
                probs = torch.tensor([
                    self.mask_token_ratio,
                    self.random_token_ratio,
                    self.unchanged_ratio
                ])

                # Sample operation types for each tested position
                operations = torch.multinomial(
                    probs, 
                    num_tested, 
                    replacement=True,
                    generator=torch.Generator().manual_seed(torch.seed())  # For reproducibility
                )

                # Split indices based on sampled operations
                mask_indices = test_indices[operations == 0]
                random_indices = test_indices[operations == 1]
                unchanged_indices = test_indices[operations == 2]

                # Apply transformations
                masked_sequence[mask_indices] = self.alphabet_size  # [MASK]
                masked_sequence[random_indices] = torch.randint(
                    0, self.alphabet_size, (len(random_indices),)
                )  # Random tokens
                # unchanged_indices remain original values

            # Labels: original tokens for all test positions, -1 elsewhere
            labels[~test] = -1  # Only compute loss on test positions

            return masked_sequence, labels
        
        return sequence



def get_loaders(
    energy_params,
    batch_size=256,
    train_ratio=0.9,
    num_samples=10000,
    test_prob=0.15,  # Research shows this can be increased for larger models 
    mlm=True,
    mask_token_ratio=0.8,  # BERT's original 80-10-10 strategy 
    random_token_ratio=0.1,  # 10% random token replacement 
    uniform_masking=True,
    generator=None  # Add generator parameter for reproducibility
):
    """Create data loaders with configurable BERT-style masking ratios.
    
    Args:
        energy_params: Energy distribution parameters
        batch_size: Number of samples per batch
        train_ratio: Fraction of data for training (0-1)
        num_samples: Total sequences to generate
        test_prob: Probability of testing any token 
        mlm: Whether to use masked language modeling
        mask_token_ratio: Fraction of tested tokens to replace with [MASK] 
        random_token_ratio: Fraction to replace with random tokens 
        generator: PyTorch generator for reproducible shuffling
    """
    # Sample all sequences first
    all_sequences = sample_sequences(energy_params, num_samples)
    
    # Group identical sequences together
    sequence_groups = {}
    for seq in all_sequences:
        seq_tuple = tuple(seq)
        if seq_tuple not in sequence_groups:
            sequence_groups[seq_tuple] = []
        sequence_groups[seq_tuple].append(seq)
    
    # Convert to list of (unique_sequence, all_copies) pairs
    groups = list(sequence_groups.items())
    
    # Randomly shuffle groups using generator if provided
    if generator is not None:
        # Convert to tensor for shuffling with generator
        indices = torch.randperm(len(groups), generator=generator)
        groups = [groups[i] for i in indices]
    else:
        np.random.shuffle(groups)
    
    # Split groups into train/test based on train_ratio
    split_idx = int(len(groups) * train_ratio)
    train_groups = groups[:split_idx]
    test_groups = groups[split_idx:]
    
    # Expand groups back into sequences
    train_sequences = [seq for group in train_groups for seq in group[1]]
    test_sequences = [seq for group in test_groups for seq in group[1]]
    
    # Create datasets
    test_dataset = EnergyDataset(
        energy_params,
        sequences=test_sequences,
        test_prob=test_prob,
        mlm=mlm,
        mask_token_ratio=mask_token_ratio,
        random_token_ratio=random_token_ratio,
        uniform_masking=uniform_masking
    )
    
    train_dataset = EnergyDataset(
        energy_params,
        sequences=train_sequences,
        test_prob=test_prob,
        mlm=mlm,
        mask_token_ratio=mask_token_ratio,
        random_token_ratio=random_token_ratio,
        uniform_masking=uniform_masking
    )

    # Special collate function for MLM batches
    def collate_mlm(batch):
        sequences, labels = zip(*batch)
        return torch.stack(sequences), torch.stack(labels)

    # Create loaders with generator for reproducible shuffling
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_mlm if mlm else torch.stack
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_mlm if mlm else torch.stack
        )
    )
