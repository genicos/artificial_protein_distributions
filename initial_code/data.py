from data_generation_energy import generate_energy_distribution, sample_sequences
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class EnergyDataset(Dataset):
    def __init__(self, energy_params, num_samples=10000, test_prob=0.15, mlm=False,
                 mask_token_ratio=0.8, random_token_ratio=0.1, uniform_masking=False):
        self.sequences = sample_sequences(energy_params, num_samples)
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
    test_prob=0.15,  # Research shows this can be increased for larger models [1][3]
    mlm=False,
    mask_token_ratio=0.8,  # BERT's original 80-10-10 strategy [5][6]
    random_token_ratio=0.1,  # 10% random token replacement [5][6]
    uniform_masking=False
):
    """Create data loaders with configurable BERT-style masking ratios.
    
    Args:
        energy_params: Energy distribution parameters
        batch_size: Number of samples per batch
        train_ratio: Fraction of data for training (0-1)
        num_samples: Total sequences to generate
        test_prob: Probability of testing any token (research suggests 15-40% [1][3])
        mlm: Whether to use masked language modeling
        mask_token_ratio: Fraction of tested tokens to replace with [MASK] [5][6]
        random_token_ratio: Fraction to replace with random tokens [5][6]
    """

    dataset = EnergyDataset(
        energy_params,
        num_samples=num_samples,
        test_prob=test_prob,
        mlm=mlm,
        mask_token_ratio=mask_token_ratio,
        random_token_ratio=random_token_ratio,
        uniform_masking=uniform_masking
    )

    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Special collate function for MLM batches
    def collate_mlm(batch):
        sequences, labels = zip(*batch)
        return torch.stack(sequences), torch.stack(labels)

    # Create loaders
    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_mlm if mlm else torch.stack
        ),
        DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_mlm if mlm else torch.stack
        )
    )
