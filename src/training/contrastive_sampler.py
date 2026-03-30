"""
Contrastive Batch Sampler for cross-optimization pairs.

Ensures each batch contains multiple O0/O2 pairs of the same function,
providing dense positive pairs for NT-Xent contrastive loss.

Strategy: Each batch has `pair_count` anchor-positive pairs (same function,
different optimization), plus random fills to reach `batch_size`.
"""
import random
from collections import defaultdict
from torch.utils.data import Sampler


class ContrastiveBatchSampler(Sampler):
    """Yields batches where each contains ~pair_count positive pairs.

    A "pair" is two dataset indices that share the same function name
    but come from different binaries (typically O0 vs O2).
    """

    def __init__(self, dataset, train_indices, batch_size=128, pair_count=32):
        self.batch_size = batch_size
        self.pair_count = pair_count
        self.train_indices = list(train_indices)

        # Build name → indices mapping (only within train split)
        self.name_to_indices = defaultdict(list)
        for idx in self.train_indices:
            sample = dataset.samples[idx]
            self.name_to_indices[sample['name']].append(idx)

        # Keep only names with 2+ samples (actual pairs)
        self.pair_names = [
            name for name, indices in self.name_to_indices.items()
            if len(indices) >= 2
        ]
        self.non_pair_indices = [
            idx for idx in self.train_indices
            if len(self.name_to_indices[dataset.samples[idx]['name']]) < 2
        ]

    def __iter__(self):
        # Shuffle pair names and non-pair indices each epoch
        pair_names = self.pair_names.copy()
        random.shuffle(pair_names)
        non_pair = self.non_pair_indices.copy()
        random.shuffle(non_pair)

        all_indices = self.train_indices.copy()
        random.shuffle(all_indices)

        name_idx = 0
        fill_idx = 0

        num_batches = len(self.train_indices) // self.batch_size

        for _ in range(num_batches):
            batch = []

            # Add pair_count pairs (2 * pair_count indices)
            pairs_added = 0
            while pairs_added < self.pair_count and name_idx < len(pair_names):
                name = pair_names[name_idx % len(pair_names)]
                name_idx += 1
                indices = self.name_to_indices[name]
                if len(indices) >= 2:
                    # Pick 2 random samples with this name
                    pair = random.sample(indices, 2)
                    batch.extend(pair)
                    pairs_added += 1

            # Fill rest with random samples
            remaining = self.batch_size - len(batch)
            while remaining > 0 and fill_idx < len(all_indices):
                idx = all_indices[fill_idx]
                fill_idx += 1
                if idx not in batch:  # avoid duplicates
                    batch.append(idx)
                    remaining -= 1

            if len(batch) >= 2:
                yield batch[:self.batch_size]

    def __len__(self):
        return len(self.train_indices) // self.batch_size
