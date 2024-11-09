# data.py
import torch
from torch.utils.data import Dataset
import os
import random
import warnings
from sklearn.model_selection import train_test_split
from itertools import combinations

class MIDIDataset(Dataset):
    def __init__(self, data_dir, t, split='train', test_size=0.2, random_state=42):
        self.t = t
        self.data_dir = data_dir
        self.split = split

        # Load data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            # Load the data
            self.x = torch.load(os.path.join(data_dir, 'midi_pieces.pt'))
            self.y = torch.load(os.path.join(data_dir, 'composer_vector.pt'))

        # Convert data from uint8 to float and normalize
        if isinstance(self.x, torch.Tensor):
            num_samples = self.x.shape[0]
            self.x = [self.x[i].float() / 255.0 for i in range(num_samples)]
        elif isinstance(self.x, list):
            self.x = [sample.float() / 255.0 for sample in self.x]
        else:
            raise TypeError("Unsupported data type for self.x")

        # Ensure labels are integers
        self.y = [int(label.item()) for label in self.y]

        # Perform train/val split
        x_train, x_val, y_train, y_val = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )

        if self.split == 'train':
            self.x = x_train
            self.y = y_train
            self.is_train = True
        elif self.split == 'val':
            self.x = x_val
            self.y = y_val
            self.is_train = False
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Build index by composer
        self.composer_indices = {}
        for idx, label in enumerate(self.y):
            composer = label  # Assuming label is composer ID
            if composer not in self.composer_indices:
                self.composer_indices[composer] = []
            self.composer_indices[composer].append(idx)

        self.indices = list(range(len(self.x)))

        if not self.is_train:
            # Pre-generate pairs for validation
            self.pairs, self.pair_labels = self._generate_validation_pairs(random_state)

    def _generate_validation_pairs(self, random_state):
        random.seed(random_state)  # Set seed for reproducibility
        pairs = []
        pair_labels = []
        MAX_SIMILAR_PAIRS_PER_COMPOSER = 5
        MAX_DISSIMILAR_PAIRS = 5 * len(self.y)  # NOTE TODO test or separate metrics

        # Similar pairs
        for composer, idxs in self.composer_indices.items():
            if len(idxs) < 2:
                continue
            # Generate all combinations of similar pairs
            similar_pairs = list(combinations(idxs, 2))
            random.shuffle(similar_pairs)
            # Select up to MAX_SIMILAR_PAIRS_PER_COMPOSER pairs
            selected_pairs = similar_pairs[:MAX_SIMILAR_PAIRS_PER_COMPOSER]
            for idx1, idx2 in selected_pairs:
                pairs.append((idx1, idx2))
                pair_labels.append(1)

        # Dissimilar pairs
        dissimilar_pairs = []
        composers = list(self.composer_indices.keys())
        composer_pairs = list(combinations(composers, 2))
        random.shuffle(composer_pairs)
        # Select up to MAX_DISSIMILAR_PAIRS composer pairs
        selected_composer_pairs = composer_pairs[:MAX_DISSIMILAR_PAIRS]
        for c1, c2 in selected_composer_pairs:
            idx1 = random.choice(self.composer_indices[c1])
            idx2 = random.choice(self.composer_indices[c2])
            pairs.append((idx1, idx2))
            pair_labels.append(0)

        return pairs, pair_labels

    def __len__(self):
        if self.is_train:
            return len(self.indices)
        else:
            return len(self.pairs)

    def __getitem__(self, idx):
        if self.is_train:
            x1 = self.x[idx]
            y1 = self.y[idx]
            # Randomly decide whether to create a similar or dissimilar pair
            if random.random() < 0.5 and len(self.composer_indices[y1]) > 1:
                # Similar pair
                same = 1
                idx2 = random.choice(self.composer_indices[y1])
                while idx2 == idx:
                    idx2 = random.choice(self.composer_indices[y1])
            else:
                # Dissimilar pair
                same = 0
                different_composers = list(self.composer_indices.keys())
                different_composers.remove(y1)
                y2 = random.choice(different_composers)
                idx2 = random.choice(self.composer_indices[y2])

            x2 = self.x[idx2]
        else:
            idx1, idx2 = self.pairs[idx]
            same = self.pair_labels[idx]
            x1 = self.x[idx1]
            x2 = self.x[idx2]

        # Subsample or pad
        x1 = self._process_sample(x1)
        x2 = self._process_sample(x2)

        return (x1, x2), same

    def _process_sample(self, sample):
        # sample: tensor of shape [12, O, T]
        C, O, T = sample.shape

        if T >= self.t:
            if self.is_train:
                # Random crop for training
                start = random.randint(0, T - self.t)
            else:
                # Center crop for validation
                start = (T - self.t) // 2
            sample = sample[:, :, start:start+self.t]
        else:
            # Random padding when sample is shorter than self.t
            pad_size = self.t - T
            if self.is_train:
                pad_left = random.randint(0, pad_size)
            else:
                pad_left = pad_size // 2  # Center padding for validation
            pad_right = pad_size - pad_left
            pad_left_tensor = torch.zeros((C, O, pad_left))
            pad_right_tensor = torch.zeros((C, O, pad_right))
            sample = torch.cat([pad_left_tensor, sample, pad_right_tensor], dim=2)
        return sample