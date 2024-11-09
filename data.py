import torch
from torch.utils.data import Dataset
import os
import random
import warnings
from sklearn.model_selection import train_test_split

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
        # Assuming self.x is a list or tensor of shape [num_samples, 12, 6, T]
        if isinstance(self.x, torch.Tensor):
            # If self.x is a tensor, we need to split it into a list of samples
            num_samples = self.x.shape[0]
            self.x = [self.x[i].float() / 255.0 for i in range(num_samples)]
        elif isinstance(self.x, list):
            # If self.x is a list, convert each sample
            self.x = [sample.float() / 255.0 for sample in self.x]
        else:
            raise TypeError("Unsupported data type for self.x")

        # Ensure labels are integers
        self.y = [int(label) for label in self.y]

        # Perform train/val split
        x_train, x_val, y_train, y_val = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )

        if self.split == 'train':
            self.x = x_train
            self.y = y_train
        elif self.split == 'val':
            self.x = x_val
            self.y = y_val
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x1 = self.x[idx]
        y1 = self.y[idx]  # Assuming y1 is composer ID

        # Get a random pair
        if random.random() < 0.5:
            # Same composer
            same = 1
            idx2 = random.choice(self.composer_indices[y1])
            # Ensure idx2 is not the same as idx
            while idx2 == idx and len(self.composer_indices[y1]) > 1:
                idx2 = random.choice(self.composer_indices[y1])
        else:
            # Different composer
            same = 0
            different_composers = list(self.composer_indices.keys())
            different_composers.remove(y1)
            y2 = random.choice(different_composers)
            idx2 = random.choice(self.composer_indices[y2])

        x2 = self.x[idx2]

        # Subsample or pad
        x1 = self._process_sample(x1)
        x2 = self._process_sample(x2)

        return (x1, x2), same

    def _process_sample(self, sample):
        # sample: tensor of shape [12, O, T]
        C, O, T = sample.shape
        if T >= self.t:
            start = random.randint(0, T - self.t)
            sample = sample[:, :, start:start+self.t]
        else:
            pad_size = self.t - T
            pad = torch.zeros((C, O, pad_size))
            sample = torch.cat([sample, pad], dim=2)
        return sample
