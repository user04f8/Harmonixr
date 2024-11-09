import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

class MIDIDataset(Dataset):
    def __init__(self, data_dir, t, split='train'):
        self.t = t
        self.data_dir = data_dir
        self.split = split
        
        # Load data
        self.x = torch.load(os.path.join(data_dir, f'x_{split}.pt'))
        self.y = torch.load(os.path.join(data_dir, f'y_{split}.pt'))
        
        # Build index by composer
        self.composer_indices = {}
        for idx, label in enumerate(self.y):
            composer = label['composer']
            if composer not in self.composer_indices:
                self.composer_indices[composer] = []
            self.composer_indices[composer].append(idx)
        
        self.indices = list(range(len(self.x)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x1 = self.x[idx]
        y1 = self.y[idx]['composer']
        
        # Get a random pair
        if random.random() < 0.5:
            # Same composer
            same = 1
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
        C, O, T = sample.shape
        if T >= self.t:
            start = random.randint(0, T - self.t)
            sample = sample[:, :, start:start+self.t]
        else:
            pad_size = self.t - T
            pad = torch.zeros((C, O, pad_size))
            sample = torch.cat([sample, pad], dim=2)
        return sample
