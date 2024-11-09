import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os
import warnings
import random

def load_model(ckpt_path, device=None):
    from model import MIDIClassifier
    model = MIDIClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

def load_composer_mapping(mapping_file='data/composer_mapping.txt'):
    composer_mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            index, name = line.strip().split(': ', 1)
            composer_mapping[int(index)] = name
    return composer_mapping

class MIDISingleDataset(Dataset):
    def __init__(self, data_dir, t, n_composers=None):
        self.t = t
        self.data_dir = data_dir

        # Load data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
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

        # Filter composers if n_composers is specified
        if n_composers is not None:
            # Get composer indices to include
            composers_to_include = sorted(set(self.y))[:n_composers]
            filtered_indices = [i for i, composer_idx in enumerate(self.y) if composer_idx in composers_to_include]
            self.x = [self.x[i] for i in filtered_indices]
            self.y = [self.y[i] for i in filtered_indices]

        self.indices = list(range(len(self.x)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.x[idx]
        composer_idx = self.y[idx]
        sample = self._process_sample(sample)
        return sample, composer_idx

    def _process_sample(self, sample):
        # sample: tensor of shape [12, O, T]
        C, O, T = sample.shape

        if T >= self.t:
            # Center crop
            start = (T - self.t) // 2
            sample = sample[:, :, start:start+self.t]
        else:
            # Pad to length self.t
            pad_size = self.t - T
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            pad_left_tensor = torch.zeros((C, O, pad_left))
            pad_right_tensor = torch.zeros((C, O, pad_right))
            sample = torch.cat([pad_left_tensor, sample, pad_right_tensor], dim=2)
        return sample

def extract_embeddings(model, dataset, device):
    embeddings = []
    labels = []
    composer_names = []
    composer_mapping = load_composer_mapping()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        for samples, composer_indices in tqdm(dataloader, desc='Extracting embeddings'):
            samples = samples.to(device)
            embs = model.forward_one(samples).cpu().numpy()
            embeddings.append(embs)
            labels.extend(composer_indices.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    composer_names = [composer_mapping.get(idx, 'Unknown') for idx in labels]
    return embeddings, labels, composer_names

def visualize_tsne(embeddings, composer_names):
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': embeddings_tsne[:, 0],
        'y': embeddings_tsne[:, 1],
        'z': embeddings_tsne[:, 2],
        'composer': composer_names
    })

    fig = px.scatter_3d(df, x='x', y='y', z='z', color='composer', hover_name='composer')
    fig.update_layout(title='3D t-SNE of Embeddings by Composer')
    fig.show()

def plot_distance_histogram(embeddings, labels, num_pairs=1000):
    """
    Plot a histogram of pairwise distances for random pairs.

    Args:
        embeddings (np.ndarray): Precomputed embeddings.
        labels (np.ndarray): Corresponding composer labels.
        num_pairs (int): Number of random pairs to sample.
    """
    distances = []
    pair_labels = []

    # Build composer indices
    composer_indices = {}
    for idx, composer_idx in enumerate(labels):
        if composer_idx not in composer_indices:
            composer_indices[composer_idx] = []
        composer_indices[composer_idx].append(idx)

    for _ in tqdm(range(num_pairs), desc='Computing distances'):
        if random.random() < 0.5:
            # Similar pair
            same = 1
            composer = random.choice([k for k, v in composer_indices.items() if len(v) > 1])
            idx1, idx2 = random.sample(composer_indices[composer], 2)
        else:
            # Dissimilar pair
            same = 0
            composers = list(composer_indices.keys())
            composer1, composer2 = random.sample(composers, 2)
            idx1 = random.choice(composer_indices[composer1])
            idx2 = random.choice(composer_indices[composer2])

        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        distance = np.linalg.norm(emb1 - emb2)
        distances.append(distance)
        pair_labels.append('Similar' if same == 1 else 'Dissimilar')

    df_hist = pd.DataFrame({
        'Distance': distances,
        'Pair Type': pair_labels
    })

    fig = px.histogram(df_hist, x='Distance', color='Pair Type', nbins=50, barmode='overlay',
                       histnorm='density', opacity=0.6)
    fig.update_layout(title='Histogram of Pair Distances',
                      xaxis_title='Euclidean Distance between Embeddings',
                      yaxis_title='Density')
    fig.show()

def plot_roc_curve(embeddings, labels):
    """
    Plot ROC curve using precomputed embeddings and labels, with threshold values displayed.

    Args:
        embeddings (np.ndarray): Precomputed embeddings.
        labels (np.ndarray): Composer labels.
    """
    # Compute pairwise distances and labels
    pairwise_distances = squareform(pdist(embeddings, metric='euclidean'))
    pairwise_labels = np.equal.outer(labels, labels).astype(int)

    # Extract upper triangle indices to avoid duplicates and self-comparisons
    triu_indices = np.triu_indices_from(pairwise_distances, k=1)
    distances = pairwise_distances[triu_indices]
    labels = pairwise_labels[triu_indices]

    fpr, tpr, thresholds = roc_curve(labels, -distances)
    # NOTE: Negative distances because lower distances imply positive class
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # Main ROC curve line
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC curve (area = {roc_auc:.2f})',
                             line=dict(color='darkorange', width=2)))

    # Random guess line
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             name='Random Guess',
                             line=dict(color='navy', width=2, dash='dash')))

    # Add scatter points with threshold annotations
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='markers',
        marker=dict(size=5, color='blue'),
        text=[f"Threshold: {-t:.4f}" for t in thresholds],
        hoverinfo="text",
        name='Threshold Points'
    ))

    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate (1-val_acc_dissimilar)',
        yaxis_title='True Positive Rate (val_acc_similar)',
        xaxis=dict(range=[-0.01, 1.0]),
        yaxis=dict(range=[0.0, 1.01])
    )

    fig.show()