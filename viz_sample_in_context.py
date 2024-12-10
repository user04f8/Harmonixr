# viz_sample_in_context.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from collections import Counter, defaultdict

from viz_utils import load_model

############################################################
# Configuration
############################################################
MODEL_CKPT_PATH = 'tb_logs/SiaViT/version_56/checkpoints/last.ckpt'
DATA_DIR = './data/expon_decay'  # main dataset directory
EXAMPLES_DIR = './data/example'   # example pieces directory
EXAMPLE_PIECES_PT = os.path.join(EXAMPLES_DIR, 'midi_pieces.pt')
EXAMPLE_COMPOSERS_PT = os.path.join(EXAMPLES_DIR, 'composer_vector.pt')
EXAMPLE_NAMES_TXT = os.path.join(EXAMPLES_DIR, 'piece_names.txt')

T = 1200  # Target length for embedding extraction
STEP = 20 # Step size for subsample extraction
TOP_COMPOSERS = 50  # Limit to the top N composers by the number of pieces

############################################################
# Utility functions
############################################################

class FixedLengthDataset(Dataset):
    """Dataset that center-crops or pads samples to a fixed length T."""
    def __init__(self, pieces, labels, t, composer_mapping=None):
        self.x = pieces
        self.y = labels
        self.t = t
        self.composer_mapping = composer_mapping if composer_mapping else {}
        self.indices = list(range(len(self.x)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.x[idx]
        composer_idx = self.y[idx]
        sample = self._process_sample(sample)
        return sample, composer_idx

    def _process_sample(self, sample):
        C, O, T_var = sample.shape
        if T_var >= self.t:
            start = (T_var - self.t) // 2
            sample = sample[:, :, start:start+self.t]
        else:
            pad_size = self.t - T_var
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            pad_left_tensor = torch.zeros((C, O, pad_left))
            pad_right_tensor = torch.zeros((C, O, pad_right))
            sample = torch.cat([pad_left_tensor, sample, pad_right_tensor], dim=2)
        return sample

def center_pad(piece, t):
    """Center-pad a piece to length t along the time dimension."""
    C, O, T_var = piece.shape
    if T_var >= t:
        start = (T_var - t) // 2
        return piece[:, :, start:start+t]
    else:
        pad_size = t - T_var
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        pad_left_tensor = torch.zeros((C, O, pad_left))
        pad_right_tensor = torch.zeros((C, O, pad_right))
        return torch.cat([pad_left_tensor, piece, pad_right_tensor], dim=2)


def load_main_dataset(data_dir):
    """Load the main dataset: pieces, labels, and composer mapping."""
    x = torch.load(os.path.join(data_dir, 'midi_pieces.pt'))
    y = torch.load(os.path.join(data_dir, 'composer_vector.pt'))
    # Normalize
    if isinstance(x, torch.Tensor):
        x = [x[i].float() / 255.0 for i in range(x.shape[0])]
    else:
        x = [xx.float() / 255.0 for xx in x]

    y = [int(label.item()) for label in y]

    # Load composer mapping (one composer name per line)
    composer_mapping = {}
    with open(os.path.join(data_dir, 'composer_mapping.txt'), 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    for i, line in enumerate(lines):
        composer_mapping[i] = line.strip()

    return x, y, composer_mapping

def filter_top_composers(pieces, labels, composer_mapping, top_n=50):
    """Filter the dataset to include only the top N composers by number of pieces."""
    composer_counts = Counter(labels)
    top_composers = [composer for composer, _ in composer_counts.most_common(top_n)]
    filtered_pieces = [piece for piece, label in zip(pieces, labels) if label in top_composers]
    filtered_labels = [label for label in labels if label in top_composers]
    filtered_mapping = {label: composer_mapping[label] for label in top_composers}
    return filtered_pieces, filtered_labels, filtered_mapping

def extract_embeddings(model, dataset, device):
    """Extract embeddings for an entire dataset."""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    embeddings, labels = [], []
    with torch.no_grad():
        for samples, composer_indices in tqdm(dataloader, desc='Extracting embeddings'):
            samples = samples.to(device)
            embs = model.forward_one(samples).cpu().numpy()
            embeddings.append(embs)
            labels.extend(composer_indices.numpy())
    return np.vstack(embeddings), np.array(labels)

def extract_subsample_embeddings(model, piece, device, t, step=20):
    """Extract embeddings for multiple overlapping subsamples of a single piece."""
    C, O, T_full = piece.shape
    piece = piece.float() / 255.0

    if T_full < t:
        # If shorter than t, just pad once
        subsamples = [center_pad(piece, t).unsqueeze(0)]
    else:
        starts = range(0, T_full - t + 1, step)
        subsamples = [piece[:, :, s:s+t].unsqueeze(0) for s in starts]

    subsample_batch = torch.cat(subsamples, dim=0).to(device)

    with torch.no_grad():
        embs = model.forward_one(subsample_batch).cpu().numpy()

    avg_emb = np.mean(embs, axis=0)
    return embs, avg_emb

def run_tsne(embeddings, perplexity=30, random_state=42):
    """Run t-SNE on the given embeddings."""
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(embeddings)

def plot_centroids_with_highlights(
    tsne_embeddings, centroids_df, centroid_colors, example_embeddings, example_labels, subsample_embeddings
):
    """Plot composer centroids, highlighting example pieces and subsamples."""
    # Plot centroids
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=centroids_df['x'],
        y=centroids_df['y'],
        z=centroids_df['z'],
        mode='markers+text',
        marker=dict(
            size=centroids_df['sample_count'] / centroids_df['sample_count'].max() * 40 + 2,
            color=[centroid_colors[c] for c in centroids_df.index],
            opacity=0.8,
        ),
        text=centroids_df.index,
        textposition='top center',
        hovertext=centroids_df['sample_count'].apply(lambda x: f"Samples: {x}"),
        hoverinfo="text"
    ))

    # Highlight example pieces
    fig.add_trace(go.Scatter3d(
        x=example_embeddings[:, 0],
        y=example_embeddings[:, 1],
        z=example_embeddings[:, 2],
        mode='markers+text',
        marker=dict(size=10, symbol='diamond', color='black'),
        text=[f"Example {i}" for i in example_labels],
        textposition='top center',
        name='Example Pieces'
    ))

    # Add subsamples
    fig.add_trace(go.Scatter3d(
        x=subsample_embeddings[:, 0],
        y=subsample_embeddings[:, 1],
        z=subsample_embeddings[:, 2],
        mode='markers',
        marker=dict(size=4, symbol='circle', color='grey'),
        name='Subsamples'
    ))

    fig.update_layout(
        title='Centroids with Example Pieces and Subsamples',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z')
    )
    fig.show()

############################################################
# Main routine
############################################################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(MODEL_CKPT_PATH, device=device)

    # Load main dataset
    main_x, main_y, main_composer_mapping = load_main_dataset(DATA_DIR)

    # Filter to top composers
    top_pieces, top_labels, top_composer_mapping = filter_top_composers(main_x, main_y, main_composer_mapping, TOP_COMPOSERS)
    top_dataset = FixedLengthDataset(top_pieces, top_labels, T, composer_mapping=top_composer_mapping)
    top_embeddings, top_labels = extract_embeddings(model, top_dataset, device=device)

    # Compute centroids for top composers
    composer_to_embeddings = defaultdict(list)
    for emb, lbl in zip(top_embeddings, top_labels):
        composer_to_embeddings[lbl].append(emb)
    composer_centroids = {c: np.mean(embs, axis=0) for c, embs in composer_to_embeddings.items()}
    centroids_embeddings = np.vstack(list(composer_centroids.values()))
    centroids_labels = list(composer_centroids.keys())

    # Compute t-SNE for centroids
    tsne_centroids = run_tsne(centroids_embeddings)

    # Create DataFrame for centroids
    centroids_df = pd.DataFrame(tsne_centroids, columns=['x', 'y', 'z'])
    centroids_df['composer'] = [top_composer_mapping[c] for c in centroids_labels]
    centroids_df['sample_count'] = [len(composer_to_embeddings[c]) for c in centroids_labels]

    # Assign unique colors to centroids
    color_scale = px.colors.sequential.Rainbow
    num_colors = len(color_scale)
    centroids_df['color_index'] = pd.qcut(centroids_df['x'] + centroids_df['y'] + centroids_df['z'], num_colors, labels=False)
    centroid_colors = {composer: color_scale[idx] for composer, idx in zip(centroids_df['composer'], centroids_df['color_index'])}

    # Load example pieces and subsamples
    example_x = torch.load(EXAMPLE_PIECES_PT)
    example_y = torch.load(EXAMPLE_COMPOSERS_PT)
    example_dataset = FixedLengthDataset(example_x, example_y, T)
    example_embeddings, example_labels = extract_embeddings(model, example_dataset, device=device)

    subsample_embeddings_list = []
    for piece in example_x:
        subsample_embs, _ = extract_subsample_embeddings(model, piece, device, T, STEP)
        subsample_embeddings_list.append(subsample_embs)
    subsample_embeddings = np.vstack(subsample_embeddings_list)

    # Combine and plot
    plot_centroids_with_highlights(tsne_centroids, centroids_df, centroid_colors, example_embeddings, example_labels, subsample_embeddings)
