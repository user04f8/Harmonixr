# viz_sample_in_context.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
import pandas as pd

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

############################################################
# Utility functions
############################################################

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

def extract_subsample_embeddings(model, piece, device, t, step=20):
    """
    Extract embeddings for multiple overlapping subsamples of a single piece.
    Subsamples are extracted every `step` frames for a window of length t.
    Returns: subsample_embeddings (N_subsamples, D), avg_embedding (D)
    """
    C, O, T_full = piece.shape
    # Do not normalize here, do it after cat
    if T_full < t:
        # If shorter than t, just pad once
        subsamples = [center_pad(piece, t).unsqueeze(0)]
    else:
        starts = range(0, T_full - t + 1, step)
        subsamples = [piece[:, :, s:s+t].unsqueeze(0) for s in starts]

    subsample_batch = torch.cat(subsamples, dim=0).float() / 255.0
    subsample_batch = subsample_batch.to(device)

    with torch.no_grad():
        embs = model.forward_one(subsample_batch).cpu().numpy()

    avg_emb = np.mean(embs, axis=0)
    return embs, avg_emb

def run_tsne(embeddings, perplexity=30, random_state=42):
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(embeddings)


############################################################
# Main routine
############################################################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(MODEL_CKPT_PATH, device=device)

    # Load main dataset
    main_x, main_y, main_composer_mapping = load_main_dataset(DATA_DIR)

    # Filter to top 50 composers by piece count
    composer_counts = Counter(main_y)
    top_50_composers = [c for c, _ in composer_counts.most_common(50)]

    # Filter main_x, main_y to these top 50 composers
    filtered_indices = [i for i, c in enumerate(main_y) if c in top_50_composers]
    main_x = [main_x[i] for i in filtered_indices]
    main_y = [main_y[i] for i in filtered_indices]

    # Also update composer_mapping to only contain those in top_50 if desired (not strictly needed)
    # We'll leave it as is, but it's fine if it has extra keys not used.

    # Create dataset from filtered main data
    main_dataset = FixedLengthDataset(main_x, main_y, T, composer_mapping=main_composer_mapping)
    main_embeddings, main_labels = extract_embeddings(model, main_dataset, device=device)

    # Compute centroids for main composers
    composer_to_embeddings = defaultdict(list)
    for emb, lbl in zip(main_embeddings, main_labels):
        composer_to_embeddings[lbl].append(emb)
    composer_centroids = []
    centroid_labels = []
    for c, embs_list in composer_to_embeddings.items():
        centroid = np.mean(embs_list, axis=0)
        composer_centroids.append(centroid)
        centroid_labels.append(c)
    composer_centroids = np.vstack(composer_centroids)
    centroid_labels = np.array(centroid_labels)

    # Load example pieces
    example_x = torch.load(EXAMPLE_PIECES_PT)
    example_y = torch.load(EXAMPLE_COMPOSERS_PT)
    with open(EXAMPLE_NAMES_TXT, 'r', encoding='utf-8') as f:
        example_names = [line.strip() for line in f.readlines()]

    # Normalize example_x
    if isinstance(example_x, torch.Tensor):
        example_x = [example_x[i].float()/255.0 for i in range(example_x.shape[0])]
    else:
        example_x = [xx.float()/255.0 for xx in example_x]

    # Create a separate composer mapping for example composers
    unique_example_composers = sorted(set([int(e.item()) for e in example_y]))
    example_composer_mapping = {int(c): f"{c}" for c in unique_example_composers}

    # Merge composer mappings
    merged_composer_mapping = dict(main_composer_mapping)
    for k, v in example_composer_mapping.items():
        if k not in merged_composer_mapping:
            merged_composer_mapping[k] = v

    # Extract embeddings for the example pieces (one embedding per piece)
    example_dataset = FixedLengthDataset(example_x, [int(yy.item()) for yy in example_y], T, composer_mapping=merged_composer_mapping)
    example_embeddings, example_labels = extract_embeddings(model, example_dataset, device=device)

    # Extract subsample embeddings for each example piece using raw data
    example_x_raw = torch.load(EXAMPLE_PIECES_PT)
    if isinstance(example_x_raw, torch.Tensor):
        example_x_raw = [example_x_raw[i] for i in range(example_x_raw.shape[0])]

    subsample_embeddings_list = []
    subsample_labels_list = []
    piece_name_labels = []

    for i, piece_raw in enumerate(example_x_raw):
        embs, avg_emb = extract_subsample_embeddings(model, piece_raw, device, t=T, step=STEP)
        subsample_embeddings_list.append(embs)
        subsample_labels_list.extend([int(example_y[i].item())]*embs.shape[0])
        piece_name_labels.append(example_names[i])

    subsample_embeddings = np.vstack(subsample_embeddings_list)
    subsample_labels = np.array(subsample_labels_list)

    # Combine all embeddings: centroids + example pieces + subsamples
    all_embeddings = np.vstack([composer_centroids, example_embeddings, subsample_embeddings])
    all_labels = np.concatenate([centroid_labels, example_labels, subsample_labels])

    n_centroids = len(composer_centroids)
    n_example = len(example_embeddings)
    subsample_start = n_centroids + n_example
    subsample_end = subsample_start + subsample_embeddings.shape[0]

    # Run TSNE on full set
    all_tsne = run_tsne(all_embeddings)

    # We will now reproduce the style of fig2 from the user's snippet.
    # We'll only use centroids + example pieces for the df used to determine color and size
    # Subsamples are plotted separately as grey points.

    # Create a DataFrame for centroids + example pieces
    # Composer names for these points:
    combined_labels = np.concatenate([centroid_labels, example_labels])
    combined_tsne = all_tsne[:n_centroids+n_example]

    # Map labels to composer names
    combined_composer_names = [merged_composer_mapping.get(int(lbl), f'Unknown_{lbl}') for lbl in combined_labels]

    # Create df as in snippet
    df = pd.DataFrame({
        'x': combined_tsne[:, 0],
        'y': combined_tsne[:, 1],
        'z': combined_tsne[:, 2],
        'composer': combined_composer_names
    })

    # Assign sample counts:
    # For main composers (in top 50), use the actual piece count from composer_counts
    # For example composers, they have 1 piece each
    piece_count_dict = {}
    for c in top_50_composers:
        # top_50_composers selected from main dataset, so composer_counts has their counts
        piece_count_dict[merged_composer_mapping[c]] = composer_counts[c]

    for c in unique_example_composers:
        piece_count_dict[merged_composer_mapping[c]] = 1

    # In snippet logic:
    # Calculate centroids and colors
    centroids = df.groupby('composer')[['x', 'y', 'z']].mean()
    centroids['color_value'] = centroids.sum(axis=1)

    color_scale = px.colors.sequential.Rainbow
    num_colors = len(color_scale)
    centroids['color_index'] = pd.qcut(centroids['color_value'], num_colors, labels=False)
    centroid_colors = {composer: color_scale[idx] for composer, idx in centroids['color_index'].items()}

    # Assign colors to df
    df['color'] = df['composer'].map(centroid_colors)

    # The snippet sets sample_count from the number of points per composer in df,
    # but we want actual piece counts:
    # centroids['sample_count'] = centroids.index.map(df['composer'].value_counts())
    # Override with actual counts from piece_count_dict
    centroids['sample_count'] = centroids.index.map(lambda c: piece_count_dict.get(c, 1))

    # Now we produce the fig2 style plot:
    # Plot centroids as in snippet fig2
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter3d(
        x=centroids['x'],
        y=centroids['y'],
        z=centroids['z'],
        mode='markers+text',
        marker=dict(
            size=centroids['sample_count'] / centroids['sample_count'].max() * 40 + 2,
            color=[centroid_colors[composer] for composer in centroids.index],
            opacity=0.8,
        ),
        text=centroids.index,
        textposition='top center',
        hovertext=centroids['sample_count'].apply(lambda x: f"Samples: {x}"),
        hoverinfo="text"
    ))

    fig2.update_layout(
        title='Centroids of Clusters',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        )
    )

    # Highlight example pieces (they are within df and centroids, but we also want them as diamonds)
    # Example pieces are from indices [n_centroids, n_centroids+n_example)
    for i, idx in enumerate(range(n_centroids, n_centroids + n_example)):
        fig2.add_trace(go.Scatter3d(
            x=[all_tsne[idx, 0]],
            y=[all_tsne[idx, 1]],
            z=[all_tsne[idx, 2]],
            mode='markers+text',
            text=[piece_name_labels[i]],
            textposition='top center',
            marker=dict(size=10, symbol='diamond', color='black'),
            name=piece_name_labels[i]
        ))

    # Add subsamples as separate grey points, not associated with any composer in df
    fig2.add_trace(go.Scatter3d(
        x=all_tsne[subsample_start:subsample_end, 0],
        y=all_tsne[subsample_start:subsample_end, 1],
        z=all_tsne[subsample_start:subsample_end, 2],
        mode='markers',
        marker=dict(size=4, symbol='circle', color='grey'),
        name='Subsamples'
    ))

    fig2.show()

    # Plot only subsamples of the two example pieces in a separate t-SNE as originally done
    subsample_tsne = run_tsne(subsample_embeddings, perplexity=5)
    sdf = pd.DataFrame({
        'x': subsample_tsne[:, 0],
        'y': subsample_tsne[:, 1],
        'z': subsample_tsne[:, 2],
        'composer': [example_composer_mapping.get(int(lbl), 'Unknown') for lbl in subsample_labels]
    })

    # Just a simple 3D scatter for subsamples
    fig_sub = px.scatter_3d(sdf, x='x', y='y', z='z', color='composer', hover_name='composer',
                            title='3D t-SNE: Subsamples of the Two Example Pieces')
    fig_sub.show()
