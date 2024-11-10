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
    from model import SiaViT
    model = SiaViT.load_from_checkpoint(ckpt_path)
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
    def __init__(self, data_dir, t, n_composers=None, n_pieces_threshold=None):
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

        # Filter composers based on n_pieces_threshold
        if n_pieces_threshold is not None:
            # Count pieces per composer
            from collections import Counter
            composer_counts = Counter(self.y)
            composers_to_include = [composer for composer, count in composer_counts.items() if count >= n_pieces_threshold]

            # Filter dataset by composers with enough pieces
            filtered_indices = [i for i, composer_idx in enumerate(self.y) if composer_idx in composers_to_include]
            self.x = [self.x[i] for i in filtered_indices]
            self.y = [self.y[i] for i in filtered_indices]

        # Further filter composers if n_composers is specified
        if n_composers is not None:
            unique_composers = sorted(set(self.y))[:n_composers]
            filtered_indices = [i for i, composer_idx in enumerate(self.y) if composer_idx in unique_composers]
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

    # Create a DataFrame to store t-SNE results with composer names
    df = pd.DataFrame({
        'x': embeddings_tsne[:, 0],
        'y': embeddings_tsne[:, 1],
        'z': embeddings_tsne[:, 2],
        'composer': composer_names
    })

    # Calculate centroids and their color values
    centroids = df.groupby('composer')[['x', 'y', 'z']].mean()
    centroids['color_value'] = centroids.sum(axis=1)  # Use x + y + z for each centroid's color basis

    # Normalize colors and assign each cluster a unique color using Plotly's colormap
    color_scale = px.colors.sequential.Rainbow  # Choose any Plotly colormap like 'Viridis', 'Cividis', etc.
    num_colors = len(color_scale)
    centroids['color_index'] = pd.qcut(centroids['color_value'], num_colors, labels=False)  # Map to color scale
    centroid_colors = {composer: color_scale[idx] for composer, idx in centroids['color_index'].items()}

    # Apply the color for each cluster
    df['color'] = df['composer'].map(centroid_colors)

    sample_counts = df['composer'].value_counts()
    centroids['sample_count'] = centroids.index.map(sample_counts)  # Map counts to centroids

    # Create the 3D scatter plot with consistent cluster colors
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='composer', hover_name='composer')
    for trace, composer in zip(fig.data, df['composer'].unique()):
        trace.marker.color = df[df['composer'] == composer]['color'].iloc[0]

    fig.update_layout(title='3D t-SNE of Embeddings by Composer')
    fig.show()

    # Plot centroids with marker size based on sample count
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter3d(
        x=centroids['x'],
        y=centroids['y'],
        z=centroids['z'],
        mode='markers+text',
        marker=dict(
            size=centroids['sample_count'] / centroids['sample_count'].max() * 40 + 2,  # Scale size
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
    fig2.show()

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
    MAX_DIST = 0.1

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
        distances.append(min(distance, MAX_DIST))
        pair_labels.append('Similar' if same == 1 else 'Dissimilar')

    df_hist = pd.DataFrame({
        'Distance': distances,
        'Pair Type': pair_labels
    })

    color_sequence = [
        '#636EFA',  # similar: blue
        '#EF553B'   # dissimilar: red
    ]

    fig = px.histogram(df_hist, x='Distance', color='Pair Type', range_x=[0, MAX_DIST], nbins=101, barmode='overlay',
                       histnorm='density', opacity=0.6, color_discrete_sequence=color_sequence)
    fig.update_layout(title='Histogram of Pair Distances',
                      xaxis_title='Euclidean Distance between Embeddings',
                      yaxis_title='Density')
    fig.show()

def plot_roc_curve(embeddings, labels, max_points=500):
    """
    Plot ROC curve using precomputed embeddings and labels, with threshold values displayed and
    points colored by the mixed accuracy metric. Optimized for performance by downsampling.

    Args:
        embeddings (np.ndarray): Precomputed embeddings.
        labels (np.ndarray): Composer labels.
        max_points (int): Maximum number of threshold points to display.
    """
    # Compute pairwise distances and labels
    pairwise_distances = squareform(pdist(embeddings, metric='euclidean'))
    pairwise_labels = np.equal.outer(labels, labels).astype(int)

    # Extract upper triangle indices to avoid duplicates and self-comparisons
    triu_indices = np.triu_indices_from(pairwise_distances, k=1)
    distances = pairwise_distances[triu_indices]
    labels_binary = pairwise_labels[triu_indices]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels_binary, -distances)  # Negative distances because lower distances imply positive class
    roc_auc = auc(fpr, tpr)

    # Calculate val_acc_similar, val_acc_dissimilar, and val_acc_mixed
    val_acc_similar = tpr
    val_acc_dissimilar = 1 - fpr
    val_acc_mixed = 0.5 * (val_acc_similar + val_acc_dissimilar)

    # Find index of the optimal threshold maximizing val_acc_mixed
    optimal_index = np.argmax(val_acc_mixed)
    optimal_fpr = fpr[optimal_index]
    optimal_tpr = tpr[optimal_index]
    optimal_threshold = thresholds[optimal_index]

    # Downsample points if necessary
    if len(fpr) > max_points:
        # Select indices evenly spaced across the ROC curve
        indices = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr_sampled = fpr[indices]
        tpr_sampled = tpr[indices]
        thresholds_sampled = thresholds[indices]
        val_acc_mixed_sampled = val_acc_mixed[indices]
    else:
        fpr_sampled = fpr
        tpr_sampled = tpr
        thresholds_sampled = thresholds
        val_acc_mixed_sampled = val_acc_mixed

    fig = go.Figure()

    # Main ROC curve line
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='darkorange', width=2)
    ))

    # Random guess line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='navy', width=2, dash='dash')
    ))

    # Add downsampled scatter points colored by val_acc_mixed
    fig.add_trace(go.Scattergl(
        x=fpr_sampled,
        y=tpr_sampled,
        mode='markers',
        marker=dict(
            size=6,
            color=val_acc_mixed_sampled,
            colorscale='Viridis',
            colorbar=dict(title='val_acc_mixed', x=1.15),
            showscale=True
        ),
        text=[f"Threshold: {-t:.3f}<br>val_acc_mixed: {m:.3f}" for t, m in zip(thresholds_sampled, val_acc_mixed_sampled)],
        hoverinfo="text",
        name='Threshold Points'
    ))

    # Highlight the optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_fpr],
        y=[optimal_tpr],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='x'),
        text=[f"Optimal Threshold: {-optimal_threshold:.4f}<br>val_acc_mixed: {val_acc_mixed[optimal_index]:.3f}"],
        textposition="top right",
        hoverinfo="text",
        name='Optimal Threshold'
    ))

    # Update layout to adjust legend and colorbar positions
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate (1 - val_acc_dissimilar)',
        yaxis_title='True Positive Rate (val_acc_similar)',
        xaxis=dict(range=[-0.01, 1.0]),
        yaxis=dict(range=[0.0, 1.01]),
        legend=dict(
            x=0.8,
            y=0.2,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1
        ),
        margin=dict(r=100)
    )

    fig.show()