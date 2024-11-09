from viz_utils import load_model, MIDISingleDataset, extract_embeddings, visualize_tsne, plot_distance_histogram, plot_roc_curve
import torch

def main():
    # Load the model
    checkpoint_path = 'tb_logs/BIG-MIDIClassifier/version_14/checkpoints/last.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device=device)

    # Load the dataset with up to n composers
    data_dir = model.hparams.data_dir  # Assuming data_dir is stored in hparams
    t = model.hparams.t  # Assuming t is stored in hparams
    n_composers = 10  # Adjust as needed
    dataset = MIDISingleDataset(data_dir=data_dir, t=t, n_composers=n_composers)

    # Extract embeddings
    embeddings, labels, composer_names = extract_embeddings(model, dataset, device)

    # Visualize t-SNE
    visualize_tsne(embeddings, composer_names)

    # Plot distance histogram
    plot_distance_histogram(embeddings, labels, num_pairs=1000)

    # Plot ROC curve
    plot_roc_curve(embeddings, labels)

if __name__ == '__main__':
    main()