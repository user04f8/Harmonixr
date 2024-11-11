from viz_utils import load_model, MIDISingleDataset, extract_embeddings, visualize_tsne, plot_distance_histogram, plot_roc_curve
import torch

def main():
    print("Loading model")
    checkpoint_path = r'tb_logs/SiaViT/version_34/checkpoints/last.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device=device)

    print("Loading dataset")
    data_dir = model.hparams.data_dir
    t = model.hparams.t
    
    dataset = MIDISingleDataset(data_dir=data_dir, t=t, n_composers=None, n_pieces_threshold=6)

    embeddings, labels, composer_names = extract_embeddings(model, dataset, device)

    visualize_tsne(embeddings, composer_names)

    plot_distance_histogram(embeddings, labels, num_pairs=50000)

    plot_roc_curve(embeddings, labels)

if __name__ == '__main__':
    main()