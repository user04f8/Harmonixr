from viz_utils import load_model, MIDISingleDataset, extract_embeddings, visualize_tsne, plot_distance_histogram, plot_roc_curve
import torch

def main():
    print("Loading model")
    checkpoint_path = r'archive\epoch=52-val_acc_mixed=0.62.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device=device)

    print("Loading dataset")
    data_dir = model.hparams.data_dir
    t = model.hparams.t
    n_composers = 50 
    dataset = MIDISingleDataset(data_dir=data_dir, t=t, n_composers=n_composers)

    embeddings, labels, composer_names = extract_embeddings(model, dataset, device)

    visualize_tsne(embeddings, composer_names)

    plot_distance_histogram(embeddings, labels, num_pairs=1000)

    plot_roc_curve(embeddings, labels)

if __name__ == '__main__':
    main()