import torch
import os
import random

def generate_synthetic_data(data_dir, num_samples=1000, t=128, o=5, T=200):
    """
    Generates synthetic MIDI data for testing.
    
    Args:
        data_dir (str): Directory to save the data.
        num_samples (int): Number of synthetic samples to generate.
        t (int): Fixed length for subsamples.
        o (int): Number of octaves.
        T (int): Max time length of each MIDI sample.
    """
    os.makedirs(data_dir, exist_ok=True)

    composers = ['Composer_A', 'Composer_B', 'Composer_C']
    titles = [f'Piece_{i}' for i in range(num_samples)]
    
    x_data = []
    y_data = []

    for i in range(num_samples):
        # Create a random tensor of shape (12, o, T) for each sample
        midi_data = torch.rand(12, o, T)
        x_data.append(midi_data)
        
        # Randomly assign a composer and title
        composer = random.choice(composers)
        title = titles[i]
        y_data.append({'composer': composer, 'title': title})

    # Save the data
    train_split = int(0.8 * num_samples)  # 80% training, 20% validation
    x_train, x_val = x_data[:train_split], x_data[train_split:]
    y_train, y_val = y_data[:train_split], y_data[train_split:]
    
    torch.save(x_train, os.path.join(data_dir, 'x_train.pt'))
    torch.save(x_val, os.path.join(data_dir, 'x_val.pt'))
    torch.save(y_train, os.path.join(data_dir, 'y_train.pt'))
    torch.save(y_val, os.path.join(data_dir, 'y_val.pt'))
    
    print(f"Data saved to {data_dir}")
    print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}")

if __name__ == "__main__":
    generate_synthetic_data('./synthetic_data', num_samples=1000, t=128, o=5, T=200)
