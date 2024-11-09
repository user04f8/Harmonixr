import torch
import os
import random

def generate_synthetic_data(data_dir, num_samples=1000, t=128, o=5, T=200):
    """
    Generates synthetic MIDI data with unique noise patterns for each composer.
    
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

    # Define unique noise patterns for each composer
    noise_patterns = {
        'Composer_A': lambda: torch.sin(torch.linspace(0, 10, T)).repeat(12, o, 1) / 10,
        'Composer_B': lambda: torch.rand(12, o, T) * torch.linspace(0.5, 1.5, T).unsqueeze(0).unsqueeze(0),
        'Composer_C': lambda: torch.zeros(12, o, T).uniform_(0, 0.2) + torch.eye(12, T).unsqueeze(1),
    }

    for i in range(num_samples):
        # Randomly assign a composer and title
        composer = random.choice(composers)
        
        # Generate unique noise based on the composer
        base_noise = noise_patterns[composer]()
        random_noise = torch.rand(12, o, T) / 100  # Add minor random noise
        midi_data = base_noise + random_noise

        x_data.append(midi_data)
        y_data.append({'composer': composer, 'title': titles[i]})

    # Split into training and validation sets
    train_split = int(0.8 * num_samples)  # 80% training, 20% validation
    x_train, x_val = x_data[:train_split], x_data[train_split:]
    y_train, y_val = y_data[:train_split], y_data[train_split:]
    
    # Save the data
    torch.save(x_train, os.path.join(data_dir, 'x_train.pt'))
    torch.save(x_val, os.path.join(data_dir, 'x_val.pt'))
    torch.save(y_train, os.path.join(data_dir, 'y_train.pt'))
    torch.save(y_val, os.path.join(data_dir, 'y_val.pt'))
    
    print(f"Data saved to {data_dir}")
    print(f"Train samples: {len(x_train)}, Validation samples: {len(x_val)}")

if __name__ == "__main__":
    generate_synthetic_data('./synthetic_data', num_samples=1000, t=128, o=5, T=200)
