import glob
import mido
import numpy as np
import os
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

assert torch.cuda.is_available(), "CUDA not available (CPU training will be very slow)"

# Function to read and convert MIDI files to note sequences
def midi_to_note_sequence(file_path, ticks_per_beat=480, step_size=120):
    """Convert a MIDI file to a sequence of notes."""
    mid = mido.MidiFile(file_path)
    notes = []
    current_time = 0

    for msg in mid:
        if not msg.is_meta:
            current_time += msg.time  # Update time first
            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                notes.append((current_time, note))
    # Quantize time steps
    quantized_sequence = []
    for time, note in notes:
        quantized_time = int(time / step_size)
        quantized_sequence.append((quantized_time, note))
    return quantized_sequence

# Process all MIDI files in a folder
def process_midi_folder_with_filenames(folder_path, percent_data=1.0):
    note_sequences = []
    filenames = []
    all_file_paths = glob.glob(folder_path)
    total_files = len(all_file_paths)
    num_files_to_use = max(1, int(total_files * percent_data))

    # Shuffle and select a subset of files
    np.random.shuffle(all_file_paths)
    selected_file_paths = all_file_paths[:num_files_to_use]

    for file_path in selected_file_paths:
        try:
            # Try to read and process the MIDI file
            note_sequence = midi_to_note_sequence(file_path)
            if note_sequence:  # Ensure the sequence is not empty
                note_sequences.append(note_sequence)
                file_name = os.path.basename(file_path)
                filenames.append(file_name)

        except (OSError, ValueError) as e:
            # Catch specific exceptions and print an error message
            print(f"Error processing file {file_path}: {e}. Skipping this file.")

        except Exception as e:
            # Catch any other unexpected exceptions
            print(f"Unexpected error with file {file_path}: {e}. Skipping this file.")

    return note_sequences, filenames

print("Loading MIDI files...")

midi_folder = r"data/clean_midi/*/*.mid"
percent_data = 0.1
note_sequences, filenames = process_midi_folder_with_filenames(midi_folder, percent_data)

# Data Augmentation Functions
def transpose_sequence(sequence, shift, min_note=21, max_note=108):
    """Transpose a sequence of notes by a certain number of semitones, ensuring notes stay within valid range."""
    transposed_sequence = []
    for time, note in sequence:
        transposed_note = note + shift
        if min_note <= transposed_note <= max_note:
            transposed_sequence.append((time, transposed_note))
    return transposed_sequence

# Apply transposition to all note sequences
def augment_data(note_sequences, shifts=[-2, -1, 1, 2]):
    augmented_sequences = []
    for seq in note_sequences:
        augmented_sequences.append(seq)  # Original sequence
        for shift in shifts:
            transposed_seq = transpose_sequence(seq, shift)
            if transposed_seq:  # Ensure the transposed sequence is not empty
                augmented_sequences.append(transposed_seq)
    return augmented_sequences

# **Perform augmentation before creating vocabulary**
augmented_sequences = note_sequences # augment_data(note_sequences)

# **Create vocabulary from augmented data**
def create_vocab(note_sequences):
    notes = set(note for seq in note_sequences for _, note in seq)
    note_to_int = {note: i for i, note in enumerate(sorted(notes))}
    int_to_note = {i: note for note, i in note_to_int.items()}
    return note_to_int, int_to_note

note_to_int, int_to_note = create_vocab(augmented_sequences)

# Convert sequences to tokenized format
def tokenize_sequences(note_sequences, note_to_int):
    tokenized_sequences = []
    for sequence in note_sequences:
        tokenized_sequence = []
        for _, note in sequence:
            if note in note_to_int:
                tokenized_sequence.append(note_to_int[note])
            else:
                # Handle unseen notes (should not occur if vocab is built correctly)
                tokenized_sequence.append(note_to_int['<UNK>'])  # Use an unknown token if implemented
        tokenized_sequences.append(tokenized_sequence)
    return tokenized_sequences

tokenized_sequences = tokenize_sequences(augmented_sequences, note_to_int)

def prepare_sequences_with_filenames(tokenized_sequences, filenames, seq_length=50):
    inputs, targets = [], []
    expanded_filenames = []

    for i, sequence in enumerate(tokenized_sequences):
        if len(sequence) > seq_length:
            for j in range(0, len(sequence) - seq_length):
                inputs.append(sequence[j:j + seq_length])
                targets.append(sequence[j + seq_length])
                expanded_filenames.append(filenames[i])  # Associate this sequence with the corresponding file name
    return np.array(inputs), np.array(targets), expanded_filenames

# Prepare sequences
seq_length = 50
inputs, targets, expanded_filenames = prepare_sequences_with_filenames(tokenized_sequences, filenames, seq_length)

# Check if we have data to train
if len(inputs) == 0 or len(targets) == 0:
    raise ValueError("Not enough data to train. Please check your MIDI files and preprocessing steps.")

# Train/test split
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    inputs, targets, expanded_filenames, test_size=0.2, random_state=42
)

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=16, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.2):
        super(MusicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for better compatibility
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.embedding(src)  # [batch_size, seq_len, d_model]
        tgt = self.embedding(tgt)
        # Generate masks
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc_out(output)  # [batch_size, seq_len, vocab_size]
        return output

# Hyperparameters
vocab_size = len(note_to_int)  # Number of unique notes
model = MusicTransformer(vocab_size)

optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Training loop with improvements
def train(model, X_train, y_train, epochs=40, batch_size=32, save_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = iter = 0

        for inputs_batch, targets_batch in dataloader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_batch, inputs_batch)  # Using inputs as both src and tgt
            outputs = outputs[:, -1, :]  # Take the last output

            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item() * inputs_batch.size(0)  # Multiply by batch size
            
            iter += 1
            if iter % 100 == 0:
                print(f"epoch loss {100 * iter / (len(X_train) / inputs_batch.size(0)):.1f}%: {epoch_loss / iter / inputs_batch.size(0)}")

        average_loss = epoch_loss / len(X_train)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')

        # Step the scheduler after each epoch
        scheduler.step(average_loss)

        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

# Train the model
print("Starting training...")
train(model, X_train, y_train, epochs=1, batch_size=128)

def generate_notes(model, start_sequence, num_notes=100, temperature=1.0, top_k=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    generated_sequence = start_sequence.copy()

    for _ in range(num_notes):
        input_seq = torch.tensor([generated_sequence[-seq_length:]], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_seq, input_seq)
        next_note_logits = output[:, -1, :]  # Get the last time step's output

        # Apply temperature scaling
        next_note_logits = next_note_logits / temperature

        # Apply top-k sampling
        top_k = min(top_k, next_note_logits.size(-1))  # Ensure top_k doesn't exceed vocabulary size
        topk_values, topk_indices = torch.topk(next_note_logits, k=top_k, dim=-1)

        # Convert to probabilities
        probabilities = torch.softmax(topk_values, dim=-1)

        # Sample from the top-k probabilities
        next_note = torch.multinomial(probabilities, num_samples=1).item()
        # Map back to the original index
        next_note = topk_indices[0, next_note].item()

        generated_sequence.append(next_note)

    return generated_sequence

for i in range(15):
    # Get the original filename for this test sample
    original_filename = filenames_test[i].replace('.mid', '')  # Remove ".mid" extension for cleaner file naming

    print(f"{i}: Generating for {original_filename} . . .")

    # Generate a sequence of notes starting from a sample in the test data
    start_sequence = X_test[i].tolist()
    print(f"Start sequence: {start_sequence}")

    # Generate notes with adjusted temperature and top_k
    generated_sequence = generate_notes(model, start_sequence, num_notes=100, temperature=1.2, top_k=10)

    # Convert back to MIDI and save with the original filename included
    def sequence_to_midi(generated_sequence, int_to_note, output_file):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        time_step = 90  # Adjust as needed
        previous_time = 0

        # Combine start_sequence and generated_sequence for full output
        full_sequence = start_sequence + generated_sequence

        for token in full_sequence:
            note = int_to_note[token]
            track.append(mido.Message('note_on', note=note, velocity=64, time=int(time_step)))
            track.append(mido.Message('note_off', note=note, velocity=64, time=int(time_step)))
            previous_time += time_step

        mid.save(output_file)
        print(f"MIDI file saved as {output_file}")

    # Save the output with the source file's name
    output_file = f'out/generated_output_from_{original_filename}_{i}.mid'
    sequence_to_midi(generated_sequence, int_to_note, output_file=output_file)