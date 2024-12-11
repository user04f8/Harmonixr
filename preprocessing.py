import os
import numpy as np
import torch
import mido
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def parse_filename(filename):
    """Parses the composer and piece name from the filename."""
    parts = filename.split(',')
    composer_last = parts[0].strip()
    composer_first = parts[1].strip()
    composer_name = f"{composer_last} {composer_first}"
    piece_name = ','.join(parts[2:-1]).strip().replace('.mid', '')
    return composer_name, piece_name

def transpose_pitch(midi_note):
    """Transpose MIDI pitch to start at F within each octave (e.g., F, F#, G, ..., E)."""
    pitch_class = (midi_note - 5) % 12  # Shift C-based octave to start from F
    return pitch_class

def midi_to_array(filepath, time_resolution_ms=50, decay_time_ms=10000, vis=False):
    """Encodes MIDI file data into a numpy array with velocity-based values and exponential decay, stored in uint8 format."""
    
    # Define the 6-octave range (F1 to E7)
    min_midi_note = 29  # F1 (MIDI note 29)
    max_midi_note = 88  # E7 (MIDI note 88)
    octave_range = range(1, 7)  # Octaves 1 through 6
    pitch_range = range(12)  # Transposed pitches within an octave (0 to 11, starting from F)
    
    # Load MIDI file and set up time grid
    mid = mido.MidiFile(filepath)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # Default tempo in microseconds per beat (120 BPM)
    
    # Find the tempo if it's set in the MIDI file
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    
    # Calculate milliseconds per tick
    ms_per_tick = (tempo / ticks_per_beat) / 1000

    # Determine total time in milliseconds
    total_ticks = sum(msg.time for track in mid.tracks for msg in track)
    total_ms = int(total_ticks * ms_per_tick)
    
    # Set up array dimensions based on the time grid
    time_steps = int(np.ceil(total_ms / time_resolution_ms))
    midi_array = np.zeros((len(pitch_range), len(octave_range), time_steps), dtype=np.uint8)

    # Exponential decay parameters
    decay_steps = int(decay_time_ms / time_resolution_ms)  # Decay duration in time steps
    decay_factors = np.exp(4 * -np.linspace(0, decay_time_ms, decay_steps) / decay_time_ms)  # Decay factors in float
    octave_decay_multipliers = [0.7, 1, 1.5, 2, 3, 4]

    # Process MIDI events and fill in the array
    current_time_ms = 0
    active_notes = {}  # Track active notes by (pitch, octave)
    
    for track in mid.tracks:
        for msg in track:
            current_time_ms += int(msg.time * ms_per_tick)
            time_index = int(current_time_ms / time_resolution_ms)
            
            if msg.type == 'note_on' and msg.velocity > 0:
                pitch = transpose_pitch(msg.note)  # Transpose pitch to F-based octave
                octave = (msg.note - min_midi_note) // 12 + 1
                if min_midi_note <= msg.note <= max_midi_note and octave in octave_range:
                    # Scale velocity to integer range 1–127
                    initial_velocity = np.clip(int(msg.velocity), 1, 127)
                    active_notes[(pitch, octave)] = (time_index, initial_velocity)
            
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                pitch = transpose_pitch(msg.note)
                octave = (msg.note - min_midi_note) // 12 + 1
                if (pitch, octave) in active_notes and min_midi_note <= msg.note <= max_midi_note:
                    start_time, initial_velocity = active_notes.pop((pitch, octave))
                    
                    # Apply integer decay to velocity values in the array
                    for i, t in enumerate(range(start_time, time_index)):
                        if i < decay_steps:
                            midi_array[pitch, octave - 1, t] = np.clip(int(initial_velocity * (decay_factors[i] ** octave_decay_multipliers[octave - 1])), 1, 127)
                        else:
                            midi_array[pitch, octave - 1, t] = 1  # Sustain at minimum velocity of 1 when decay period is over
    if vis:
        output_folder = 'viztest'
        os.makedirs(output_folder, exist_ok=True)

        pitch_range, octave_range, time_steps = midi_array.shape

        for t in range(time_steps):
            # Extract the slice for this time step
            time_slice = np.rot90(midi_array[:, :, t], k=1)

            # Convert time slice to an image with Pillow
            # Map values to grayscale intensities (0-255)
            img = Image.fromarray((time_slice * 2).astype(np.uint8), mode='L')  # Scale values to full grayscale range

            # Resize image to make each cell more visible (optional)
            img = img.resize((pitch_range * 50, octave_range * 50), Image.NEAREST)

            # Save the image for this time step
            file_path = os.path.join(output_folder, f"frame_{t:04d}.png")
            img.save(file_path)
    return midi_array

def process_midi_folder(folder_path, time_resolution_ms=10):
    """Processes each MIDI file, creates a list of tensors (no padding) and saves a composer embedding vector."""
    composer_to_id = {}
    piece_arrays = []
    composer_ids = []
    piece_names = []

    # Get list of all MIDI files
    midi_files = [f for f in os.listdir(folder_path) if f.endswith('.mid')]

    #midi_files = midi_files[0:20]
    
    # Parse files with a progress bar
    for filename in tqdm(midi_files, desc="Processing MIDI files"):
        composer, piece = parse_filename(filename)
        if composer not in composer_to_id:
            composer_to_id[composer] = len(composer_to_id)
        
        file_path = os.path.join(folder_path, filename)
        midi_array = midi_to_array(file_path, time_resolution_ms)
        
        # Convert to PyTorch tensor with uint8 type
        piece_arrays.append(torch.tensor(midi_array, dtype=torch.uint8))
        composer_ids.append(composer_to_id[composer])
        piece_names.append(piece)

    # Save composer embedding vector as a tensor
    composer_vector = torch.tensor(composer_ids, dtype=torch.long)
    data_dir = os.path.dirname("./data/composer_mapping.txt")
    os.makedirs(data_dir, exist_ok=True)
    # Save composer-to-id mapping
    with open("./data/composer_mapping.txt", "w") as f:
        for composer, composer_id in composer_to_id.items():
            f.write(f"{composer}\n")
    
    return piece_arrays, composer_vector, piece_names

def save_tensor(piece_arrays, composer_vector, piece_names, tensor_path="./data/midi_pieces.pt", composer_vector_path="./data/composer_vector.pt", piece_names_path="./data/piece_names.txt"):
    """Saves the list of piece tensors, composer vector, and piece names to the specified paths."""
    data_dir = os.path.dirname(tensor_path)
    os.makedirs(data_dir, exist_ok=True)
    torch.save(piece_arrays, tensor_path)  # Save list of piece tensors
    torch.save(composer_vector, composer_vector_path)  # Save composer vector
    
    # Save piece names list as a text file
    with open(piece_names_path, "w") as f:
        for _, piece_name in enumerate(piece_names):
            f.write(f"{piece_name}\n")
    
    print(f"Piece tensors saved to {tensor_path}")
    print(f"Composer vector saved to {composer_vector_path}")
    print(f"Piece names saved to {piece_names_path}")

# Usage example
example_data = True

folder_path = 'surname_checked_midis'
if example_data:
    folder_path = 'midi_example_data'
time_resolution_ms = 50  # Define the time grid in milliseconds

# Process the folder and save the tensor and composer vector
#midi_to_array('surname_checked_midis/Ravel, Maurice, Gaspard de la nuit, jJRnNm_jhEs.mid', vis=True)

piece_arrays, composer_vector, piece_names = process_midi_folder(folder_path, time_resolution_ms)
save_tensor(piece_arrays, composer_vector, piece_names)
