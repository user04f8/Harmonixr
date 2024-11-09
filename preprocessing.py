import os
import numpy as np
import torch
import mido
from collections import defaultdict
from tqdm import tqdm

def parse_filename(filename):
    """Parses the composer and piece name from the filename."""
    parts = filename.split(',')
    composer_last = parts[0].strip()
    composer_first = parts[1].strip()
    composer_name = f"{composer_last} {composer_first}"
    piece_name = ','.join(parts[2:]).strip().replace('.mid', '')
    return composer_name, piece_name

def transpose_pitch(midi_note):
    """Transpose MIDI pitch to start at F within each octave (e.g., F, F#, G, ..., E)."""
    pitch_class = (midi_note - 5) % 12  # Shift C-based octave to start from F
    return pitch_class

def midi_to_array(filepath, time_resolution_ms=50):
    """Encodes MIDI file data into a numpy array for pitch, octave, and time within a 6-octave range (F1 to E7)."""
    
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

    # Process MIDI events and fill in the array
    current_time_ms = 0
    active_notes = {}  # Track active notes by (pitch, octave)
    
    for track in mid.tracks:
        for msg in track:
            current_time_ms += int(msg.time * ms_per_tick)
            time_index = int(current_time_ms / time_resolution_ms)
            
            if msg.type == 'note_on' and msg.velocity > 0:
                pitch = transpose_pitch(msg.note)  # Transpose pitch to F-based octave
                octave = (msg.note // 12)
                if min_midi_note <= msg.note <= max_midi_note and octave in octave_range:
                    active_notes[(pitch, octave)] = time_index
            
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                pitch = transpose_pitch(msg.note)
                octave = (msg.note // 12)
                if (pitch, octave) in active_notes and min_midi_note <= msg.note <= max_midi_note:
                    start_time = active_notes.pop((pitch, octave))
                    for t in range(start_time, time_index):
                        midi_array[pitch, octave - 1, t] = 1  # Offset by 1 to match F1 as Octave 1
    
    return midi_array

def process_midi_folder(folder_path, time_resolution_ms=10):
    """Processes each MIDI file, creates a list of tensors (no padding) and saves a composer embedding vector."""
    composer_to_id = {}
    piece_arrays = []
    composer_ids = []

    # Get list of all MIDI files
    midi_files = [f for f in os.listdir(folder_path) if f.endswith('.mid')]
    
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

    # Save composer embedding vector as a tensor
    composer_vector = torch.tensor(composer_ids, dtype=torch.long)
    
    # Save composer-to-id mapping
    with open("composer_mapping.txt", "w") as f:
        for composer, composer_id in composer_to_id.items():
            f.write(f"{composer_id}: {composer}\n")
    
    return piece_arrays, composer_vector

def save_tensor(piece_arrays, composer_vector, tensor_path="midi_pieces.pt", composer_vector_path="composer_vector.pt"):
    """Saves the list of piece tensors and composer vector to the specified paths."""
    torch.save(piece_arrays, tensor_path)  # Save list of piece tensors
    torch.save(composer_vector, composer_vector_path)  # Save composer vector
    print(f"Piece tensors saved to {tensor_path}")
    print(f"Composer vector saved to {composer_vector_path}")

# Usage example
folder_path = 'surname_checked_midis'
time_resolution_ms = 50  # Define the time grid in milliseconds

# Process the folder and save the tensor and composer vector
piece_arrays, composer_vector = process_midi_folder(folder_path, time_resolution_ms)
save_tensor(piece_arrays, composer_vector)
