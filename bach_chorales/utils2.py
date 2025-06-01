import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter


SEQUENCE_LENGTH = 64  
BATCH_SIZE = 4
SILENCE_INDEX = 0

class VoiceRange:
    """Define the range for each voice"""
    def __init__(self, min_note, max_note):
        self.min_note = min_note
        self.max_note = max_note
    
    def range_and_silence_length(self):
        """Total encoding size including silence"""
        return (self.max_note - self.min_note + 1) + 1
    
    def is_in_range(self, note):
        return self.min_note <= note <= self.max_note

VOICE_RANGES = {
    'soprano': VoiceRange(61 - 5, 81 + 6),  # Range: 56-87
    'alto': VoiceRange(56 - 5, 76 + 6),     # Range: 51-82  
    'tenor': VoiceRange(51 - 5, 71 + 6),    # Range: 46-77
    'bass': VoiceRange(36 - 5, 63 + 6)      # Range: 31-69
}

# =============================================================================
# STEP 3: Voice Assignment Logic
# =============================================================================

class VoiceTracker:
    """Track and assign notes to voices based on previous context"""
    
    def __init__(self, voice_name, voice_range, voice_index):
        self.voice_name = voice_name
        self.voice_range = voice_range
        self.voice_index = voice_index
        self.prev_note = None
        self.prev_chord = None
    
    def get_voice_note(self, chord):
        """Extract the most likely note for this voice from a chord"""
        
        if not chord or len(chord) == 0:
            return -1  # Silence
        
        # Convert chord to list of integers (handle any numpy types)
        chord = [int(note) for note in chord if note is not None]
        
        if not chord:
            return -1
        
        # If chord has 4 notes (SATB), use direct mapping
        if len(chord) == 4:
            note = chord[self.voice_index]
            return note if self.voice_range.is_in_range(note) else -1
        
        # For incomplete chords, use heuristics
        if self.prev_note is not None and self.prev_note in chord:
            return self.prev_note
        
        # Find closest note to previous note that's in range
        if self.prev_note is not None:
            valid_notes = [n for n in chord if self.voice_range.is_in_range(n)]
            if valid_notes:
                closest = min(valid_notes, key=lambda n: abs(n - self.prev_note))
                self.prev_note = closest
                self.prev_chord = chord
                return closest
        
        # Fallback: try to assign based on typical voice ranges
        range_candidates = [n for n in chord if self.voice_range.is_in_range(n)]
        if range_candidates:
            # For soprano: highest, for bass: lowest, for middle voices: middle
            if self.voice_name == 'soprano':
                note = max(range_candidates)
            elif self.voice_name == 'bass':
                note = min(range_candidates)
            else:
                note = sorted(range_candidates)[len(range_candidates)//2]
            
            self.prev_note = note
            self.prev_chord = chord
            return note
        
        return -1  # Silence if no suitable note found
    



# =============================================================================
# STEP 4: One-Hot Encoding
# =============================================================================

def song_to_one_hot(song, voice_name, voice_range):
    """Convert a song to one-hot encoding for a specific voice"""
    
    voice_tracker = VoiceTracker(voice_name, voice_range, 
                                list(VOICE_RANGES.keys()).index(voice_name))
    
    # Create one-hot matrix
    encoding_size = voice_range.range_and_silence_length()
    one_hot = np.zeros((len(song), encoding_size))
    
    for i, chord in enumerate(song):
        note = voice_tracker.get_voice_note(chord)
        
        if note == -1 or note is None:
            # Silence
            one_hot[i][SILENCE_INDEX] = 1
        else:
            # Note position (offset by 1 for silence)
            note_pos = int(note - voice_range.min_note + 1)  # Convert to int
            if 0 <= note_pos < encoding_size:
                one_hot[i][note_pos] = 1
            else:
                # Out of range, mark as silence
                print(f"Warning: Note {note} out of range for {voice_name} (range: {voice_range.min_note}-{voice_range.max_note})")
                one_hot[i][SILENCE_INDEX] = 1
    
    return one_hot

def process_songs_to_one_hot(songs, max_songs=None):
    """Convert multiple songs to one-hot encoding for all voices"""
    
    if max_songs:
        songs = songs[:max_songs]
    
    voice_data = {voice: [] for voice in VOICE_RANGES.keys()}
    
    print(f"Converting {len(songs)} songs to one-hot encoding...")
    
    for i, song in enumerate(songs):
        if i % 100 == 0:
            print(f"  Processing song {i+1}/{len(songs)}")
        
        for voice_name, voice_range in VOICE_RANGES.items():
            one_hot = song_to_one_hot(song, voice_name, voice_range)
            voice_data[voice_name].append(one_hot)
    
    return voice_data



# =============================================================================
# STEP 5: Sequence Splitting
# =============================================================================

def split_into_sequences(one_hot_songs, sequence_length=SEQUENCE_LENGTH):
    """Split long songs into fixed-length sequences"""
    
    sequences = []
    
    for song in one_hot_songs:
        song_length = song.shape[0]
        
        # Create overlapping windows
        for start_idx in range(0, song_length - sequence_length + 1, sequence_length):
            end_idx = start_idx + sequence_length
            
            if end_idx <= song_length:
                sequence = song[start_idx:end_idx]
                sequences.append(sequence)
    
    return np.array(sequences)

def create_sequence_dataset(voice_data):
    """Create sequences for all voices"""
    
    sequence_data = {}
    
    for voice_name, songs in voice_data.items():
        print(f"Splitting {voice_name} into sequences...")
        sequences = split_into_sequences(songs)
        sequence_data[voice_name] = sequences
        print(f"  Created {len(sequences)} sequences of length {SEQUENCE_LENGTH}")
    
    return sequence_data



def encode_to_target_indexes(one_hot_sequences):
    """Convert one-hot sequences to target indexes for loss calculation"""
    
    target_indexes = []
    
    for sequence in one_hot_sequences:
        sequence_indexes = []
        for time_step in sequence:
            # Find the index of the active note (should be exactly one 1.0)
            active_idx = np.where(time_step == 1.0)[0]
            if len(active_idx) > 0:
                sequence_indexes.append(active_idx[0])
            else:
                sequence_indexes.append(0)  # Default to silence
        target_indexes.append(sequence_indexes)
    
    return np.array(target_indexes)

class ChoralesDataset(Dataset):
    """PyTorch Dataset for Bach Chorales"""
    
    def __init__(self, sequence_data):
        # Input: Soprano (one-hot)
        self.X_soprano = torch.tensor(sequence_data['soprano'], dtype=torch.float32)
        
        # Targets: Alto, Tenor, Bass (as class indexes)
        self.Y_alto = torch.tensor(
            encode_to_target_indexes(sequence_data['alto']), 
            dtype=torch.long
        )
        self.Y_tenor = torch.tensor(
            encode_to_target_indexes(sequence_data['tenor']), 
            dtype=torch.long
        )
        self.Y_bass = torch.tensor(
            encode_to_target_indexes(sequence_data['bass']), 
            dtype=torch.long
        )
        
        self.length = len(sequence_data['soprano'])
        
        print(f"Dataset created with {self.length} sequences")
        print(f"  Input shape: {self.X_soprano.shape}")
        print(f"  Target shapes: {self.Y_alto.shape}, {self.Y_tenor.shape}, {self.Y_bass.shape}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return (
            self.X_soprano[idx],
            self.Y_alto[idx], 
            self.Y_tenor[idx],
            self.Y_bass[idx]
        )

def create_data_loaders(train_dataset, test_dataset, valid_dataset, 
                       batch_size=BATCH_SIZE, shuffle=True):
    """Create PyTorch DataLoaders"""
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches of size {batch_size}")
    print(f"  Test: {len(test_loader)} batches of size {batch_size}")
    print(f"  Valid: {len(valid_loader)} batches of size {batch_size}")
    
    return train_loader, test_loader, valid_loader

def validate_data_loader(data_loader, name="Dataset"):
    """Validate that the data loader works correctly"""
    
    print(f"\n🔍 Validating {name}...")
    
    # Test loading one batch
    try:
        batch = next(iter(data_loader))
        x_soprano, y_alto, y_tenor, y_bass = batch
        
        print(f"✅ Batch loaded successfully!")
        print(f"  Soprano input: {x_soprano.shape} (should be [{BATCH_SIZE}, {SEQUENCE_LENGTH}, {VOICE_RANGES['soprano'].range_and_silence_length()}])")
        print(f"  Alto target: {y_alto.shape} (should be [{BATCH_SIZE}, {SEQUENCE_LENGTH}])")
        print(f"  Tenor target: {y_tenor.shape}")
        print(f"  Bass target: {y_bass.shape}")
        
        # Check data ranges
        print(f"  Alto target range: {y_alto.min().item()} to {y_alto.max().item()}")
        print(f"  Soprano input sum per timestep (should be 1.0): {x_soprano[0, 0].sum().item():.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def visualize_training_data(data_loader):
    """Visualize some training examples"""
    
    # Get one batch
    batch = next(iter(data_loader))
    x_soprano, y_alto, y_tenor, y_bass = batch
    
    # Convert soprano back to note numbers for visualization
    def one_hot_to_notes(one_hot_seq, voice_range):
        notes = []
        for timestep in one_hot_seq:
            active_idx = torch.argmax(timestep).item()
            if active_idx == 0:
                notes.append(None)  # Silence
            else:
                note = active_idx + voice_range.min_note - 1
                notes.append(note)
        return notes
    
    # Visualize first sequence in batch
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Soprano (input)
    soprano_notes = one_hot_to_notes(x_soprano[0], VOICE_RANGES['soprano'])
    soprano_clean = [n for n in soprano_notes if n is not None]
    
    axes[0, 0].plot(range(len(soprano_notes)), 
                   [n if n is not None else 0 for n in soprano_notes], 'b-o')
    axes[0, 0].set_title('Soprano (Input)')
    axes[0, 0].set_ylabel('MIDI Note')
    axes[0, 0].grid(True)
    
    # Targets (convert indexes back to notes)
    target_voices = [
        (y_alto[0], 'Alto', VOICE_RANGES['alto']),
        (y_tenor[0], 'Tenor', VOICE_RANGES['tenor']),
        (y_bass[0], 'Bass', VOICE_RANGES['bass'])
    ]
    
    for i, (target_seq, voice_name, voice_range) in enumerate(target_voices):
        ax = axes[0, 1] if i == 0 else axes[1, i-1]
        
        # Convert target indexes to notes
        target_notes = []
        for target_idx in target_seq:
            if target_idx.item() == 0:
                target_notes.append(None)  # Silence
            else:
                note = target_idx.item() + voice_range.min_note - 1
                target_notes.append(note)
        
        ax.plot(range(len(target_notes)), 
               [n if n is not None else 0 for n in target_notes], 'r-o')
        ax.set_title(f'{voice_name} (Target)')
        ax.set_ylabel('MIDI Note')
        ax.set_xlabel('Time Step')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Sample sequence statistics:")
    print(f"  Soprano range: {min(soprano_clean)} - {max(soprano_clean)}")
    print(f"  Unique notes: {len(set(soprano_clean))}")