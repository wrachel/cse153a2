import pickle
import numpy as np
import random
from typing import List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

## Classes

class NoteInfo:
    """Recreation of the NoteInfo class from the original codebase"""
    
    def __init__(self, note_data):
        self.starting_beat = float(note_data[0])
        self.pitch = int(note_data[1])
        self.length = float(note_data[2])
        self.midi_channel = note_data[3] if len(note_data) > 3 else 0
    
    def is_on_at_beat(self, beat):
        """Check if note is playing at given beat"""
        ROUND_ERROR = 1/50
        SUBDIV = 12
        
        if beat < self.starting_beat - ROUND_ERROR:
            return False
        
        inclusive_end = (self.starting_beat + self.length - 
                        (1/SUBDIV) + 
                        ROUND_ERROR)
        
        if beat > inclusive_end:
            return False
        
        return True
    
    def with_new_pitch(self, new_pitch):
        """Create a copy with new pitch"""
        return NoteInfo.create(self.starting_beat, self.length, new_pitch)
    
    def copy(self):
        """Create a copy of this note"""
        return NoteInfo.create(self.starting_beat, self.length, self.pitch)
    
    @classmethod
    def create(cls, starting_beat, length=1, pitch=0):
        """Factory method to create NoteInfo"""
        return cls([starting_beat, pitch, length, 0])
    
    def __str__(self):
        return f"NoteInfo(beat={self.starting_beat}, pitch={self.pitch}, length={self.length})"





def analyze_dataset(midi_data):
    """Analyze the structure and content of the dataset"""
    
    # Collect statistics
    all_songs = midi_data['train'] + midi_data['test'] + midi_data['valid']
    song_lengths = [len(song) for song in all_songs]
    
    # Note range analysis
    all_notes = []
    for song in all_songs:
        for chord in song:
            all_notes.extend(chord)
    
    print(f"Dataset Statistics:")
    print(f"  Total songs: {len(all_songs)}")
    print(f"  Average song length: {np.mean(song_lengths):.1f} time steps")
    print(f"  Note range: {min(all_notes)} to {max(all_notes)}")
    print(f"  Total notes: {len(all_notes)}")
    
    # Plot song length distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(song_lengths, bins=20, alpha=0.7)
    plt.xlabel('Song Length (time steps)')
    plt.ylabel('Count')
    plt.title('Distribution of Song Lengths')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_notes, bins=50, alpha=0.7)
    plt.xlabel('MIDI Note Number')
    plt.ylabel('Count')
    plt.title('Distribution of Notes')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'song_lengths': song_lengths,
        'note_range': (min(all_notes), max(all_notes)),
        'total_songs': len(all_songs)
    }


def get_next_note_duration_and_note(dataset_song, track_number: int):

    """Generator that yields (position, duration, note) for each note change"""
    cur_pos = 0
    cur_note_pos = 0
    cur_length = 0
    cur_note_number = (dataset_song[0][track_number] 
                      if len(dataset_song[0]) > track_number else None)
    
    for note_index in range(len(dataset_song)):
        current_notes_tuple = dataset_song[note_index]
        next_note_number = (current_notes_tuple[track_number] 
                           if len(current_notes_tuple) > track_number else None)
        
        if next_note_number != cur_note_number:
            yield (cur_note_pos, cur_length, cur_note_number)
            cur_length = 0
            cur_note_pos = cur_pos
        
        cur_note_number = next_note_number
        cur_length += 0.25
        cur_pos += 0.25

def get_track_notes(dataset_song, track_number: int) -> List[NoteInfo]:
    """Extract notes for a specific track/voice"""
    note_infos = []
    
    for pos, note_length, note_number in get_next_note_duration_and_note(dataset_song, track_number):
        if note_number is None:
            continue
        
        note_info = NoteInfo.create(pos, note_length, note_number)
        note_infos.append(note_info)
    
    return note_infos

def generate_note_info(dataset_song) -> List[List[NoteInfo]]:
    """Convert a song from tuple format to NoteInfo format for all 4 voices"""
    track_note_infos = []
    
    for track_number in range(4):  # SATB
        note_infos = get_track_notes(dataset_song, track_number)
        track_note_infos.append(note_infos)
    
    return track_note_infos

def get_notes_transformed(song_voices: List[List[NoteInfo]], scale_transform: int) -> List[List[NoteInfo]]:
    """Transpose all notes by given number of semitones"""
    voices_augmented = []
    
    for voice_notes in song_voices:
        notes_augmented = []
        for note_info in voice_notes:
            new_pitch = note_info.pitch + scale_transform
            note_augmented = note_info.with_new_pitch(new_pitch)
            notes_augmented.append(note_augmented)
        voices_augmented.append(notes_augmented)
    
    return voices_augmented

def augment_scales(song_voices: List[List[NoteInfo]]) -> List[List[List[NoteInfo]]]:
    """Create scale augmentations from -5 to +6 semitones"""
    scale_augmentations = []
    
    for scale_transform in range(-5, 7):  # -5 to +6 inclusive (12 total)
        scale_augmentation = get_notes_transformed(song_voices, scale_transform)
        scale_augmentations.append(scale_augmentation)
    
    return scale_augmentations



def get_notes_joined(song_voices: List[List[NoteInfo]], modification_prob: float) -> List[List[NoteInfo]]:
    """Join adjacent notes randomly based on modification probability"""
    voices_notes_joined = []
    
    for voice_notes in song_voices:
        voice_notes_joined = []
        
        for note in voice_notes:
            # Only consider notes not on measure beats that are smaller than one beat
            if note.starting_beat % 1.0 == 0 or note.length >= 1.0:
                voice_notes_joined.append(note.copy())
                continue
            
            keep_prob = random.uniform(0, 1)
            keep = keep_prob > modification_prob
            
            if not keep and len(voice_notes_joined) > 0:
                # Join note duration to last note
                last_note = voice_notes_joined[-1]
                last_note.length += note.length
                continue
            
            voice_notes_joined.append(note.copy())
        
        voices_notes_joined.append(voice_notes_joined)
    
    return voices_notes_joined

def augment_note_joins(song_voices: List[List[NoteInfo]]) -> List[List[List[NoteInfo]]]:
    """Create note join augmentations"""
    # Keep vanilla data
    note_join_augmentations = [song_voices]
    
    # Create 3 additional variations with random modification probabilities
    for _ in range(3):
        modification_prob = random.uniform(0, 1)
        note_join_augmentations.append(get_notes_joined(song_voices, modification_prob))
    
    return note_join_augmentations



def augment_song_data(song_voices: List[List[NoteInfo]]) -> List[List[List[NoteInfo]]]:
    """Apply both scale and note join augmentations"""
    
    # First, create scale augmentations
    scale_augmentations = augment_scales(song_voices)
    
    # Then, apply note join augmentations to each scale augmentation
    all_augmentations = []
    
    for scale_augmentation in scale_augmentations:
        note_join_augmentations = augment_note_joins(scale_augmentation)
        all_augmentations.extend(note_join_augmentations)
    
    return all_augmentations




def get_last_pos(voices_note_infos: List[List[NoteInfo]]) -> float:
    """Find the end position of the longest note across all voices"""
    last_pos = 0
    
    for voice_notes in voices_note_infos:
        for note_info in voice_notes:
            note_end = note_info.starting_beat + note_info.length
            last_pos = max(last_pos, note_end)
    
    return last_pos

def get_notes_on_at_position(voices_note_infos: List[List[NoteInfo]], position: float):
    """Generator that yields notes playing at a given position"""
    for voice_notes in voices_note_infos:
        for note_info in voice_notes:
            if note_info.is_on_at_beat(position):
                yield note_info

def get_voice_tuples(voices_note_infos: List[List[NoteInfo]]):
    """Fixed version that preserves voice ordering"""
    last_pos = get_last_pos(voices_note_infos)
    
    for cur_pos in np.arange(0, last_pos, 0.25):
        # Get notes for each voice separately, in order
        voice_notes = []
        
        for voice_idx in range(4):  # Soprano, Alto, Tenor, Bass
            voice_note_infos = voices_note_infos[voice_idx]
            
            # Find note playing in this voice at this time
            note_for_voice = None
            for note_info in voice_note_infos:
                if note_info.is_on_at_beat(cur_pos):
                    note_for_voice = note_info.pitch
                    break
            
            voice_notes.append(note_for_voice)
        
        # Filter out None values and create tuple
        filtered_notes = [note for note in voice_notes if note is not None]
        yield tuple(filtered_notes)

def generate_tuple_form(voices_note_infos: List[List[NoteInfo]]) -> List[Tuple[int]]:
    """Fixed version that preserves voice ordering"""
    song_tuple_form = []
    
    for current_voice_tuple in get_voice_tuples(voices_note_infos):
        song_tuple_form.append(current_voice_tuple)
    
    return song_tuple_form
# def generate_tuple_form(voices_note_infos: List[List[NoteInfo]]) -> List[Tuple[int]]:
#     """Debug version with logging"""
    
#     last_pos = get_last_pos(voices_note_infos)
 
    
#     song_tuple_form = []
#     step_count = 0
    
#     for cur_pos in np.arange(0, last_pos, 0.25):
#         notes_on_pos = list(get_notes_on_at_position(voices_note_infos, cur_pos))
#         note_numbers = tuple(note.pitch for note in notes_on_pos)
#         song_tuple_form.append(note_numbers)
#         step_count += 1
        
#         if step_count <= 5 or step_count % 50 == 0:
#             print(f"  Step {step_count}: pos={cur_pos:.2f}, notes={note_numbers}")
    
#     return song_tuple_form





def augment_full_dataset(midi_data, max_songs_per_split=None):
    """Apply full augmentation pipeline to entire dataset"""
    
    print("🚀 Starting full dataset augmentation...")
    
    augmented_data = {
        'train': [],
        'test': midi_data['test'],     
        'valid': midi_data['valid']     
    }
    
    train_songs = midi_data['train']
    if max_songs_per_split:
        train_songs = train_songs[:max_songs_per_split]
        print(f"  Processing subset of {len(train_songs)} train songs")
    
    total_songs = len(train_songs)
    
    for index, song_tuple_data in enumerate(train_songs):
        if (index + 1) % 10 == 0:
            print(f'  Augmenting song {index + 1} of {total_songs} songs')
        
        voice_note_infos = generate_note_info(song_tuple_data)
        
        song_augmentations = augment_song_data(voice_note_infos)
        
        for song_augmentation in song_augmentations:
            tuple_augmentation = generate_tuple_form(song_augmentation)
            augmented_data['train'].append(tuple_augmentation)
    
    original_count = len(midi_data['train'])
    augmented_count = len(augmented_data['train'])
    
    print(f"\n✅ Augmentation completed!")
    print(f"  Original training songs: {original_count}")
    print(f"  Augmented training songs: {augmented_count}")
    print(f"  Augmentation factor: {augmented_count / len(train_songs):.1f}x")
    
    return augmented_data





def save_augmented_dataset(augmented_data, filename='scales_note_join_augmented.pkl'):
    """Save the augmented dataset to pickle file"""
    
    Path('./data').mkdir(exist_ok=True)
    
    filepath = f'./data/{filename}'
    
    with open(filepath, 'wb') as handle:
        pickle.dump(augmented_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"💾 Augmented dataset saved to: {filepath}")
    
    file_size = Path(filepath).stat().st_size / (1024 * 1024)  
    print(f"  File size: {file_size:.1f} MB")
    
    return filepath



def verify_saved_data(filepath):
    """Load and verify the saved augmented dataset"""
    
    print(f"\n🔍 Verifying saved data...")
    
    try:
        with open(filepath, 'rb') as file:
            loaded_data = pickle.load(file, encoding="latin1")
        
        print(f"✅ Successfully loaded augmented dataset")
        print(f"  Train: {len(loaded_data['train'])} songs")
        print(f"  Test: {len(loaded_data['test'])} songs")
        print(f"  Valid: {len(loaded_data['valid'])} songs")
        
        sample_song = loaded_data['train'][0]
        print(f"  Sample song length: {len(sample_song)} time steps")
        print(f"  Sample first chord: {sample_song[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading saved data: {e}")
        return False

