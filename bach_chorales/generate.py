from mido import Message, MidiFile, MidiTrack
from dataclasses import dataclass
from typing import List, Tuple
import os
from utils import *
from utils2 import *
from network import *


def load_trained_model(model_path, device):
    
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # First, initialize the model architecture
    input_dimensions = SEQUENCE_LENGTH * VOICE_RANGES['soprano'].range_and_silence_length()
    model = MultiVoiceHarmonizationNetwork(
        input_dimensions=input_dimensions,
        hidden_layer_size=256,
        dropout_rate=0.4
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"📊 Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

class HarmonyGenerator:
    """Generate harmonies from your trained model"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_note_from_tensor_position(self, tensor_position, voice_range):
        """Convert tensor position to MIDI note number"""
        if tensor_position == 0:  # Silence
            return None
        return tensor_position + voice_range.min_note - 1
    
    def generate_harmony(self, soprano_input):
        """Generate complete SATB harmony from soprano input"""
        with torch.no_grad():
            soprano_input = soprano_input.to(self.device)
            alto_pred, tenor_pred, bass_pred = self.model(soprano_input)
            
            # Use first batch item and transpose for time-first access
            soprano_notes = soprano_input[0]  # [seq_len, encoding_dim]
            alto_notes = alto_pred[0].transpose(0, 1)  # [seq_len, encoding_dim]
            tenor_notes = tenor_pred[0].transpose(0, 1)
            bass_notes = bass_pred[0].transpose(0, 1)
            
            harmony_sequence = []
            
            for time_step in range(SEQUENCE_LENGTH):
                # Get soprano note (find the active note in one-hot encoding)
                soprano_active = torch.argmax(soprano_notes[time_step])
                soprano_note = self.get_note_from_tensor_position(
                    soprano_active.item(), VOICE_RANGES['soprano']
                )
                
                # Get predicted notes for other voices
                alto_active = torch.argmax(alto_notes[time_step])
                alto_note = self.get_note_from_tensor_position(
                    alto_active.item(), VOICE_RANGES['alto']
                )
                
                tenor_active = torch.argmax(tenor_notes[time_step])
                tenor_note = self.get_note_from_tensor_position(
                    tenor_active.item(), VOICE_RANGES['tenor']
                )
                
                bass_active = torch.argmax(bass_notes[time_step])
                bass_note = self.get_note_from_tensor_position(
                    bass_active.item(), VOICE_RANGES['bass']
                )
                
                # Create tuple (filter out None values for now, or keep them)
                notes_tuple = (soprano_note, alto_note, tenor_note, bass_note)
                harmony_sequence.append(notes_tuple)
            
            return harmony_sequence

def convert_harmony_to_note_info(harmony_sequence, time_step_duration=0.25):
    """Convert harmony sequence to NoteInfo objects for each voice"""
    
    voice_note_infos = [[], [], [], []]  # soprano, alto, tenor, bass
    
    for voice_idx in range(4):
        current_note = None
        note_start_time = 0.0
        note_duration = 0.0
        
        for time_idx, chord in enumerate(harmony_sequence):
            current_time = time_idx * time_step_duration
            note_at_time = chord[voice_idx]
            
            if note_at_time != current_note:
                # Save previous note if it exists
                if current_note is not None:
                    note_info = NoteInfo.create(
                        starting_beat=note_start_time,
                        length=note_duration,
                        pitch=current_note
                    )
                    voice_note_infos[voice_idx].append(note_info)
                
                # Start new note
                current_note = note_at_time
                note_start_time = current_time
                note_duration = time_step_duration
            else:
                # Continue current note
                note_duration += time_step_duration
        
        # Don't forget the last note
        if current_note is not None:
            note_info = NoteInfo.create(
                starting_beat=note_start_time,
                length=note_duration,
                pitch=current_note
            )
            voice_note_infos[voice_idx].append(note_info)
    
    return voice_note_infos

class MidiMessageGenerator:
    """Generate MIDI messages from note information"""
    
    def __init__(self, note_infos: List[NoteInfo], track_number=0):
        self.note_infos = note_infos
        self.track_number = track_number
        self.ticks_per_beat = 480  # Standard MIDI resolution
    
    def get_event_positions(self):
        """Get all unique event positions (note starts and ends)"""
        positions = set()
        for note in self.note_infos:
            positions.add(note.starting_beat)
            positions.add(note.starting_beat + note.length)
        return sorted(positions)
    
    def get_notes_at_position(self, position):
        """Get notes that start or end at a specific position"""
        starting_notes = []
        ending_notes = []
        
        for note in self.note_infos:
            if abs(note.starting_beat - position) < 0.01:  # Small tolerance
                starting_notes.append(note)
            if abs((note.starting_beat + note.length) - position) < 0.01:
                ending_notes.append(note)
        
        return starting_notes, ending_notes
    
    def generate_midi_messages(self):
        """Generate MIDI messages for the track"""
        messages = []
        event_positions = self.get_event_positions()
        last_position = 0.0
        
        for position in event_positions:
            # Calculate time delta
            time_delta = position - last_position
            delta_ticks = int(time_delta * self.ticks_per_beat)
            
            starting_notes, ending_notes = self.get_notes_at_position(position)
            
            # Add note_off messages first (use remaining delta time)
            for i, note in enumerate(ending_notes):
                time_offset = delta_ticks if i == 0 else 0
                message = Message('note_off', 
                                note=note.pitch, 
                                velocity=0, 
                                time=time_offset,
                                channel=self.track_number)
                messages.append(message)
            
            # Add note_on messages
            for i, note in enumerate(starting_notes):
                time_offset = delta_ticks if i == 0 and not ending_notes else 0
                message = Message('note_on', 
                                note=note.pitch, 
                                velocity=64, 
                                time=time_offset,
                                channel=self.track_number)
                messages.append(message)
            
            last_position = position
        
        return messages

def create_midi_track(note_infos: List[NoteInfo], track_number=0):
    """Create a MIDI track from note information"""
    track = MidiTrack()
    
    # Add program change (optional - sets instrument)
    track.append(Message('program_change', program=0, time=0, channel=track_number))
    
    # Generate and add note messages
    message_generator = MidiMessageGenerator(note_infos, track_number)
    for message in message_generator.generate_midi_messages():
        track.append(message)
    
    return track

def generate_midi_file(voice_note_infos: List[List[NoteInfo]], filename="generated_harmony.mid"):
    
    midi_file = MidiFile(type=1, ticks_per_beat=480)
    
    voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
    
    for voice_idx, note_infos in enumerate(voice_note_infos):
        if note_infos:  
            track = create_midi_track(note_infos, track_number=voice_idx)
            midi_file.tracks.append(track)
            print(f"Created track for {voice_names[voice_idx]} with {len(note_infos)} notes")
    
    os.makedirs("generated_midi", exist_ok=True)
    filepath = f"generated_midi/{filename}"
    midi_file.save(filepath)
    print(f"MIDI file saved: {filepath}")
    
    return filepath


def generate_midi_from_model(model, test_loader, device, num_songs=5, output_dir="generated_midi"):
    """Generate MIDI files from your trained model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize harmony generator
    harmony_gen = HarmonyGenerator(model, device)
    
    print(f"Generating {num_songs} MIDI files...")
    
    # Get some test data
    test_iter = iter(test_loader)
    
    for song_idx in range(num_songs):
        try:
            # Get next batch
            soprano_input, alto_target, tenor_target, bass_target = next(test_iter)
            
            print(f"\nGenerating song {song_idx + 1}/{num_songs}")
            
            # Generate harmony
            harmony_sequence = harmony_gen.generate_harmony(soprano_input)
            
            # Convert to note info
            voice_note_infos = convert_harmony_to_note_info(harmony_sequence)
            
            # Generate MIDI file
            filename = f"harmony_song_{song_idx + 1:03d}.mid"
            filepath = generate_midi_file(voice_note_infos, filename)
            
            # Also generate just the melody (soprano) for comparison
            melody_filename = f"melody_song_{song_idx + 1:03d}.mid"
            melody_note_infos = [voice_note_infos[0]]  # Just soprano
            generate_midi_file(melody_note_infos, melody_filename)
            
        except StopIteration:
            print(f"Reached end of test data at song {song_idx + 1}")
            break
    
    print(f"\n✅ Generated {song_idx + 1} MIDI files in '{output_dir}' directory")


def run_midi_generation(test_loader=None):
    """Run MIDI generation with your trained model"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Model path
    model_path = "./models/best_harmonization_model.pt"
    
    try:
        # Load the trained model
        harmonization_model = load_trained_model(model_path, device)
        
        # Check if we have test data
        if test_loader is None:
            print("⚠️  Warning: No test_loader provided. You need to load your test data first.")
            print("Example:")
            print("test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)")
            return harmonization_model  # Return model for manual use
        
        # Generate MIDI files
        print("🎵 Starting MIDI generation...")
        generate_midi_from_model(
            model=harmonization_model,
            test_loader=test_loader,
            device=device,
            num_songs=5
        )
        
        return harmonization_model
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please train your model first or check the model path!")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

