import mido
from mido import MidiFile, MidiTrack, Message
import os

def add_silence_to_track(track, silence_duration_ticks):
    """Add silence to the end of a track by extending the last message's time"""
    if len(track) > 0:
        # Add silence by creating a dummy message with the silence duration
        # We'll use a meta message that doesn't affect playback
        silence_msg = Message('note_off', note=60, velocity=0, time=silence_duration_ticks, channel=0)
        track.append(silence_msg)

def get_track_duration_ticks(track):
    """Calculate the total duration of a track in ticks"""
    total_ticks = 0
    for msg in track:
        total_ticks += msg.time
    return total_ticks

def combine_midi_files(melody_files, harmony_files, output_file, silence_between_files=480):
    """
    Combine multiple MIDI files into one, alternating between melody and harmony
    
    Args:
        melody_files: List of melody MIDI file paths
        harmony_files: List of harmony MIDI file paths  
        output_file: Output MIDI file path
        silence_between_files: Silence duration between files in ticks (480 = quarter note at 480 PPQ)
    """
    
    # Create new MIDI file
    combined_midi = MidiFile(type=1)  # Type 1 supports multiple tracks
    
    # We'll combine all files into tracks, keeping track structure
    all_tracks = {}  # track_index -> combined_track
    
    def process_file(filename, file_type):
        filename = './generated_midi/' + filename
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping...")
            return
            
        print(f"  Adding {filename} ({file_type})...")
        midi_file = MidiFile(filename)
        
        # Calculate offset for this file (sum of all previous files + silences)
        current_offset = 0
        if len(all_tracks) > 0:
            # Find the maximum duration across all existing tracks
            max_duration = max(get_track_duration_ticks(track) for track in all_tracks.values())
            current_offset = max_duration + silence_between_files
        
        # Process each track in the current file
        for track_idx, track in enumerate(midi_file.tracks):
            if track_idx not in all_tracks:
                all_tracks[track_idx] = MidiTrack()
            
            # If this isn't the first file, add silence to align timing
            if current_offset > 0 and len(all_tracks[track_idx]) == 0:
                # Add initial silence to this track
                all_tracks[track_idx].append(Message('note_off', note=60, velocity=0, time=current_offset, channel=0))
            elif current_offset > 0:
                # Calculate how much silence this track needs
                current_track_duration = get_track_duration_ticks(all_tracks[track_idx])
                needed_silence = current_offset - current_track_duration
                if needed_silence > 0:
                    add_silence_to_track(all_tracks[track_idx], needed_silence)
            
            # Add all messages from current track, adjusting timing for first message
            first_message = True
            for msg in track:
                if first_message and current_offset > 0 and len(all_tracks[track_idx]) > 0:
                    # For the first message, we don't add the offset since we already added silence
                    new_msg = msg.copy()
                    first_message = False
                else:
                    new_msg = msg.copy()
                    first_message = False
                
                all_tracks[track_idx].append(new_msg)
    
    # Process files in alternating order: melody001, harmony001, melody002, harmony002, etc.
    print("\nProcessing files in alternating order:")
    max_files = max(len(melody_files), len(harmony_files))
    
    for i in range(max_files):
        if i < len(melody_files):
            process_file(melody_files[i], "melody")
        if i < len(harmony_files):
            process_file(harmony_files[i], "harmony")
    
    # Add all tracks to the combined MIDI file
    for track_idx in sorted(all_tracks.keys()):
        combined_midi.tracks.append(all_tracks[track_idx])
    
    # Save the combined file
    combined_midi.save(output_file)
    print(f"\nCombined MIDI saved as: {output_file}")
    
    # Print some stats
    total_duration_ticks = max(get_track_duration_ticks(track) for track in all_tracks.values()) if all_tracks else 0
    total_duration_seconds = mido.tick2second(total_duration_ticks, combined_midi.ticks_per_beat, 500000)  # 500000 = default tempo
    print(f"Total duration: {total_duration_seconds:.2f} seconds")
    print(f"Number of tracks: {len(combined_midi.tracks)}")


def main():
    # Define your MIDI files
    melody_files = [
        "melody_song_003.mid",
        "melody_song_001.mid",
        "melody_song_002.mid", 
        "melody_song_004.mid",
        "melody_song_005.mid"
    ]
    
    harmony_files = [
        "harmony_song_003.mid", 
        "harmony_song_001.mid",
        "harmony_song_002.mid",
        "harmony_song_004.mid",
        "harmony_song_005.mid"
    ]
    
    output_file = "combined_melody_harmony.mid"
    
    # Combine the files
    # silence_between_files is in ticks (480 = quarter note at standard 480 PPQ)
    # Adjust this value to change the silence duration between pieces
    combine_midi_files(melody_files, harmony_files, output_file, silence_between_files=960)  # Half note of silence
    
if __name__ == "__main__":
    main()