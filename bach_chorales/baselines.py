import torch
import numpy as np
import random
from collections import Counter



class RandomBaseline:
    """Generates completely random harmonies within valid voice ranges."""
    
    def __init__(self, voice_ranges):
        self.voice_ranges = voice_ranges
        
    def evaluate(self, test_loader, device):
        """Evaluate random baseline on test set"""
        correct_predictions = 0
        total_predictions = 0
        
        for soprano_input, alto_target, tenor_target, bass_target in test_loader:
            batch_size, sequence_length = alto_target.shape
            
            for sample_idx in range(batch_size):
                # Generate random harmony for this sequence
                for time_idx in range(sequence_length):
                    # Generate random predictions
                    alto_pred = random.randint(0, self.voice_ranges['alto'].range_and_silence_length() - 1)
                    tenor_pred = random.randint(0, self.voice_ranges['tenor'].range_and_silence_length() - 1)
                    bass_pred = random.randint(0, self.voice_ranges['bass'].range_and_silence_length() - 1)
                    
                    # Check if predictions match targets
                    if alto_pred == alto_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    if tenor_pred == tenor_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    if bass_pred == bass_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    
                    total_predictions += 3
        
        accuracy = correct_predictions / total_predictions
        return accuracy

class MostCommonBaseline:
    """Always predicts the most common note for each voice."""
    
    def __init__(self):
        self.most_common_notes = {}
        
    def fit(self, train_loader):
        """Learn the most common notes from training data"""
        note_counts = {
            'alto': Counter(),
            'tenor': Counter(), 
            'bass': Counter()
        }
        
        for soprano_input, alto_target, tenor_target, bass_target in train_loader:
            # Count all notes in the batch
            note_counts['alto'].update(alto_target.flatten().tolist())
            note_counts['tenor'].update(tenor_target.flatten().tolist())
            note_counts['bass'].update(bass_target.flatten().tolist())
        
        # Find most common note for each voice
        self.most_common_notes = {
            'alto': note_counts['alto'].most_common(1)[0][0],
            'tenor': note_counts['tenor'].most_common(1)[0][0],
            'bass': note_counts['bass'].most_common(1)[0][0]
        }
        
        print(f"Most common notes: {self.most_common_notes}")
    
    def evaluate(self, test_loader):
        """Evaluate most common baseline"""
        correct_predictions = 0
        total_predictions = 0
        
        for soprano_input, alto_target, tenor_target, bass_target in test_loader:
            batch_size, sequence_length = alto_target.shape
            
            for sample_idx in range(batch_size):
                for time_idx in range(sequence_length):
                    # Use most common notes as predictions
                    alto_pred = self.most_common_notes['alto']
                    tenor_pred = self.most_common_notes['tenor']
                    bass_pred = self.most_common_notes['bass']
                    
                    # Check if predictions match targets
                    if alto_pred == alto_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    if tenor_pred == tenor_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    if bass_pred == bass_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    
                    total_predictions += 3
        
        accuracy = correct_predictions / total_predictions
        return accuracy



class RuleBasedBaseline:
    """Rule-based harmonization using basic music theory."""
    
    def __init__(self, voice_ranges):
        self.voice_ranges = voice_ranges
        
        # Basic chord patterns (intervals from soprano in semitones)
        self.chord_patterns = [
            (-5, -9, -12),   # Major chord pattern
            (-3, -7, -12),   # Minor chord variant
            (-4, -8, -12),   # Another common pattern
            (-5, -8, -15),   # Wider spacing
            (-3, -10, -12),  # Alternative voicing
        ]
    
    def soprano_index_to_midi(self, soprano_index):
        """Convert soprano encoding index to MIDI note"""
        if soprano_index == 0:  # Silence
            return None
        return soprano_index + self.voice_ranges['soprano'].min_note - 1
    
    def midi_to_voice_index(self, midi_note, voice_name):
        """Convert MIDI note to voice encoding index"""
        if midi_note is None:
            return 0  # Silence
        
        voice_range = self.voice_ranges[voice_name]
        if midi_note < voice_range.min_note or midi_note > voice_range.max_note:
            return 0  # Out of range, use silence
        
        return midi_note - voice_range.min_note + 1
    
    def evaluate(self, test_loader):
        """Evaluate rule-based baseline"""
        correct_predictions = 0
        total_predictions = 0
        
        for soprano_input, alto_target, tenor_target, bass_target in test_loader:
            batch_size, sequence_length = alto_target.shape
            
            for sample_idx in range(batch_size):
                # Convert soprano from one-hot to indices
                soprano_indices = torch.argmax(soprano_input[sample_idx], dim=1)
                
                for time_idx in range(sequence_length):
                    soprano_idx = soprano_indices[time_idx].item()
                    soprano_midi = self.soprano_index_to_midi(soprano_idx)
                    
                    if soprano_midi is None:
                        # Soprano is silent, predict silence for all voices
                        alto_pred = 0
                        tenor_pred = 0
                        bass_pred = 0
                    else:
                        # Choose a chord pattern (cycle through them)
                        pattern = self.chord_patterns[time_idx % len(self.chord_patterns)]
                        
                        # Generate harmony notes
                        alto_midi = soprano_midi + pattern[0]
                        tenor_midi = soprano_midi + pattern[1]
                        bass_midi = soprano_midi + pattern[2]
                        
                        # Convert to voice indices
                        alto_pred = self.midi_to_voice_index(alto_midi, 'alto')
                        tenor_pred = self.midi_to_voice_index(tenor_midi, 'tenor')
                        bass_pred = self.midi_to_voice_index(bass_midi, 'bass')
                    
                    # Check if predictions match targets
                    if alto_pred == alto_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    if tenor_pred == tenor_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    if bass_pred == bass_target[sample_idx][time_idx].item():
                        correct_predictions += 1
                    
                    total_predictions += 3
        
        accuracy = correct_predictions / total_predictions
        return accuracy


def evaluate_three_baselines(train_loader, test_loader, voice_ranges, device):
    """Evaluate the three baseline models."""
    
    print("Evaluating Three Baseline Models...")
    print("=" * 50)
    
    results = {}
    
    # 1. Random Baseline
    print("1. Random Baseline...")
    random_baseline = RandomBaseline(voice_ranges)
    results['Random'] = random_baseline.evaluate(test_loader, device)
    print(f"   Accuracy: {results['Random']:.4f}")
    
    # 2. Most Common Note Baseline
    print("\n2. Most Common Note Baseline...")
    common_baseline = MostCommonBaseline()
    common_baseline.fit(train_loader)
    results['Most Common'] = common_baseline.evaluate(test_loader)
    print(f"   Accuracy: {results['Most Common']:.4f}")
    
    # 3. Rule-Based Baseline
    print("\n3. Rule-Based Baseline...")
    rule_baseline = RuleBasedBaseline(voice_ranges)
    results['Rule-Based'] = rule_baseline.evaluate(test_loader)
    print(f"   Accuracy: {results['Rule-Based']:.4f}")
    
    print("\n" + "=" * 50)
    print("BASELINE COMPARISON:")
    for name, accuracy in results.items():
        print(f"  {name:15s}: {accuracy:.4f}")
    
    return results