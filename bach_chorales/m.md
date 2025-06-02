# Bach Chorales Dataset Processing: Complete Guide

## Table of Contents
1. [Introduction & Background](#introduction--background)
2. [Understanding the Raw Data](#understanding-the-raw-data)
3. [Data Processing Pipeline Overview](#data-processing-pipeline-overview)
4. [Step 1: Data Augmentation Pipeline](#step-1-data-augmentation-pipeline)
5. [Step 2: Training Data Preparation](#step-2-training-data-preparation)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Why This Approach Works](#why-this-approach-works)

---

## Introduction & Background

### What are Bach Chorales?
Johann Sebastian Bach composed hundreds of four-part chorales (hymns) that are considered masterpieces of Western harmony. Each chorale has four voices:
- **Soprano** (highest voice, melody)
- **Alto** (second highest)
- **Tenor** (second lowest)
- **Bass** (lowest voice, foundation)

### The Machine Learning Goal
We want to train a neural network that can:
1. Take a soprano melody as input
2. Generate harmonious alto, tenor, and bass parts
3. Create music that sounds like Bach's style

### The Dataset
The **JSB Chorales dataset** contains 382 Bach chorales in a digital format, split into:
- **Train**: 229 chorales (for learning)
- **Test**: 77 chorales (for final evaluation)
- **Valid**: 76 chorales (for tuning during training)

---

## Understanding the Raw Data

### Data Format: Tuple Representation
Each chorale is stored as a sequence of tuples, where each tuple represents one moment in time (0.25 beats):

```python
# Example: First few moments of a chorale
[
    (69, 65, 60, 53),  # Time step 0: Soprano=69, Alto=65, Tenor=60, Bass=53
    (69, 65, 60, 53),  # Time step 1: Same notes held
    (71, 67, 62, 55),  # Time step 2: All voices move to new notes
    (72, 69, 64, 57),  # Time step 3: Harmony progression continues
    # ... continues for entire piece
]
```

### MIDI Note Numbers
Each number represents a MIDI note:
- **60** = Middle C
- **69** = A above middle C
- **53** = F below middle C
- Higher numbers = higher pitches

### Key Characteristics
- **Time Resolution**: Each tuple = 0.25 beats (16th note)
- **Always 4 voices**: Soprano, Alto, Tenor, Bass (in that order)
- **Variable Length**: Songs range from ~50 to 400+ time steps
- **Harmonic Structure**: Notes chosen to create beautiful chord progressions

---

## Data Processing Pipeline Overview

Our processing happens in two major phases:

```
Phase 1: Data Augmentation
Raw Data → Note Analysis → Augmentation → Expanded Dataset

Phase 2: Training Preparation  
Augmented Data → Voice Assignment → One-Hot Encoding → Training Batches
```

### Why Two Phases?
1. **Augmentation** creates more training examples from limited data
2. **Preparation** converts musical data into neural network-friendly format

---

## Step 1: Data Augmentation Pipeline

### Problem: Limited Training Data
- Only 229 training chorales
- Neural networks need thousands of examples
- Solution: Create variations that preserve musical structure

### Stage 1A: Convert to Structured Format

#### From Tuples to Note Objects
We convert the simple tuple format into rich **NoteInfo** objects:

```python
# Tuple format (what we start with):
(69, 65, 60, 53)

# Note Info format (what we convert to):
NoteInfo(starting_beat=0.0, pitch=69, length=1.0)  # Soprano note
NoteInfo(starting_beat=0.0, pitch=65, length=1.0)  # Alto note
# ... etc for all voices
```

#### Why This Conversion?
The tuple format only shows "what notes are playing now." The NoteInfo format captures:
- **When** each note starts
- **How long** each note lasts  
- **What pitch** it is
- **Which voice** it belongs to

This richer representation allows sophisticated musical transformations.

### Stage 1B: Musical Augmentations

#### Scale Transposition (12 variations)
We transpose each chorale to different keys:

```python
# Original in C major: (60, 64, 67, 72) - C major chord
# Transpose +2 semitones: (62, 66, 69, 74) - D major chord  
# Transpose -3 semitones: (57, 61, 64, 69) - A major chord
```

**Why this works**: 
- Harmonic relationships stay the same
- Just shifts everything up/down by same amount
- Creates 12 versions: -5 to +6 semitones

#### Rhythmic Variation (4 variations per scale)
We randomly join short notes together:

```python
# Original rhythm:
Note(beat=0, length=0.25)  # Short note
Note(beat=0.25, length=0.25)  # Another short note

# After joining:
Note(beat=0, length=0.5)  # One longer note
```

**Why this works**:
- Preserves harmony but varies rhythm
- Creates natural musical variations
- Bach often used both short and long note values

### Stage 1C: Back to Tuple Format
After augmentation, we convert back to the simple tuple format:

```
NoteInfo objects → Sample every 0.25 beats → Tuple sequences
```

### Augmentation Results
- **Input**: 229 original chorales
- **Output**: ~11,000 augmented chorales (48x increase!)
- **Saved as**: `scales_note_join_augmented.pkl`

---

## Step 2: Training Data Preparation

### Stage 2A: Voice Assignment Challenge

#### The Problem
Not all chords have exactly 4 notes. Sometimes we see:
- `(69, 65, 60)` - Only 3 notes
- `(69, 65)` - Only 2 notes  
- `(69, 65, 60, 53, 57)` - 5 notes

But our neural network expects exactly 4 voices (SATB). How do we assign notes to voices?

#### The Solution: Smart Voice Tracking
We use a **VoiceTracker** for each voice that:

1. **Prefers consistency**: If soprano was playing note 69, try to keep it on 69
2. **Respects ranges**: Soprano gets high notes, bass gets low notes
3. **Uses musical logic**: When in doubt, soprano takes highest available note

```python
# Example assignment logic:
chord = (65, 69, 72)  # 3 notes available
# Soprano (highest): gets 72
# Alto (middle): gets 69  
# Tenor (lower): gets 65
# Bass: gets silence (no suitable note)
```

### Stage 2B: One-Hot Encoding

#### Why One-Hot Encoding?
Neural networks work with numbers, not musical concepts. We convert each note to a vector:

```python
# Note 65 in Alto voice (range 51-82, total 33 positions):
# Position 0: Silence
# Position 1: Note 51  
# Position 2: Note 52
# ...
# Position 15: Note 65 ← This gets value 1.0
# Position 16: Note 66
# ...
# Position 32: Note 82
```

#### The Encoding Process
For each voice at each time step:
1. Determine which note is playing (or silence)
2. Create a vector of zeros
3. Set exactly one position to 1.0

**Result**: Each time step becomes a vector of 33 numbers (32 possible notes + silence)

### Stage 2C: Sequence Splitting

#### The Problem
- Songs have different lengths (50-400+ time steps)
- Neural networks need fixed input sizes
- Solution: Cut songs into fixed-length pieces

#### Sequence Length: 64 Time Steps
- **Musical Meaning**: 64 × 0.25 beats = 16 beats = 4 measures
- **Perfect Size**: Long enough for musical phrases, short enough for memory
- **Standard Practice**: Common length in music AI research

```python
# Long song (200 time steps):
Song: [measures 1-2-3-4-5-6-7-8-9-10-11-12-13...]

# Split into sequences:
Sequence 1: [measures 1-2-3-4]     # Time steps 0-63
Sequence 2: [measures 5-6-7-8]     # Time steps 64-127  
Sequence 3: [measures 9-10-11-12]  # Time steps 128-191
```

### Stage 2D: Dataset Creation

#### Input vs. Output Format
- **Input (X)**: Soprano melody as one-hot vectors `[batch_size, 64, 33]`
- **Output (Y)**: Alto/Tenor/Bass as class indices `[batch_size, 64]`

#### Why Different Formats?
- **Input**: One-hot gives network rich information about melody
- **Output**: Class indices work better with loss functions for classification

### Stage 2E: DataLoader Creation

#### Batching for Training
Instead of processing one sequence at a time, we group them:

```python
# Batch of 4 sequences:
Batch = [
    Sequence_1,  # 64 time steps of 4-part harmony
    Sequence_2,  # 64 time steps of 4-part harmony  
    Sequence_3,  # 64 time steps of 4-part harmony
    Sequence_4   # 64 time steps of 4-part harmony
]
```

**Final Tensor Shapes**:
- **Soprano Input**: `[4, 64, 33]` - 4 sequences, 64 time steps, 33 possible notes
- **Alto Target**: `[4, 64]` - 4 sequences, 64 time steps, 1 note index per step
- **Tenor Target**: `[4, 64]` - Same format
- **Bass Target**: `[4, 64]` - Same format

---

## Technical Implementation Details

### Voice Ranges (with Augmentation Buffer)
Since we transpose by ±5-6 semitones, we expand each voice's range:

```python
# Original ranges + augmentation buffer:
Soprano: MIDI notes 56-87 (32 notes + silence = 33 total)
Alto:    MIDI notes 51-82 (32 notes + silence = 33 total)  
Tenor:   MIDI notes 46-77 (32 notes + silence = 33 total)
Bass:    MIDI notes 31-69 (39 notes + silence = 40 total)
```

### Memory and Performance
- **Original Dataset**: ~0.4 MB
- **Augmented Dataset**: ~20 MB  
- **Training Memory**: ~500 MB for full processing
- **Processing Time**: ~5-10 minutes for full augmentation

### Error Handling
The pipeline handles:
- **Out-of-range notes**: Marked as silence
- **Missing voices**: Filled with silence  
- **Variable chord sizes**: Smart voice assignment
- **Short songs**: Padded or skipped appropriately

---

## Why This Approach Works

### Musical Validity
1. **Harmonic Relationships Preserved**: Transposition keeps chord progressions intact
2. **Voice Leading Maintained**: Note assignment respects voice movement principles  
3. **Rhythmic Variety**: Note joining creates natural rhythmic variations
4. **Style Consistency**: All variations still sound like Bach

### Machine Learning Benefits
1. **Data Abundance**: 48x more training examples
2. **Generalization**: Network learns harmony independent of key
3. **Robustness**: Handles various musical situations (missing notes, etc.)
4. **Efficiency**: Fixed-size inputs enable batch processing

### Neural Network Compatibility
1. **Consistent Shapes**: All inputs/outputs have same dimensions
2. **Numerical Stability**: One-hot encoding prevents scaling issues
3. **Classification Setup**: Target format works well with cross-entropy loss
4. **Sequence Modeling**: 64-step sequences capture musical phrases

### Data Pipeline Robustness
1. **Automated Processing**: Handles entire dataset without manual intervention
2. **Error Recovery**: Gracefully handles edge cases and malformed data
3. **Validation**: Built-in checks ensure data quality throughout pipeline
4. **Reproducibility**: Same input always produces same output

---

## Summary

This processing pipeline transforms raw Bach chorales into neural network-ready data through:

1. **Intelligent Augmentation**: Creates musical variations that preserve Bach's style
2. **Smart Voice Assignment**: Handles incomplete chords using musical logic  
3. **Proper Encoding**: Converts musical notes into numerical representations
4. **Sequence Processing**: Creates fixed-size inputs for efficient training
5. **Quality Assurance**: Validates data at every step

The result is a robust dataset of ~11,000 musical sequences, each containing 4 measures of 4-part harmony, ready to train a neural network to compose like Bach.

**Total Processing Chain**:
```
Raw Bach Chorales (382 songs)
    ↓ [Augmentation Pipeline]
Augmented Dataset (11,000+ songs)  
    ↓ [Training Preparation]
Neural Network Ready Data (50,000+ sequences)
    ↓ [DataLoader]
Training Batches for Machine Learning
```

This comprehensive approach ensures that our neural network receives high-quality, musically meaningful data that will enable it to learn Bach's harmonic language effectively.





# Bach Harmonization Neural Network Architecture

## Overview
A **feedforward neural network** that learns to generate 4-part Bach-style harmonies by taking a soprano melody as input and predicting alto, tenor, and bass voice parts.

---

## Network Architecture

### **Network Type**: Feedforward (Dense/Linear layers only)
### **Task**: Multi-output classification (3 separate voice predictions)
### **Input**: Complete soprano melody sequence
### **Output**: Complete harmony sequences for 3 voices

---

## Input/Output Specifications

### **Input Dimensions**
```python
# Soprano melody input
Input Shape: [batch_size, sequence_length, encoding_size]
           = [4, 64, 33]
           
# After flattening for network:
Flattened:  [batch_size, sequence_length × encoding_size]  
          = [4, 2112]

# Where:
# - batch_size = 4 (training batch size)
# - sequence_length = 64 (time steps)
# - encoding_size = 33 (32 MIDI notes + 1 silence)
```

### **Output Dimensions**
```python
# Three separate voice outputs
Alto Output:  [batch_size, encoding_size, sequence_length] = [4, 33, 64]
Tenor Output: [batch_size, encoding_size, sequence_length] = [4, 33, 64]  
Bass Output:  [batch_size, encoding_size, sequence_length] = [4, 40, 64]

# Note: Bass has larger encoding (40) due to wider MIDI range
```

---

## Layer-by-Layer Architecture

```
Input: Soprano Melody
       [4, 64, 33] → Flatten → [4, 2112]
              ↓
┌─────────────────────────────────────────┐
│          INPUT LAYER                    │
│  Linear(2112 → 200)                     │
│  Learnable Parameters: 422,600          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         HIDDEN LAYER 1                  │
│  Linear(200 → 200)                      │
│  ReLU Activation                        │
│  Dropout(p=0.5)                         │
│  Learnable Parameters: 40,200           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         HIDDEN LAYER 2                  │
│  Linear(200 → 200)                      │
│  ReLU Activation                        │
│  Dropout(p=0.5)                         │
│  Learnable Parameters: 40,200           │
└─────────────────────────────────────────┘
              ↓
         ┌─────────┐
         │ SPLIT   │
         └─────────┘
    ┌────────┴────────┬────────┐
    ↓                 ↓        ↓
┌─────────┐    ┌──────────┐   ┌─────────┐
│  ALTO   │    │  TENOR   │   │  BASS   │
│ BRANCH  │    │  BRANCH  │   │ BRANCH  │
└─────────┘    └──────────┘   └─────────┘
```

---

## Voice-Specific Output Branches

### **Alto Branch**
```python
# Alto harmonizer
Linear(200 → 2112)              # 200 → 64×33
ReLU Activation
Reshape(2112 → [33, 64])        # [encoding_size, sequence_length]  
Log Softmax(dim=1)              # Probability distribution over notes

# Output: [batch_size, 33, 64] log probabilities
```

### **Tenor Branch**  
```python
# Tenor harmonizer  
Linear(200 → 2112)              # 200 → 64×33
ReLU Activation
Reshape(2112 → [33, 64])        # [encoding_size, sequence_length]
Log Softmax(dim=1)              # Probability distribution over notes

# Output: [batch_size, 33, 64] log probabilities
```

### **Bass Branch**
```python
# Bass harmonizer
Linear(200 → 2560)              # 200 → 64×40 (wider range!)
ReLU Activation  
Reshape(2560 → [40, 64])        # [encoding_size, sequence_length]
Log Softmax(dim=1)              # Probability distribution over notes

# Output: [batch_size, 40, 64] log probabilities
```

---

## Parameter Count

```python
# Layer-by-layer parameter breakdown:

Input Layer:    2112 × 200 + 200 = 422,600 parameters
Hidden Layer 1: 200 × 200 + 200  = 40,200 parameters  
Hidden Layer 2: 200 × 200 + 200  = 40,200 parameters
Alto Output:    200 × 2112 + 2112 = 424,512 parameters
Tenor Output:   200 × 2112 + 2112 = 424,512 parameters  
Bass Output:    200 × 2560 + 2560 = 514,560 parameters

Total Parameters: 1,866,584 (~1.87M parameters)
```

---

## Key Design Decisions

### **1. Feedforward Architecture**
- **Why**: Simpler than RNNs, focuses on harmonic relationships
- **Trade-off**: Less temporal modeling, but faster training
- **Justification**: Bach chorales have strong harmonic patterns that feedforward can capture

### **2. Sequence-to-Sequence Processing**
- **Input**: Entire 64-note soprano sequence at once
- **Output**: Entire 64-note harmony sequences for 3 voices
- **Benefit**: Network sees full musical context, can plan ahead

### **3. Separate Output Branches**
- **Why**: Each voice has different characteristics and ranges
- **Implementation**: Dedicated linear layer for each voice
- **Advantage**: Voice-specific learning, different output dimensions

### **4. Log Softmax Output**
- **Purpose**: Converts raw logits to log probabilities
- **Compatibility**: Works with `nn.NLLLoss()` for training
- **Interpretation**: Each time step gets probability distribution over possible notes

---

## Forward Pass Flow

```python
def forward(self, soprano_input):
    # Input: [batch, 64, 33] soprano melody
    
    # 1. Flatten temporal dimension
    x = torch.flatten(soprano_input, start_dim=1)  # [batch, 2112]
    
    # 2. Process through shared layers
    x = self.input(x)                              # [batch, 200]
    
    x = self.hidden1(x)                            # [batch, 200]
    x = F.relu(x)                                  
    x = self.dropout1(x)                           
    
    x = self.hidden2(x)                            # [batch, 200]
    x = F.relu(x)
    x = self.dropout2(x)                           
    
    # 3. Generate voice-specific outputs
    
    # Alto branch
    alto_logits = self.forward_alto(x)             # [batch, 2112]
    alto_logits = F.relu(alto_logits)
    alto_reshaped = torch.reshape(alto_logits, 
                     (BATCH_SIZE, 33, 64))         # [batch, 33, 64]
    alto_output = F.log_softmax(alto_reshaped, dim=1)
    
    # Tenor branch (similar to alto)
    tenor_logits = self.forward_tenor(x)           # [batch, 2112]
    tenor_logits = F.relu(tenor_logits)
    tenor_reshaped = torch.reshape(tenor_logits,
                      (BATCH_SIZE, 33, 64))        # [batch, 33, 64]
    tenor_output = F.log_softmax(tenor_reshaped, dim=1)
    
    # Bass branch (larger output)
    bass_logits = self.forward_bass(x)             # [batch, 2560]
    bass_logits = F.relu(bass_logits)
    bass_reshaped = torch.reshape(bass_logits,
                     (BATCH_SIZE, 40, 64))         # [batch, 40, 64]
    bass_output = F.log_softmax(bass_reshaped, dim=1)
    
    return alto_output, tenor_output, bass_output
```

---

## Musical Interpretation

### **What the Network Learns:**
1. **Harmonic Progressions**: Which chords typically follow others
2. **Voice Leading**: How individual voices move smoothly between notes
3. **Bach's Style**: Specific harmonic preferences and compositional patterns
4. **Counterpoint Rules**: Avoiding parallel fifths, proper voice spacing

### **How It Harmonizes:**
1. **Sees entire melody**: Network processes full 64-note soprano sequence
2. **Extracts features**: Hidden layers learn musical patterns and relationships
3. **Generates harmonies**: Each output branch predicts one voice part
4. **Maintains context**: Each predicted note considers the entire musical phrase

### **Output Interpretation:**
```python
# For each time step and voice:
# Network outputs probability distribution over all possible notes

# Example alto output at time step 0:
alto_probs_t0 = [0.01, 0.05, 0.02, 0.85, 0.03, 0.02, 0.02, ...]
#                  ↑     ↑     ↑     ↑     ↑
#               silence note1 note2 note3 note4
#                              (most likely)

# Prediction: argmax = note3 (index 3)
```

---

## Training Characteristics

### **Loss Function**: `nn.NLLLoss()` (Negative Log Likelihood)
### **Optimizer**: SGD with momentum (lr=0.01, momentum=0.9)
### **Regularization**: Dropout (p=0.5) in hidden layers
### **Batch Size**: 4 sequences per batch
### **Sequence Length**: 64 time steps (16 beats, ~4 measures)

### **Training Strategy:**
- **Multi-task learning**: Trains 3 voice parts simultaneously
- **Teacher forcing**: Uses ground truth soprano, predicts other voices
- **Curriculum**: No curriculum, learns on full sequences from start

---

## Strengths & Limitations

### **Strengths:**
- **Simple and fast**: Feedforward architecture trains quickly  
- **Full context**: Sees entire melody when making predictions  
- **Multi-voice**: Generates complete 4-part harmonies  
- **Style learning**: Captures Bach's harmonic patterns effectively  

### **Limitations:**  
- **Fixed length**: Only works with exactly 64-note sequences  
- **No recurrence**: Limited temporal modeling compared to RNNs  
- **Rhythm constraints**: Basic rhythm representation (quantized grid)  
- **Voice independence**: Doesn't model inter-voice dependencies strongly

---

## Modern Improvements

This architecture could be enhanced with:
- **Transformer layers**: Better sequence modeling
- **Attention mechanisms**: Focus on relevant parts of melody  
- **Recurrent connections**: Better temporal dependencies
- **Variable length**: Handle sequences of different lengths
- **Multi-scale**: Multiple time resolutions for rhythm



# Bach Harmonization Baseline Models

## Overview
This document describes three baseline models used to evaluate the performance of neural network approaches to Bach chorale harmonization. These baselines provide essential comparison points to determine if the neural network is actually learning meaningful musical patterns.

---

## Purpose of Baselines

Baselines serve several critical functions in machine learning evaluation:

1. **Sanity Check**: Ensure the main model is learning something meaningful
2. **Performance Floor**: Establish minimum expected performance levels  
3. **Learning Validation**: Verify the model learns beyond simple statistical patterns
4. **Method Comparison**: Compare data-driven vs. rule-based approaches

---

## Task Definition

### **Input**: Soprano melody sequence
- Format: One-hot encoded tensor `[64, 33]`
- 64 time steps (16 beats, approximately 4 measures)
- 33 possible values (32 MIDI notes + 1 silence)

### **Output**: Three-voice harmony prediction
- Alto voice: Indices `[0-32]` (0 = silence, 1-32 = MIDI notes)
- Tenor voice: Indices `[0-32]` (0 = silence, 1-32 = MIDI notes)  
- Bass voice: Indices `[0-39]` (0 = silence, 1-39 = MIDI notes)

### **Evaluation Metric**: Note-level accuracy
- Percentage of correctly predicted notes across all voices and time steps
- Each time step contributes 3 predictions (alto, tenor, bass)

---

## Baseline 1: Random Harmonization

### **Concept**
Generates completely random note predictions within valid voice ranges. This represents the weakest possible baseline - any reasonable model should significantly outperform random guessing.

### **Implementation**
```python
class RandomBaseline:
    def __init__(self, voice_ranges):
        self.voice_ranges = voice_ranges
    
    def predict_note(self, voice_name):
        max_note = self.voice_ranges[voice_name].range_and_silence_length() - 1
        return random.randint(0, max_note)
```

### **Algorithm**
For each time step and each voice:
1. Generate random integer between 0 and maximum note index
2. Compare prediction with ground truth target
3. Count correct predictions

### **Expected Performance**
- **Theoretical**: 1/33 ≈ 3.0% for alto/tenor, 1/40 = 2.5% for bass
- **Average**: ~3.0% overall accuracy
- **Interpretation**: If neural network performs near this level, it's not learning

### **Use Case**
- **Sanity check**: Verify evaluation pipeline works correctly
- **Performance floor**: Absolute minimum any model should achieve
- **Debugging**: Identify issues with data preprocessing or evaluation

---

## Baseline 2: Most Common Note

### **Concept**  
Always predicts the most frequently occurring note for each voice in the training data. This tests whether simple frequency-based statistical learning can solve the harmonization task.

### **Implementation**
```python
class MostCommonBaseline:
    def fit(self, train_loader):
        # Count frequency of each note in training data
        note_counts = Counter()
        for batch in train_loader:
            note_counts.update(batch.flatten().tolist())
        
        # Store most common note
        self.most_common_note = note_counts.most_common(1)[0][0]
    
    def predict_note(self, voice_name):
        return self.most_common_notes[voice_name]
```

### **Algorithm**
**Training Phase:**
1. Iterate through all training sequences
2. Count frequency of each note index for each voice
3. Store the most frequent note for each voice

**Prediction Phase:**
1. For every time step, predict the most common note
2. Same prediction regardless of soprano melody or musical context

### **Expected Performance**
- **Range**: 5-15% depending on data distribution
- **Higher if**: Training data has strong note frequency bias
- **Lower if**: Bach's harmonies are relatively uniform across note ranges

### **Musical Interpretation**
This baseline tests whether Bach chorales have strong statistical biases toward certain notes. In classical harmony:
- **Alto/Tenor**: Often use middle register notes (G4, A4, B4, C5)
- **Bass**: Frequently uses chord roots and fifths
- **Limitations**: Ignores melodic context, harmonic function, voice leading

### **Use Case**
- **Statistical learning**: Tests if frequency-based prediction works
- **Data analysis**: Reveals note distribution patterns in Bach corpus
- **Neural network comparison**: Model should learn more than simple statistics

---

## Baseline 3: Rule-Based Harmonization

### **Concept**
Applies basic music theory rules to generate harmonies. Uses common chord patterns and intervallic relationships from traditional music theory to harmonize soprano melodies.

### **Implementation**
```python
class RuleBasedBaseline:
    def __init__(self, voice_ranges):
        self.chord_patterns = [
            (-5, -9, -12),   # Major chord: 4th, 5th, octave below soprano
            (-3, -7, -12),   # Minor variant: 3rd, 5th, octave below
            (-4, -8, -12),   # Alternative spacing
            (-5, -8, -15),   # Wider voicing
            (-3, -10, -12),  # Different inversion
        ]
    
    def harmonize(self, soprano_note, pattern_index):
        pattern = self.chord_patterns[pattern_index % len(self.chord_patterns)]
        alto_note = soprano_note + pattern[0]
        tenor_note = soprano_note + pattern[1] 
        bass_note = soprano_note + pattern[2]
        return alto_note, tenor_note, bass_note
```

### **Algorithm**
For each time step:
1. **Extract soprano note** from one-hot encoding
2. **Handle silence**: If soprano is silent, predict silence for all voices
3. **Select chord pattern**: Cycle through predefined patterns for variety
4. **Calculate intervals**: Add pattern intervals to soprano MIDI note
5. **Range checking**: Ensure generated notes fall within valid voice ranges
6. **Convert to indices**: Transform MIDI notes back to encoding format

### **Chord Patterns Explained**

#### **Pattern 1: (-5, -9, -12) - Major Chord**
- **Alto**: 4th below soprano (perfect 4th interval)
- **Tenor**: 5th below soprano (perfect 5th interval)  
- **Bass**: Octave below soprano
- **Example**: Soprano C5 → Alto G4, Tenor F4, Bass C4
- **Sound**: Close-position major triad

#### **Pattern 2: (-3, -7, -12) - Minor Variant**
- **Alto**: 3rd below soprano (major 3rd interval)
- **Tenor**: 5th below soprano (perfect 5th interval)
- **Bass**: Octave below soprano  
- **Example**: Soprano C5 → Alto A4, Tenor F4, Bass C4
- **Sound**: Minor triad feeling

#### **Pattern 3: (-5, -8, -15) - Wider Spacing**
- **Alto**: 4th below soprano
- **Tenor**: 6th below soprano (major 6th interval)
- **Bass**: Two octaves below soprano
- **Example**: Soprano C5 → Alto G4, Tenor E4, Bass C3
- **Sound**: More open, orchestral spacing

### **Expected Performance**
- **Range**: 15-30% accuracy
- **Strengths**: Musically coherent, follows voice leading principles
- **Limitations**: 
  - No harmonic progression logic
  - Doesn't consider musical context
  - Fixed patterns regardless of melodic direction
  - No awareness of cadences or phrase structure

### **Musical Validity**
**What it gets right:**
- **Voice spacing**: Maintains appropriate intervals between voices
- **Range adherence**: Keeps voices in singable ranges
- **Consonance**: Generates mostly consonant harmonies
- **Voice leading**: Some patterns promote smooth voice motion

**What it misses:**
- **Harmonic function**: No concept of tonic, dominant, subdominant
- **Chord progressions**: No logic for chord-to-chord movement  
- **Voice leading rules**: No prevention of parallel fifths/octaves
- **Musical phrases**: No understanding of phrase beginnings/endings
- **Stylistic nuance**: Missing Bach's specific harmonic language

### **Use Case**
- **Music theory validation**: Tests if basic theory beats data-driven approaches
- **Expert knowledge**: Represents human musical knowledge baseline
- **Interpretability**: Easy to understand why it makes specific predictions
- **Neural network target**: Model should learn these patterns plus much more

---

## Baseline Comparison Framework

### **Performance Hierarchy (Expected)**
```
Random (3%) < Most Common (8-12%) < Rule-Based (20-25%) < Neural Network (40-60%+)
```

### **What Each Baseline Tests**

| Baseline | Tests If Model Learns... | Failure Indicates... |
|----------|-------------------------|----------------------|
| Random | Anything at all | Broken evaluation/training |
| Most Common | More than frequency statistics | Only memorizing note distributions |
| Rule-Based | Musical relationships beyond basic theory | Model hasn't learned harmonic patterns |

### **Interpretation Guidelines**

#### **If Neural Network ≤ Random (3%)**
- **Problem**: Training failed completely
- **Causes**: Learning rate too high, broken loss function, data preprocessing error
- **Action**: Debug training pipeline, check data loading

#### **If Neural Network ≈ Most Common (8-12%)**  
- **Problem**: Model only learning statistical bias
- **Causes**: Insufficient model capacity, poor architecture, inadequate training
- **Action**: Increase model complexity, train longer, check feature engineering

#### **If Neural Network ≈ Rule-Based (20-25%)**
- **Problem**: Model learning basic theory but not Bach's style
- **Causes**: Limited training data, insufficient model depth, need better architecture
- **Action**: Data augmentation, deeper networks, attention mechanisms

#### **If Neural Network >> Rule-Based (40%+)**
- **Success**: Model learning complex musical patterns beyond basic theory
- **Indicates**: Capturing Bach's harmonic language, voice leading, stylistic nuances
- **Next**: Analyze what additional patterns the model discovered

---

## Implementation Notes

### **Data Format Compatibility**
- **Input**: `soprano_input[sample_idx]` shape `[64, 33]`
- **Targets**: `alto_target[sample_idx]` shape `[64]` (class indices)
- **Conversion**: `torch.argmax(soprano_input, dim=1)` converts one-hot to indices

### **Evaluation Consistency**
All baselines use identical evaluation:
```python
for time_idx in range(sequence_length):
    if prediction == target[time_idx].item():
        correct_predictions += 1
    total_predictions += 1

accuracy = correct_predictions / total_predictions
```

### **Voice Range Handling**
- **Alto/Tenor**: 33 possible values (0-32)
- **Bass**: 40 possible values (0-39) due to wider MIDI range
- **Range checking**: Rule-based baseline clips out-of-range notes to silence (index 0)

---

## Usage Example

```python
# Evaluate all baselines
results = evaluate_three_baselines(train_loader, test_loader, VOICE_RANGES, device)

# Expected output:
# Random        : 0.0334
# Most Common   : 0.1127  
# Rule-Based    : 0.2341

# Compare with your neural network
neural_accuracy = 0.4567  # Example

print(f"Improvement over best baseline: {neural_accuracy - max(results.values()):.4f}")
print(f"Relative improvement: {neural_accuracy / max(results.values()):.2f}x")
```

This baseline suite provides comprehensive evaluation framework for understanding whether and how well your neural network learns Bach's harmonization patterns.