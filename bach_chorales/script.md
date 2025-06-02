"Hello everyone,
Our next task is Task 2 from Assignment 2, which focuses on symbolic music generation with a given condition.

Specifically, we are tackling the problem of harmonizing a piece of music where the soprano voice is provided as a fixed input.

The goal is to generate the remaining voices—typically alto, tenor, and bass—in a musically coherent way, such that they complement the given soprano line while adhering to the rules of harmony and voice leading."


Inspiration and Guidance: We also gained inspiration from Kevin Summerian’s blog post on melody harmonization and AI in music generation, which can be found here.
We are grateful for the open-source efforts and academic work that made this exploration into polyphonic music modeling possible.


As you can see we have limited data so it would be nice to convert existing data into more data.

To work more effectively with the musical data, we convert each note from every voice into a NoteInfo object.
This object provides us with crucial information, such as:
When the note starts,
How long it lasts, and
Which voice it belongs to.
By organizing the data this way, we can precisely track the structure and behavior of each note across time.
Furthermore, these NoteInfo objects allow us to reconstruct the full musical sequence or transform the data into formats that are compatible with symbolic music models.


"We transpose each piece across all 12 semitones to teach the model key invariance. This helps it learn scale-independent patterns by focusing on relative pitch and harmony, not just specific keys."

We also will create variations by combining very short notes with each other, but only on integer beats like 1 and 2. 

We bring these note objects back so we can pump into the neural network

"You might notice a discrepancy in the number of time steps in our sequences.This occurs due to sequence length truncation, which we applied to remove the last two measures of each piece.
These final measures consist entirely of rests—specifically, they span 8 time steps, corresponding to two full beats in 8/4 time.
Since these sections don't contain any musical content, we trimmed them to keep our training data clean and focused on meaningful sequences."


To maintain consistency across the different vocal parts, we use a Voice Tracker.
In symbolic music, it's crucial that each voice—soprano, alto, tenor, and bass—has a complete and independent melodic line.
Rather than keeping them bundled as a single tuple, we split the tuple into four separate lines, giving us a clear and continuous sequence for each voice.
This ensures that no voice is dropped or misaligned during generation or evaluation, keeping the musical structure intact.

"We use one-hot encoding for the soprano input because neural networks require numerical inputs—but raw note indices (like 0 to 32) imply false ordinal relationships.
For example, MIDI note 32 isn't 'twice' MIDI note 16.
One-hot encoding treats each note as a distinct category, avoiding this issue and allowing the network to focus on musical relationships rather than misleading arithmetic patterns.
This helps the model learn true harmonic structures, like those in Bach's style."

We want to split the sequence to pump into the network as our network is not flexible input sizes. Not autoregressive.

### Network Architecture

Grab the soprano melody --> flatten and put into the network and output th

"We made four key design choices:

1. Feedforward architecture – It's simple and fast, and still captures Bach’s harmonic patterns.

2. Sequence-to-sequence – The model sees the full 64-note soprano line and predicts all three harmony lines at once, enabling global coherence.

3. Separate output branches – Each voice has its own linear layer to account for different ranges and characteristics.

4. Log softmax output – Gives log probabilities for use with NLLLoss, providing a proper note distribution at each step."





"The network learns several musical concepts:
It picks up on harmonic progressions, smooth voice leading, Bach’s stylistic patterns, and basic counterpoint rules like avoiding parallel fifths.

To harmonize, the model sees the entire soprano melody, learns musical features through hidden layers, and outputs one voice per branch—alto, tenor, and bass—while maintaining full context.

At each time step, the model outputs a probability distribution over possible notes.
For example, the alto part at time step 0 might assign highest probability to a specific note—say, index 3—meaning that note is most likely chosen as the harmony at that point."



"Our first baseline is random harmonization—the model guesses notes randomly within each voice's range.
It’s a sanity check and sets the performance floor:
Around 3% accuracy for alto/tenor, 2.5% for bass.
If our model performs near this, it's not learning.
This baseline helps us verify that the evaluation pipeline works and gives us something to beat."



"The second baseline always predicts the most frequent note for each voice from the training data—ignoring the soprano and context entirely.
Performance ranges from 5% to 15%, depending on how biased the dataset is.
This tests whether simple frequency-based learning is enough.
For example, alto and tenor often sit in the middle register, and bass sticks to chord roots and fifths.
But this approach ignores melody, harmony, and voice leading, so a real model should do much better."


"Our third baseline uses basic music theory to harmonize.
It applies fixed chord patterns to the soprano line—like major or minor triads—adjusted to fit each voice’s range.
For example, it might add a 4th, 5th, and octave below the soprano to get alto, tenor, and bass.
It produces musically valid chords, respects voice ranges, and avoids random dissonance.
But it doesn’t understand harmonic progressions, cadences, or context—so it can’t replicate Bach’s style.
This baseline reflects human theory rules, and a good model should go far beyond that."


Our Bach harmonization neural network achieved a test accuracy of 22.3%, demonstrating significant learning beyond baseline approaches. The model outperformed random prediction by a factor of 8.3 (2.69% vs 22.3%), exceeded frequency-based statistical learning by 97% (11.34% vs 22.3%), and surpassed rule-based music theory approaches by 70% (13.08% vs 22.3%). Voice-specific performance showed expected musical patterns, with alto achieving the highest accuracy at 26.6%, tenor at 21.5%, and bass at 18.8% - reflecting the increasing complexity of harmonic roles from accompaniment to foundational voices. These results validate that our neural architecture successfully learned Bach's compositional style, moving beyond simple note frequency memorization to acquire contextual harmonization patterns specific to Baroque musical language. While the absolute accuracy indicates room for improvement toward expert human performance levels, the substantial improvements over all baselines confirm that our model has internalized meaningful musical relationships and represents a solid foundation for automated Bach-style composition.


We use accuracy as our evaluation metric because it provides a fair comparison across all model types - our neural network, random baseline, statistical baseline, and rule-based baseline all produce the same output format (class indices from 0-32/39), making accuracy a natural common ground for evaluation. Unlike perplexity, which requires probability distributions that our baselines don't provide, accuracy directly measures the fundamental task performance: "What percentage of notes are correctly predicted?" This makes it both technically feasible across all approaches and easily interpretable - a 22.3% accuracy means the model correctly harmonizes roughly 1 in 4 notes, providing clear insight into relative performance improvements between methods.



"We trained the model using NLLLoss with SGD + momentum, and added dropout for regularization.
Each batch had 4 sequences of 64 time steps—about 4 measures.
The network uses multi-task learning to predict all 3 harmony voices at once, with teacher forcing on the soprano.
Strengths: It’s simple, fast, sees the full melody at once, and learns Bach’s harmonic style well.
Limitations:
It only handles fixed-length input, lacks recurrence, has limited rhythm modeling, and doesn’t model voice dependencies deeply.
To improve it, we could add transformers, attention, recurrence, or allow variable-length sequences for more flexibility."




# 🎼 Task 2: Symbolic Music Generation – Bach Harmonization

## 🔊 Slide 1: Introduction
Hello everyone.  
Our next task is **Task 2 from Assignment 2**, focused on **symbolic music generation**.

The challenge:  
Given a **fixed soprano melody**, generate the **alto, tenor, and bass** lines in a musically coherent way, respecting harmony and voice leading.

---

## 🎼 Slide 2: Inspiration & Data Challenges
We drew inspiration from **Kevin Sumerian’s blog** on melody harmonization with AI.  
This project builds on open-source efforts and academic work in polyphonic music modeling.

### Key Issue:  
**Limited data**, which we addressed using:
- Data augmentation
- Richer note representation

---

## 🎵 Slide 3: Note Representation & Preprocessing
We represent each note as a `NoteInfo` object:
- Start time
- Duration
- Voice

We also use a **Voice Tracker** to split multivoice tuples into four separate, continuous voice lines—S, A, T, B—ensuring structural consistency.

---

## 🔁 Slide 4: Data Augmentation
To increase data:
- **Transpose** each piece into all 12 keys
- **Merge short notes** on integer beats
- **Trim trailing rests** to remove uninformative steps

---

## 🎹 Slide 5: Input Encoding
We use **one-hot encoding** for the soprano:
- Avoids false ordinal relationships (e.g., MIDI 32 ≠ 2× MIDI 16)
- Helps model learn **true harmonic patterns**

---

## 🧠 Slide 6: Model Architecture
**Feedforward sequence-to-sequence model**:

**Key design choices**:
1. Full 64-note soprano input → all 3 harmonies at once
2. **Separate output heads** for A, T, B
3. **Log-softmax outputs** + NLLLoss
4. Simple, fast, interpretable

---

## 🎯 Slide 7: Learned Musical Concepts
The model learns:
- Harmonic progressions
- Smooth voice leading
- Counterpoint rules (e.g., avoid parallel fifths)

It predicts a **note distribution** per voice at each time step.

---

## 📉 Slide 8: Baselines
We compare to 3 baselines:

1. **Random guessing** (~2.5–3%)
2. **Most frequent note per voice** (5–15%)
3. **Rule-based triads** (~13%)  
   → Valid chords but no progression or style

Our goal: Beat all of them.

---

## 📈 Slide 9: Final Results
**Test accuracy: 22.3%**

Voice-wise performance:
- Alto: **26.6%**
- Tenor: **21.5%**
- Bass: **18.8%**

Compared to baselines:
- **8.3×** better than random
- **+97%** over frequency-based
- **+70%** over rule-based

---

## ✅ Slide 10: Evaluation & Training
We use **accuracy** since all models output note indices.  
Perplexity doesn’t apply to rule-based approaches.

Training:
- Loss: **NLLLoss**
- Optimizer: **SGD + momentum**
- Regularization: **Dropout**
- Input: 4× 64-timestep sequences per batch

---

## 🚀 Slide 11: Future Work
Potential improvements:
- Add **transformers or attention**
- Support **variable-length input**
- Model **inter-voice dependencies**
- Improve rhythm modeling

---

## 🎵 Slide 12: Conclusion
Our model captures **Bach-like harmonization** using symbolic inputs and outperforms all baseline strategies.

It demonstrates the power of simple neural architectures in learning musical structure—and provides a strong foundation for future work in AI music generation.















