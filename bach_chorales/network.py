import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random
from utils2 import VoiceRange

SEQUENCE_LENGTH = 64  
BATCH_SIZE = 4
SILENCE_INDEX = 0

VOICE_RANGES = {
    'soprano': VoiceRange(61 - 5, 81 + 6),  # Range: 56-87
    'alto': VoiceRange(56 - 5, 76 + 6),     # Range: 51-82  
    'tenor': VoiceRange(51 - 5, 71 + 6),    # Range: 46-77
    'bass': VoiceRange(36 - 5, 63 + 6)      # Range: 31-69
}

class MultiVoiceHarmonizationNetwork(nn.Module):
    
    
    def __init__(self, input_dimensions, hidden_layer_size=256, dropout_rate=0.4):
        super(MultiVoiceHarmonizationNetwork, self).__init__()
        self.input = nn.Linear(SEQUENCE_LENGTH *
                               VOICE_RANGES['soprano'].range_and_silence_length(), 200)
        self.hidden1 = nn.Linear(200, 200)
        self.dropout1 = nn.Dropout(0.5)
        self.hidden2 = nn.Linear(200, 200)
        self.dropout2 = nn.Dropout(0.5)

        self.forward_alto = nn.Linear(
            200, SEQUENCE_LENGTH * VOICE_RANGES['alto'].range_and_silence_length())
        self.forward_tenor = nn.Linear(
            200, SEQUENCE_LENGTH * VOICE_RANGES['tenor'].range_and_silence_length())
        self.forward_bass = nn.Linear(
            200, SEQUENCE_LENGTH * VOICE_RANGES['bass'].range_and_silence_length())

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.input(x)

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # alto
        x_alto = self.forward_alto(x)
        x_alto = F.relu(x_alto)
        x_alto = torch.reshape(
            x_alto, (BATCH_SIZE, VOICE_RANGES['alto'].range_and_silence_length(), SEQUENCE_LENGTH))
        
        y_alto = F.log_softmax(x_alto, dim=1)

        # tenor
        x_tenor = self.forward_tenor(x)
        x_tenor = F.relu(x_tenor)
        x_tenor = torch.reshape(
            x_tenor, (BATCH_SIZE, VOICE_RANGES['tenor'].range_and_silence_length(), SEQUENCE_LENGTH))

        y_tenor = F.log_softmax(x_tenor, dim=1)

        #bass
        x_bass = self.forward_bass(x)
        x_bass = F.relu(x_bass)
        x_bass = torch.reshape(
            x_bass, (BATCH_SIZE, VOICE_RANGES['bass'].range_and_silence_length(), SEQUENCE_LENGTH))
        y_bass = F.log_softmax(x_bass, dim=1)

        return y_alto, y_tenor, y_bass
        
    #     # Store network configuration
    #     self.input_dims = input_dimensions
    #     self.hidden_size = hidden_layer_size
    #     self.dropout_prob = dropout_rate
        
    #     # Input processing layer
    #     self.melody_processor = nn.Linear(
    #         SEQUENCE_LENGTH * VOICE_RANGES['soprano'].range_and_silence_length(), 
    #         hidden_layer_size
    #     )
        
    #     # Deep feature extraction layers
    #     self.feature_extractor_1 = nn.Linear(hidden_layer_size, hidden_layer_size)
    #     self.regularization_1 = nn.Dropout(dropout_rate)
        
    #     self.feature_extractor_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
    #     self.regularization_2 = nn.Dropout(dropout_rate)
        
    #     # Voice-specific output layers
    #     self.alto_harmonizer = nn.Linear(
    #         hidden_layer_size, 
    #         SEQUENCE_LENGTH * VOICE_RANGES['alto'].range_and_silence_length()
    #     )
        
    #     self.tenor_harmonizer = nn.Linear(
    #         hidden_layer_size, 
    #         SEQUENCE_LENGTH * VOICE_RANGES['tenor'].range_and_silence_length()
    #     )
        
    #     self.bass_harmonizer = nn.Linear(
    #         hidden_layer_size, 
    #         SEQUENCE_LENGTH * VOICE_RANGES['bass'].range_and_silence_length()
    #     )
        
    #     print(f"🎼 Harmonization network initialized:")
    #     print(f"  Input dimensions: {self.input_dims}")
    #     print(f"  Hidden layer size: {hidden_layer_size}")
    #     print(f"  Dropout rate: {dropout_rate}")
    
    # def forward(self, soprano_input):
    #     """
    #     Forward pass: soprano melody → harmony
        
    #     Args:
    #         soprano_input: Tensor [batch_size, sequence_length, encoding_dims]
        
    #     Returns:
    #         alto_output, tenor_output, bass_output: Harmony predictions
    #     """
    #     # Flatten temporal sequence for processing
    #     flattened_melody = torch.flatten(soprano_input, start_dim=1)
        
    #     # Process melody through input layer
    #     processed_features = self.melody_processor(flattened_melody)
        
    #     # Deep feature extraction with regularization
    #     harmonic_features = self.feature_extractor_1(processed_features)
    #     harmonic_features = F.relu(harmonic_features)
    #     harmonic_features = self.regularization_1(harmonic_features)
        
    #     harmonic_features = self.feature_extractor_2(harmonic_features)
    #     harmonic_features = F.relu(harmonic_features)
    #     harmonic_features = self.regularization_2(harmonic_features)
        
    #     # Generate voice-specific harmonies
    #     alto_logits = self.alto_harmonizer(harmonic_features)
    #     alto_logits = F.relu(alto_logits)
    #     alto_output = torch.reshape(
    #         alto_logits, 
    #         (BATCH_SIZE, VOICE_RANGES['alto'].range_and_silence_length(), SEQUENCE_LENGTH)
    #     )
    #     alto_probabilities = F.log_softmax(alto_output, dim=1)
        
    #     tenor_logits = self.tenor_harmonizer(harmonic_features)
    #     tenor_logits = F.relu(tenor_logits)
    #     tenor_output = torch.reshape(
    #         tenor_logits, 
    #         (BATCH_SIZE, VOICE_RANGES['tenor'].range_and_silence_length(), SEQUENCE_LENGTH)
    #     )
    #     tenor_probabilities = F.log_softmax(tenor_output, dim=1)
        
    #     bass_logits = self.bass_harmonizer(harmonic_features)
    #     bass_logits = F.relu(bass_logits)
    #     bass_output = torch.reshape(
    #         bass_logits, 
    #         (BATCH_SIZE, VOICE_RANGES['bass'].range_and_silence_length(), SEQUENCE_LENGTH)
    #     )
    #     bass_probabilities = F.log_softmax(bass_output, dim=1)
        
    #     return alto_probabilities, tenor_probabilities, bass_probabilities



class HarmonizationMetrics:
    """Advanced metrics for evaluating harmonization quality"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all accumulated metrics"""
        self.total_predictions = 0
        self.correct_predictions = 0
        self.voice_accuracies = {'alto': 0, 'tenor': 0, 'bass': 0}
        self.voice_counts = {'alto': 0, 'tenor': 0, 'bass': 0}
    
    def calculate_note_accuracy(self, predictions, targets):
        """Calculate per-note prediction accuracy"""
        correct_notes = 0
        total_notes = 0
        
        for batch_idx in range(BATCH_SIZE):
            prediction_transposed = predictions[batch_idx].transpose(0, 1)
            
            for time_idx in range(SEQUENCE_LENGTH):
                predicted_note = torch.argmax(prediction_transposed[time_idx])
                target_note = targets[batch_idx][time_idx]
                
                if predicted_note == target_note:
                    correct_notes += 1
                total_notes += 1
        
        return correct_notes, total_notes
    
    def calculate_musical_accuracy(self, predictions, targets):
        """Calculate musically-informed accuracy metrics"""
        musical_correct = 0
        total_measures = 0
        
        for batch_idx in range(BATCH_SIZE):
            prediction_transposed = predictions[batch_idx].transpose(0, 1)
            
            # Analyze by musical measures (16 time steps each)
            measures_per_sequence = SEQUENCE_LENGTH // 16
            
            for measure_idx in range(measures_per_sequence):
                measure_start = measure_idx * 16
                measure_end = (measure_idx + 1) * 16
                measure_targets = targets[batch_idx][measure_start:measure_end]
                
                for time_idx in range(measure_start, measure_end):
                    predicted_note = torch.argmax(prediction_transposed[time_idx])
                    
                    # Check if prediction is musically reasonable (within measure context)
                    if predicted_note.item() in measure_targets:
                        musical_correct += 1
                
                total_measures += 16  # 16 time steps per measure
        
        return musical_correct, total_measures
    
    def update_batch_metrics(self, alto_pred, tenor_pred, bass_pred, 
                            alto_target, tenor_target, bass_target):
        """Update metrics with batch results"""
        
        # Standard accuracy
        alto_correct, alto_total = self.calculate_note_accuracy(alto_pred, alto_target)
        tenor_correct, tenor_total = self.calculate_note_accuracy(tenor_pred, tenor_target)
        bass_correct, bass_total = self.calculate_note_accuracy(bass_pred, bass_target)
        
        self.correct_predictions += (alto_correct + tenor_correct + bass_correct)
        self.total_predictions += (alto_total + tenor_total + bass_total)
        
        # Voice-specific accuracies
        self.voice_accuracies['alto'] += alto_correct
        self.voice_accuracies['tenor'] += tenor_correct
        self.voice_accuracies['bass'] += bass_correct
        
        self.voice_counts['alto'] += alto_total
        self.voice_counts['tenor'] += tenor_total
        self.voice_counts['bass'] += bass_total
        
        # Musical accuracy
        alto_musical, alto_musical_total = self.calculate_musical_accuracy(alto_pred, alto_target)
        return alto_musical, alto_musical_total
    
    def get_accuracy_summary(self):
        """Get comprehensive accuracy summary"""
        if self.total_predictions == 0:
            return 0.0, {}
        
        overall_accuracy = self.correct_predictions / self.total_predictions
        
        voice_accuracy_summary = {}
        for voice in ['alto', 'tenor', 'bass']:
            if self.voice_counts[voice] > 0:
                voice_accuracy_summary[voice] = self.voice_accuracies[voice] / self.voice_counts[voice]
            else:
                voice_accuracy_summary[voice] = 0.0
        
        return overall_accuracy, voice_accuracy_summary



class HarmonizationTrainer:
    """
    Comprehensive training system for Bach chorale harmonization
    Includes advanced optimization, scheduling, and monitoring
    """
    
    def __init__(self, model, train_loader, validation_loader, test_loader, 
                 learning_rate=0.001, device='cuda'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        
        # Advanced optimization setup
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training monitoring
        self.training_metrics = HarmonizationMetrics()
        self.validation_metrics = HarmonizationMetrics()
        
        # Tensorboard logging (optional)
        self.use_tensorboard = True
        if self.use_tensorboard:
            self.writer = SummaryWriter('runs/bach_harmonization')
            
        # Training history
        self.training_history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'learning_rates': []
        }
        
        print(f"🚀 Harmonization trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Optimization: Adam with weight decay")
        print(f"  Scheduler: ReduceLROnPlateau")
    
    def calculate_voice_loss(self, predictions, targets):
        """Calculate loss for one voice part"""
        return self.loss_function(predictions, targets)
    
    def train_single_epoch(self, epoch_number):
        """Execute one complete training epoch"""
        
        self.model.train()
        self.training_metrics.reset_metrics()
        
        epoch_loss = 0.0
        batch_count = 0
        
        print(f"\n🎵 Training Epoch {epoch_number}")
        
        for batch_idx, (soprano_input, alto_target, tenor_target, bass_target) in enumerate(self.train_loader):
            
            # Move data to device
            soprano_input = soprano_input.to(self.device)
            alto_target = alto_target.to(self.device)
            tenor_target = tenor_target.to(self.device)
            bass_target = bass_target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            alto_pred, tenor_pred, bass_pred = self.model(soprano_input)
            
            # Calculate losses
            alto_loss = self.calculate_voice_loss(alto_pred, alto_target)
            tenor_loss = self.calculate_voice_loss(tenor_pred, tenor_target)
            bass_loss = self.calculate_voice_loss(bass_pred, bass_target)
            
            combined_loss = alto_loss + tenor_loss + bass_loss
            
            # Backward pass and optimization
            combined_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += combined_loss.item()
            batch_count += 1
            
            # Update training metrics periodically
            if batch_idx % 10 == 0:  # More frequent logging for smaller datasets
                self.training_metrics.update_batch_metrics(
                    alto_pred, tenor_pred, bass_pred,
                    alto_target, tenor_target, bass_target
                )
                
                # Progress reporting
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx:3d}/{len(self.train_loader):3d} | "
                      f"Loss: {combined_loss.item():.4f} | LR: {current_lr:.6f}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / batch_count
        overall_accuracy, voice_accuracies = self.training_metrics.get_accuracy_summary()
        
        # Store training history
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['train_accuracy'].append(overall_accuracy)
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        print(f"  ✅ Epoch {epoch_number} Training Complete:")
        print(f"     Average Loss: {avg_loss:.4f}")
        print(f"     Overall Accuracy: {overall_accuracy:.3f}")
        print(f"     Voice Accuracies: Alto={voice_accuracies['alto']:.3f}, "
              f"Tenor={voice_accuracies['tenor']:.3f}, Bass={voice_accuracies['bass']:.3f}")
        
        return avg_loss, overall_accuracy
    
    def validate_epoch(self, epoch_number):
        """Execute validation for current epoch"""
        
        self.model.eval()
        self.validation_metrics.reset_metrics()
        
        validation_loss = 0.0
        batch_count = 0
        
        print(f"\n🔍 Validating Epoch {epoch_number}")
        
        with torch.no_grad():
            for batch_idx, (soprano_input, alto_target, tenor_target, bass_target) in enumerate(self.validation_loader):
                
                # Move data to device
                soprano_input = soprano_input.to(self.device)
                alto_target = alto_target.to(self.device)
                tenor_target = tenor_target.to(self.device)
                bass_target = bass_target.to(self.device)
                
                # Forward pass
                alto_pred, tenor_pred, bass_pred = self.model(soprano_input)
                
                # Calculate losses
                alto_loss = self.calculate_voice_loss(alto_pred, alto_target)
                tenor_loss = self.calculate_voice_loss(tenor_pred, tenor_target)
                bass_loss = self.calculate_voice_loss(bass_pred, bass_target)
                
                combined_loss = alto_loss + tenor_loss + bass_loss
                validation_loss += combined_loss.item()
                batch_count += 1
                
                # Update validation metrics
                self.validation_metrics.update_batch_metrics(
                    alto_pred, tenor_pred, bass_pred,
                    alto_target, tenor_target, bass_target
                )
        
        # Calculate validation metrics
        avg_val_loss = validation_loss / batch_count
        val_accuracy, val_voice_accuracies = self.validation_metrics.get_accuracy_summary()
        
        # Store validation history
        self.training_history['val_loss'].append(avg_val_loss)
        self.training_history['val_accuracy'].append(val_accuracy)
        
        # Learning rate scheduling
        self.scheduler.step(avg_val_loss)
        
        print(f"  ✅ Epoch {epoch_number} Validation Complete:")
        print(f"     Validation Loss: {avg_val_loss:.4f}")
        print(f"     Validation Accuracy: {val_accuracy:.3f}")
        print(f"     Voice Accuracies: Alto={val_voice_accuracies['alto']:.3f}, "
              f"Tenor={val_voice_accuracies['tenor']:.3f}, Bass={val_voice_accuracies['bass']:.3f}")
        
        # Tensorboard logging
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch_number)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch_number)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch_number)
        
        return avg_val_loss, val_accuracy
    
    def execute_training(self, num_epochs=50):
        """Execute complete training process"""
        
        print(f"\n🎼 Starting Bach Harmonization Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.validation_loader)}")
        print(f"="*60)
        
        best_validation_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_accuracy = self.train_single_epoch(epoch)
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch(epoch)
            
            # Save best model
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                self.save_model(f'best_harmonization_model.pt')
                print(f"  💾 New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Tensorboard logging
            if self.use_tensorboard:
                self.writer.add_scalar('Loss/Training', train_loss, epoch)
                self.writer.add_scalar('Accuracy/Training', train_accuracy, epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f"  ⏱️  Epoch {epoch} completed in {epoch_time:.1f}s")
            print("-" * 50)
        
        total_training_time = time.time() - training_start_time
        print(f"\n🎉 Training completed in {total_training_time/60:.1f} minutes!")
        print(f"   Best validation loss: {best_validation_loss:.4f}")
        
        # Final evaluation
        self.final_evaluation()
        
        if self.use_tensorboard:
            self.writer.close()
    
    def final_evaluation(self):
        """Comprehensive final evaluation on test set"""
        
        print(f"\n🎯 Final Model Evaluation")
        
        self.model.eval()
        test_metrics = HarmonizationMetrics()
        test_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for soprano_input, alto_target, tenor_target, bass_target in self.test_loader:
                
                soprano_input = soprano_input.to(self.device)
                alto_target = alto_target.to(self.device)
                tenor_target = tenor_target.to(self.device)
                bass_target = bass_target.to(self.device)
                
                alto_pred, tenor_pred, bass_pred = self.model(soprano_input)
                
                alto_loss = self.calculate_voice_loss(alto_pred, alto_target)
                tenor_loss = self.calculate_voice_loss(tenor_pred, tenor_target)
                bass_loss = self.calculate_voice_loss(bass_pred, bass_target)
                
                combined_loss = alto_loss + tenor_loss + bass_loss
                test_loss += combined_loss.item()
                batch_count += 1
                
                test_metrics.update_batch_metrics(
                    alto_pred, tenor_pred, bass_pred,
                    alto_target, tenor_target, bass_target
                )
        
        avg_test_loss = test_loss / batch_count
        test_accuracy, test_voice_accuracies = test_metrics.get_accuracy_summary()
        
        print(f"📊 Final Test Results:")
        print(f"   Test Loss: {avg_test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.3f}")
        print(f"   Voice Accuracies:")
        print(f"     Alto: {test_voice_accuracies['alto']:.3f}")
        print(f"     Tenor: {test_voice_accuracies['tenor']:.3f}")
        print(f"     Bass: {test_voice_accuracies['bass']:.3f}")
        
        return avg_test_loss, test_accuracy
    
    def save_model(self, filename):
        """Save trained model"""
        Path("./models").mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, f"./models/{filename}")
    
    def plot_training_progress(self):
        """Visualize training progress"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training vs Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate schedule
        axes[1, 0].plot(self.training_history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Combined loss and accuracy
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        ax1.plot(self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax2.plot(self.training_history['val_accuracy'], 'b-', label='Validation Accuracy')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='r')
        ax2.set_ylabel('Accuracy', color='b')
        ax1.set_title('Validation Metrics')
        
        plt.tight_layout()
        plt.show()

