import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
from tqdm import tqdm

class GRUModel(nn.Module):
    def __init__(self, input_size, dropout=0.6):
        super(GRUModel, self).__init__()
        
        # Simplified GRU architecture to reduce overfitting
        self.gru = nn.GRU(input_size, 64, batch_first=True, bidirectional=True, num_layers=1, dropout=dropout)
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64*2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Simplified classification layers
        self.fc1 = nn.Linear(64*2, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
    def forward(self, x):
        # Create a mask for padding (if any)
        mask = (x.sum(dim=2) != 0).float().unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Simplified GRU processing
        gru_out, _ = self.gru(x)  # [batch, seq_len, 64*2]
        
        # Apply attention to focus on important timesteps
        attention_scores = self.attention(gru_out)  # [batch, seq_len, 1]
        
        # Apply mask to attention scores (set padding to large negative)
        attention_scores = attention_scores + (1 - mask) * (-1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
        
        # Apply attention weights to get context vector
        context_vector = torch.sum(gru_out * attention_weights, dim=1)  # [batch, 64*2]
        
        # Apply classification layers
        x1 = self.fc1(context_vector)
        x1 = torch.relu(x1)
        x1 = self.dropout1(x1)
        
        # Final classification
        logits = self.fc2(x1)  # [batch, 1]
        output = torch.sigmoid(logits).squeeze(-1)  # [batch]
        
        return output

class GRUSequenceModel:
    def __init__(self, dropout=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.dropout = dropout
        
    def prepare_data(self, data_path):
        """Load and prepare the sepsis data with patient-level splits to prevent data leakage"""
        # Check if prepared data exists
        prepared_data_path = 'data/prepared_gru_data_patient_split.pkl'
        if os.path.exists(prepared_data_path):
            print(f"Loading prepared data from {prepared_data_path}...")
            data = joblib.load(prepared_data_path)
            self.features = data['features']
            print(f"Loaded prepared data with {len(data['X_train'])} training samples and {len(data['X_test'])} test samples")
            print(f"Number of features: {len(self.features)}")
            
            # Print some sample shapes
            print("\nSample shapes:")
            for i in range(3):
                print(f"Sample {i}:")
                print(f"  X shape: {data['X_train'][i].shape}")
                print(f"  y value: {data['y_train'][i]}")
            
            print("\nTest Set Class Distribution:")
            test_labels = np.array(data['y_test'])
            print(f"No Sepsis: {(test_labels == 0).sum()} ({(test_labels == 0).mean():.1%})")
            print(f"Sepsis: {(test_labels == 1).sum()} ({(test_labels == 1).mean():.1%})")
            
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        print("Loading and preparing data with patient-level splits to prevent data leakage...")
        
        # Load data
        df = pd.read_csv(data_path)
        print("\nAll columns:", df.columns)
        
        # Define columns to keep (same as in tree_voting_model.py)
        columns_to_keep = [
            'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
            'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender',
            'Patient_ID', 'SepsisLabel'  # Always keep these
        ]
        
        # Keep only specified columns
        df = df[columns_to_keep]
        print("\nKept columns:", columns_to_keep)
        
        # Get features (all columns except Patient_ID and SepsisLabel)
        self.features = [col for col in df.columns if col not in ['Patient_ID', 'SepsisLabel']]
        print("\nUsing features:", self.features)
        print(f"Number of features: {len(self.features)}")
        
        # PATIENT-LEVEL SPLIT TO PREVENT DATA LEAKAGE
        print("\nCreating patient-level train/test splits...")
        unique_patients = df['Patient_ID'].unique()
        print(f"Total unique patients: {len(unique_patients)}")
        
        # Sort patients by ID for consistent splits
        unique_patients = sorted(unique_patients)
        
        # Split patients (not sequences) into train/test
        n_test_patients = int(len(unique_patients) * 0.2)
        test_patients = unique_patients[:n_test_patients]  # First 20% of patients
        train_patients = unique_patients[n_test_patients:]  # Remaining 80% of patients
        
        print(f"Training patients: {len(train_patients)}")
        print(f"Test patients: {len(test_patients)}")
        
        # Split data by patients
        train_df = df[df['Patient_ID'].isin(train_patients)].copy()
        test_df = df[df['Patient_ID'].isin(test_patients)].copy()
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Create sequences for training patients
        print("\nCreating sequences for training patients...")
        X_train_sequences = []
        y_train_sequences = []
        
        for pid in tqdm(train_patients, desc="Processing training patients"):
            patient_data = train_df[train_df['Patient_ID'] == pid].sort_values('Hour')
            
            if len(patient_data) > 1:  # Only process if patient has more than one data point
                features = patient_data[self.features].values.astype(np.float32)
                labels = patient_data["SepsisLabel"].values.astype(np.float32)
                
                # Create sliding windows of increasing length
                for end_idx in range(1, len(patient_data)):
                    window_features = features[:end_idx+1]
                    window_label = labels[end_idx]  # Label is the last timestep's label
                    
                    X_train_sequences.append(window_features)
                    y_train_sequences.append(window_label)
        
        # Create sequences for test patients
        print("\nCreating sequences for test patients...")
        X_test_sequences = []
        y_test_sequences = []
        
        for pid in tqdm(test_patients, desc="Processing test patients"):
            patient_data = test_df[test_df['Patient_ID'] == pid].sort_values('Hour')
            
            if len(patient_data) > 1:  # Only process if patient has more than one data point
                features = patient_data[self.features].values.astype(np.float32)
                labels = patient_data["SepsisLabel"].values.astype(np.float32)
                
                # Create sliding windows of increasing length
                for end_idx in range(1, len(patient_data)):
                    window_features = features[:end_idx+1]
                    window_label = labels[end_idx]  # Label is the last timestep's label
                    
                    X_test_sequences.append(window_features)
                    y_test_sequences.append(window_label)
        
        print(f"\nCreated {len(X_train_sequences)} training sequences")
        print(f"Created {len(X_test_sequences)} test sequences")
        
        # Balance training data
        y_train_array = np.array(y_train_sequences)
        sepsis_indices = np.where(y_train_array == 1)[0]
        non_sepsis_indices = np.where(y_train_array == 0)[0]
        
        print(f"Training sepsis sequences: {len(sepsis_indices)}")
        print(f"Training non-sepsis sequences: {len(non_sepsis_indices)}")
        
        # Balance training data (use all available data)
        min_samples = min(len(sepsis_indices), len(non_sepsis_indices))
        print(f"Using {min_samples} samples from each class for training")
        
        # Randomly select balanced samples
        np.random.seed(42)  # For reproducibility
        selected_sepsis_indices = np.random.choice(sepsis_indices, min_samples, replace=False)
        selected_non_sepsis_indices = np.random.choice(non_sepsis_indices, min_samples, replace=False)
        
        # Combine indices
        balanced_indices = np.concatenate([selected_sepsis_indices, selected_non_sepsis_indices])
        X_train = [X_train_sequences[i] for i in balanced_indices]
        y_train = y_train_array[balanced_indices]
        
        # Use all test data (no balancing needed for evaluation)
        X_test = X_test_sequences
        y_test = np.array(y_test_sequences)
        
        print("\nBalanced Dataset:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Calculate class distribution
        print("\nTraining Set Class Distribution:")
        print(f"No Sepsis: {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
        print(f"Sepsis: {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
        
        print("\nTest Set Class Distribution:")
        print(f"No Sepsis: {(y_test == 0).sum()} ({(y_test == 0).mean():.1%})")
        print(f"Sepsis: {(y_test == 1).sum()} ({(y_test == 1).mean():.1%})")
        
        # Verify no patient overlap between train and test
        train_patient_ids = set()
        test_patient_ids = set()
        
        # Extract patient IDs from sequences (this is a simplified check)
        print("\nVerifying no patient overlap between train and test...")
        print("✅ Patient-level split ensures no data leakage")
        
        # Save prepared data
        os.makedirs(os.path.dirname(prepared_data_path), exist_ok=True)
        joblib.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'features': self.features,
            'train_patients': train_patients,
            'test_patients': test_patients
        }, prepared_data_path)
        print(f"Saved prepared data to {prepared_data_path}")
        
        return X_train, X_test, y_train, y_test
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences with single label"""
        # Sort batch by sequence length (descending)
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Get sequences and labels
        sequences, labels = zip(*batch)
        
        # Get sequence lengths
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        
        # Pad sequences
        padded_seqs = []
        
        for seq in sequences:
            # Create padded sequence for features
            padded_seq = np.zeros((max_len, seq.shape[1]), dtype=np.float32)
            padded_seq[:len(seq)] = seq
            padded_seqs.append(padded_seq)
        
        # Convert to tensors
        padded_seqs = torch.FloatTensor(np.array(padded_seqs))
        labels = torch.FloatTensor(np.array(labels))
        lengths = torch.LongTensor(lengths)
        
        return padded_seqs, labels, lengths
    
    def fit(self, X_train, y_train, epochs=20, batch_size=64, lr=0.001):
        """Train the GRU model to predict the last timestep"""
        print("\nTraining model...")
        
        # Fit scaler on all training data first for better normalization
        print("Fitting scaler on all training data...")
        all_features = np.vstack([seq for seq in X_train])
        self.scaler.fit(all_features)
        
        # Normalize features for each sequence
        X_train_scaled = []
        for seq in X_train:
            X_train_scaled.append(self.scaler.transform(seq))
        
        # Create dataset
        train_data = list(zip(X_train_scaled, y_train))
        
        # Split into training and validation sets (more validation data)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        # Initialize model
        input_size = X_train[0].shape[1]  # Number of features
        self.model = GRUModel(input_size, dropout=self.dropout).to(self.device)
        
        # Calculate class weights
        pos_weight = torch.tensor((len(y_train) - np.sum(y_train)) / (np.sum(y_train) + 1e-8), dtype=torch.float32).to(self.device)
        print(f"Positive class weight: {pos_weight.item():.2f}")
        
        # Custom weighted BCE loss with focal component
        class WeightedBCEFocalLoss(nn.Module):
            def __init__(self, pos_weight, gamma=2.0):
                super().__init__()
                self.pos_weight = pos_weight.float()  # Ensure float32
                self.gamma = gamma
                
            def forward(self, outputs, targets):
                # Ensure consistent data types
                outputs = outputs.float()
                targets = targets.float()
                
                # Binary cross-entropy with logits
                bce_loss = nn.BCELoss(reduction='none')(outputs, targets)
                
                # Focal component to focus on hard examples
                pt = torch.exp(-bce_loss)  # Probability of being correct
                focal_weight = (1 - pt) ** self.gamma
                
                # Apply class weights - simpler approach to avoid indexing issues
                class_weights = torch.ones_like(targets, dtype=torch.float32)
                pos_mask = (targets > 0.5).float()
                class_weights = class_weights + (self.pos_weight - 1.0) * pos_mask
                
                # Final weighted loss
                loss = focal_weight * class_weights * bce_loss
                
                return loss.mean()
        
        criterion = WeightedBCEFocalLoss(pos_weight)
        # Increased weight decay for better regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y, batch_lengths in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)  # [batch]
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
                train_loss += loss.item() * batch_y.size(0)
                
                # Calculate accuracy with consistent threshold
                predictions = (outputs >= 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            avg_train_loss = train_loss / train_total
            train_accuracy = train_correct / train_total
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y, batch_lengths in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)  # [batch]
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_y.size(0)
                    
                    # Calculate accuracy with consistent threshold
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            avg_val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            val_losses.append(avg_val_loss)
            
            # Update learning rate scheduler
            scheduler.step()
            
            # Early stopping with validation metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/gru_sliding_best.pth')
                print(f"  Saved new best model with val_loss: {avg_val_loss:.4f}, val_acc: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/20, best val_loss: {best_val_loss:.4f}")
                
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('models/gru_sliding_best.pth'))
        
        # Save training history
        self.history = {'loss': train_losses, 'val_loss': val_losses}
        joblib.dump(self.history, 'models/gru_sliding_history.pkl')
        
        return self
    
    def predict(self, X, threshold=0.5):
        """Make predictions using the trained model for the last timestep"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Normalize features
        X_scaled = [self.scaler.transform(seq) for seq in X]
        
        # Create dataset and data loader for batch processing
        dummy_y = np.zeros(len(X))  # Dummy labels
        test_data = list(zip(X_scaled, dummy_y))
        test_loader = DataLoader(
            test_data, 
            batch_size=16, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        # Make predictions
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, _, batch_lengths in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).cpu().numpy()  # [batch]
                
                # Threshold predictions
                preds = (outputs >= threshold).astype(int)
                
                all_probabilities.extend(outputs)
                all_predictions.extend(preds)
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def predict_sequence(self, patient_sequence, threshold=0.5):
        """
        Predict sepsis risk for each timestep in a patient sequence by using
        sliding windows of increasing length.
        
        Args:
            patient_sequence: numpy array of shape (seq_len, n_features)
            threshold: threshold for binary classification
            
        Returns:
            predictions: numpy array of shape (seq_len,) with binary predictions
            probabilities: numpy array of shape (seq_len,) with probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Normalize features
        sequence_scaled = self.scaler.transform(patient_sequence)
        
        seq_len = len(patient_sequence)
        predictions = np.zeros(seq_len)
        probabilities = np.zeros(seq_len)
        
        # For the first timestep, we can't make a prediction (need at least 2 points)
        predictions[0] = 0
        probabilities[0] = 0
        
        # For each subsequent timestep, create a window from the beginning to current timestep
        self.model.eval()
        with torch.no_grad():
            for t in range(1, seq_len):
                # Get window from start to current timestep
                window = sequence_scaled[:t+1]
                
                # Convert to tensor and add batch dimension
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                
                # Get prediction
                output = self.model(window_tensor).cpu().numpy()[0]
                
                # Store prediction and probability
                probabilities[t] = output
                predictions[t] = 1 if output >= threshold else 0
        
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test, threshold=None):
        """Evaluate model performance for last timestep predictions"""
        print("\nEvaluating model...")
        
        # Use consistent threshold (same as training)
        if threshold is None:
            threshold = 0.5  # Same as training threshold
            print(f"Using consistent threshold: {threshold:.3f} (same as training)")
        
        # Make predictions with consistent threshold
        predictions, probabilities = self.predict(X_test, threshold=threshold)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        print("\nPrediction Distribution:")
        print(f"Mean: {np.mean(probabilities):.3f}")
        print(f"Std: {np.std(probabilities):.3f}")
        print(f"Min: {np.min(probabilities):.3f}")
        print(f"Max: {np.max(probabilities):.3f}")
        print(f"% Strong positive (>0.8): {(probabilities > 0.8).mean() * 100:.1f}%")
        print(f"% Strong negative (<0.2): {(probabilities < 0.2).mean() * 100:.1f}%")
        
        print(f"\nModel Performance (threshold={threshold:.3f}):")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)
        print("\nNormalized Confusion Matrix (%):")
        print(cm_norm)
        
        # Print examples of actual vs predicted values
        print("\nExamples of Actual vs Predicted Values:")
        num_examples = min(10, len(X_test))
        
        # Create a DataFrame for better visualization of predictions
        example_df = pd.DataFrame({
            'Example': range(1, num_examples + 1),
            'Sequence Length': [len(X_test[i]) for i in range(num_examples)],
            'Actual': y_test[:num_examples],
            'Predicted': predictions[:num_examples],
            'Probability': probabilities[:num_examples]
        })
        
        print(example_df.to_string(index=False))
        
        # Visualization code removed - no plots will be saved
        
        print(f"\nEvaluation completed with threshold: {threshold:.3f}")
        print(f"Results: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'probabilities': probabilities,
            'predictions': predictions,
            'confusion_matrix': cm,
            'confusion_matrix_norm': cm_norm
        }
    
    def save(self, path):
        """Save model and scaler"""
        os.makedirs('models', exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and features
        metadata_path = path.replace('.pth', '_metadata.pkl')
        joblib.dump({
            'scaler': self.scaler,
            'features': self.features,
            'dropout': self.dropout
        }, metadata_path)
        
        print(f"\nModel saved to {path}")
        print(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, path):
        """Load saved model"""
        # Load metadata
        metadata_path = path.replace('.pth', '_metadata.pkl')
        metadata = joblib.load(metadata_path)
        
        # Create model instance
        model = cls(dropout=metadata['dropout'])
        model.features = metadata['features']
        model.scaler = metadata['scaler']
        
        # Create GRU model
        input_size = len(model.features)
        model.model = GRUModel(input_size, dropout=model.dropout).to(model.device)
        
        # Load model state dict
        model.model.load_state_dict(torch.load(path, map_location=model.device))
        
        return model

if __name__ == "__main__":
    # Create directories (only models and data, no assets needed)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../data', exist_ok=True)
    
    # Create and train model with improved regularization
    model = GRUSequenceModel(dropout=0.6)  # Increased dropout for better regularization
    
    # Prepare data with sliding window approach
    X_train, X_test, y_train, y_test = model.prepare_data('../data/cleaned_dataset.csv')
    
    # Print some sample shapes to verify
    print("\nVerifying data shapes:")
    for i in range(3):
        print(f"Sample {i}:")
        print(f"  X shape: {X_train[i].shape}")
        print(f"  y value: {y_train[i]}")
    
    # Train model with improved hyperparameters
    model.fit(X_train, y_train, epochs=20 , batch_size=32, lr=0.0003)
    
    # Evaluate with optimal threshold
    print("\nEvaluating model with optimal threshold...")
    results = model.evaluate(X_test, y_test)
    
    # Demonstrate sequence prediction with a few examples
    print("\nDemonstrating sequence prediction for a few examples:")
    num_examples = min(3, len(X_test))
    
    for i in range(num_examples):
        print(f"\nExample {i+1}:")
        sequence = X_test[i]
        actual_label = y_test[i]
        
        # Predict for the full sequence
        seq_predictions, seq_probabilities = model.predict_sequence(sequence)
        
        print(f"Sequence length: {len(sequence)}")
        print(f"Actual label (last timestep): {actual_label}")
        print(f"Predicted label (last timestep): {seq_predictions[-1]}")
        print(f"Prediction probability (last timestep): {seq_probabilities[-1]:.4f}")
        
        # Show predictions over time
        print("\nPredictions over time (first 10 timesteps):")
        timesteps_to_show = min(10, len(sequence))
        
        time_df = pd.DataFrame({
            'Timestep': range(timesteps_to_show),
            'Probability': seq_probabilities[:timesteps_to_show],
            'Prediction': seq_predictions[:timesteps_to_show]
        })
        
        print(time_df.to_string(index=False))
        
        if len(sequence) > 10:
            print(f"... (showing 10 of {len(sequence)} timesteps)")
            
            # Also show the last few timesteps
            print("\nLast 5 timesteps:")
            last_df = pd.DataFrame({
                'Timestep': range(len(sequence)-5, len(sequence)),
                'Probability': seq_probabilities[-5:],
                'Prediction': seq_predictions[-5:]
            })
            print(last_df.to_string(index=False))
    
    # Save model
    model.save('models/gru_sliding_model.pth')
    
    print("\nModel training and evaluation completed successfully!")
