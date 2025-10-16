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
    def __init__(self, input_size, dropout=0.4):
        super(GRUModel, self).__init__()
        
        # Enhanced GRU architecture with bidirectional layers
        self.gru1 = nn.GRU(input_size, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.gru2 = nn.GRU(128*2, 64, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.Linear(64*2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Classification layers
        self.fc1 = nn.Linear(64, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 1)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Create a mask for padding (if any)
        # Assuming zeros are padding
        mask = (x.sum(dim=2) != 0).float().unsqueeze(-1)  # [batch, seq_len, 1]
        
        # GRU processing with bidirectional layers
        gru1_out, _ = self.gru1(x)  # [batch, seq_len, 128*2]
        gru2_out, _ = self.gru2(gru1_out)  # [batch, seq_len, 64*2]
        
        # We only need the last timestep's prediction
        # Get the last valid timestep for each sequence using the mask
        batch_size = x.size(0)
        
        # Apply attention to focus on important timesteps
        attention_scores = self.attention(gru2_out)  # [batch, seq_len, 1]
        
        # Apply mask to attention scores (set padding to large negative)
        attention_scores = attention_scores + (1 - mask) * (-1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
        
        # Apply attention weights to get context vector
        context_vector = torch.sum(gru2_out * attention_weights, dim=1)  # [batch, 64*2]
        
        # Apply feature transformation
        features = self.feature_transform(context_vector)
        
        # Apply classification layers
        x1 = self.fc1(features)
        x1 = self.ln1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout1(x1)
        
        # Final classification
        logits = self.fc2(x1)  # [batch, 1]
        output = torch.sigmoid(logits).squeeze(-1)  # [batch]
        
        return output

class GRUSequenceModel:
    def __init__(self, dropout=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.dropout = dropout
        
    def prepare_data(self, data_path):
        """Load and prepare the sepsis data with sliding windows, predicting only the last timestep"""
        # Check if prepared data exists
        prepared_data_path = 'data/prepared_gru_data_sliding.pkl'
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
        
        print("Loading and preparing data...")
        
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
        
        # Create sliding window sequences for each patient
        print("\nCreating sliding window sequences for each patient...")
        X_windows = []
        y_windows = []
        patient_ids_windows = []  # Keep track of which patient each window belongs to
        
        # Process all patients
        patient_ids = df['Patient_ID'].unique()
        print(f"Processing {len(patient_ids)} patients...")
        
        # Count total windows created
        total_windows = 0
        sepsis_windows = 0
        
        for i, pid in enumerate(tqdm(patient_ids, desc="Processing patients")):
            group = df[df['Patient_ID'] == pid].sort_values('Hour')
            
            if len(group) > 1:  # Only process if patient has more than one data point
                features = group[self.features].values.astype(np.float32)
                labels = group["SepsisLabel"].values.astype(np.float32)
                
                # Create sliding windows of increasing length
                for end_idx in range(1, len(group)):
                    # Get window from start to current end_idx
                    window_features = features[:end_idx+1]
                    window_label = labels[end_idx]  # Label is the last timestep's label
                    
                    X_windows.append(window_features)
                    y_windows.append(window_label)
                    patient_ids_windows.append(pid)
                    
                    total_windows += 1
                    if window_label == 1:
                        sepsis_windows += 1
            
            # Progress update every 1000 patients
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(patient_ids)} patients")
        
        print(f"\nCreated {total_windows} sliding windows from {len(patient_ids)} patients")
        print(f"Sepsis rate in windows: {sepsis_windows/total_windows:.1%}")
        
        # Convert to numpy arrays
        X_windows_array = np.array(X_windows, dtype=object)
        y_windows_array = np.array(y_windows)
        
        # Create balanced training set
        sepsis_indices = np.where(y_windows_array == 1)[0]
        non_sepsis_indices = np.where(y_windows_array == 0)[0]
        
        # Undersample majority class to match minority class
        # If we have too few sepsis samples, we'll use all of them and undersample non-sepsis
        if len(sepsis_indices) < 1000:
            n_samples = len(sepsis_indices)
        else:
            n_samples = min(len(sepsis_indices), len(non_sepsis_indices))
            
        # Randomly select n_samples from each class
        selected_sepsis_indices = np.random.choice(sepsis_indices, n_samples, replace=False)
        selected_non_sepsis_indices = np.random.choice(non_sepsis_indices, n_samples, replace=False)
        
        # Combine indices and extract balanced dataset
        balanced_indices = np.concatenate([selected_sepsis_indices, selected_non_sepsis_indices])
        X_balanced = [X_windows_array[i] for i in balanced_indices]
        y_balanced = y_windows_array[balanced_indices]
        
        # Print a sample
        print("\nSample window:")
        print(f"X shape: {X_balanced[0].shape}")
        print(f"y value: {y_balanced[0]}")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.05,
            random_state=42,
            stratify=y_balanced  # Stratify by sepsis label
        )
        
        print("\nBalanced Dataset:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Calculate class distribution
        print("\nClass Distribution:")
        print(f"No Sepsis: {(y_balanced == 0).sum()} ({(y_balanced == 0).mean():.1%})")
        print(f"Sepsis: {(y_balanced == 1).sum()} ({(y_balanced == 1).mean():.1%})")
        
        print("\nTest Set Class Distribution:")
        print(f"No Sepsis: {(y_test == 0).sum()} ({(y_test == 0).mean():.1%})")
        print(f"Sepsis: {(y_test == 1).sum()} ({(y_test == 1).mean():.1%})")
        
        # Save prepared data
        os.makedirs(os.path.dirname(prepared_data_path), exist_ok=True)
        joblib.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'features': self.features
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
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, lr=0.0005):
        """Train the GRU model to predict the last timestep"""
        print("\nTraining model...")
        
        # Normalize features for each sequence
        X_train_scaled = []
        for seq in X_train:
            X_train_scaled.append(self.scaler.fit_transform(seq))
        
        # Create dataset
        train_data = list(zip(X_train_scaled, y_train))
        
        # Split into training and validation sets
        train_size = int(0.95 * len(train_data))
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
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
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
                
                # Calculate accuracy
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
                    
                    # Calculate accuracy
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            avg_val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            val_losses.append(avg_val_loss)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping with validation metrics
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/gru_sliding_best.pth')
                print(f"  Saved new best model with val_loss: {avg_val_loss:.4f}, val_acc: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/10, best val_loss: {best_val_loss:.4f}")
                
            if patience_counter >= 10:
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
        
        # Make predictions
        predictions, probabilities = self.predict(X_test, threshold=0.5 if threshold is None else threshold)
        
        # Find optimal threshold if not provided
        if threshold is None:
            best_f1 = 0
            best_threshold = 0.5
            
            for t in np.arange(0.1, 0.9, 0.05):
                y_pred = (probabilities >= t).astype(int)
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            
            threshold = best_threshold
            print(f"Optimal threshold: {threshold:.3f}")
            
            # Update predictions with optimal threshold
            predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
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
        
        # Set up the visualization style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # First figure: Distribution of predictions
        plt.figure(figsize=(12, 6))
        sns.histplot(data=pd.DataFrame({
            'Score': probabilities,
            'Actual': ['Sepsis' if y == 1 else 'No Sepsis' for y in y_test]
        }), x='Score', hue='Actual', bins=30, stat='density')
        
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.title('Distribution of Predictions by Actual Class')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('assets/prediction_distributions_sliding.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        plt.close()
        
        # Second figure: ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('assets/roc_curves_sliding.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        plt.close()
        
        # Third figure: Confusion Matrix
        plt.figure(figsize=(10, 8))
        
        # Create heatmap without annotations first
        sns.heatmap(cm, cmap='Blues', annot=False,
                   xticklabels=['No Sepsis', 'Sepsis'],
                   yticklabels=['No Sepsis', 'Sepsis'])
        
        # Add annotations with count and percentage
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f'{cm[i,j]}\n({cm_norm[i,j]:.1f}%)'
                plt.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center',
                        color='black' if cm[i,j] / cm[i,:].sum() < 0.5 else 'white')
        
        plt.title('Confusion Matrix\nPredicted vs Actual')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        
        plt.tight_layout()
        plt.savefig('assets/confusion_matrix_sliding.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        plt.close()
        
        # Plot training history if available
        if hasattr(self, 'history'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Training History')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('assets/training_history_sliding.png', 
                        dpi=300, 
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none')
            plt.close()
        
        # Visualize predictions for full sequences for a few examples
        # This demonstrates how the model would predict on a full patient sequence
        print("\nFull sequence predictions for selected examples:")
        
        # Select a few examples with different outcomes
        sepsis_indices = np.where(y_test == 1)[0]
        non_sepsis_indices = np.where(y_test == 0)[0]
        
        # Get up to 2 examples of each class
        sepsis_examples = sepsis_indices[:2] if len(sepsis_indices) > 0 else []
        non_sepsis_examples = non_sepsis_indices[:2] if len(non_sepsis_indices) > 0 else []
        selected_examples = list(sepsis_examples) + list(non_sepsis_examples)
        
        if selected_examples:
            plt.figure(figsize=(15, 10))
            
            for idx, i in enumerate(selected_examples[:4]):  # Plot up to 4 examples
                # Get the full sequence
                sequence = X_test[i]
                
                # Predict for each timestep in the sequence
                seq_predictions, seq_probabilities = self.predict_sequence(sequence, threshold)
                
                plt.subplot(len(selected_examples[:4]), 1, idx+1)
                
                x_axis = np.arange(len(sequence))
                
                # Plot predicted probabilities
                plt.plot(x_axis, seq_probabilities, label='Predicted Probability', color='red')
                
                # Plot threshold line
                plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, 
                           label=f'Threshold ({threshold:.2f})')
                
                # Plot predicted values
                plt.step(x_axis, seq_predictions, where='post', label='Predicted', color='green', alpha=0.7)
                
                # Highlight the final prediction (which is what we evaluated)
                plt.scatter([len(sequence)-1], [seq_probabilities[-1]], color='blue', s=100, 
                           label='Final Prediction', zorder=5)
                
                # Add a title with the actual label
                label_text = "Sepsis" if y_test[i] == 1 else "No Sepsis"
                pred_text = "Sepsis" if predictions[i] == 1 else "No Sepsis"
                plt.title(f'Example {i+1}: Actual: {label_text}, Predicted: {pred_text} ' +
                         f'(prob: {probabilities[i]:.3f}), Sequence Length: {len(sequence)}')
                
                plt.xlabel('Timestep')
                plt.ylabel('Sepsis Probability')
                plt.ylim(-0.1, 1.1)
                
                # Only show legend for the first subplot
                if idx == 0:
                    plt.legend(loc='upper left')
            
            plt.tight_layout()
            plt.savefig('assets/sequence_predictions_sliding.png', 
                        dpi=300, 
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none')
            plt.close()
            
            # Print detailed sequence predictions for one example
            if len(selected_examples) > 0:
                example_idx = selected_examples[0]
                sequence = X_test[example_idx]
                seq_predictions, seq_probabilities = self.predict_sequence(sequence, threshold)
                
                print(f"\nDetailed predictions for Example {example_idx+1} (Actual: {y_test[example_idx]}):")
                detail_df = pd.DataFrame({
                    'Timestep': range(len(sequence)),
                    'Probability': seq_probabilities,
                    'Prediction': seq_predictions
                })
                
                # Show up to 20 timesteps
                display_len = min(20, len(sequence))
                print(detail_df.head(display_len).to_string(index=False))
                if len(sequence) > display_len:
                    print(f"... (showing {display_len} of {len(sequence)} timesteps)")
        
        return {
            'threshold': threshold,
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
    # Create directories
    os.makedirs('assets', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create and train model
    model = GRUSequenceModel(dropout=0.3)  # Reduced dropout
    
    # Prepare data with sliding window approach
    X_train, X_test, y_train, y_test = model.prepare_data('./data/cleaned_dataset.csv')
    
    # Print some sample shapes to verify
    print("\nVerifying data shapes:")
    for i in range(3):
        print(f"Sample {i}:")
        print(f"  X shape: {X_train[i].shape}")
        print(f"  y value: {y_train[i]}")
    
    # Train model with improved hyperparameters
    model.fit(X_train, y_train, epochs=50, batch_size=32, lr=0.0003)
    
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
