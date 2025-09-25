import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class TreeVotingModel:
    def __init__(self, n_trees=35):
        self.model = xgb.XGBClassifier(
            n_estimators=n_trees,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path):
        """Load and prepare the sepsis data"""
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        print("\nAll columns:", df.columns)
        print("\nSample data:")
        print(df.head())
        
        # Define columns to drop
        columns_drop = {
            'Unnamed: 0', 'SBP', 'DBP', 'EtCO2', 'BaseExcess', 'HCO3',
            'pH', 'PaCO2', 'Alkalinephos', 'Calcium', 'Magnesium',
            'Phosphate', 'Potassium', 'PTT', 'Fibrinogen', 'Unit1', 'Unit2'
        }
        
        # Drop specified columns
        df = df.drop(columns=[col for col in columns_drop if col in df.columns])
        print("\nDropped columns:", columns_drop)
        
        # Get features (all columns except Patient_ID and SepsisLabel)
        self.features = [col for col in df.columns if col not in ['Patient_ID', 'SepsisLabel']]
        print("\nUsing features:", self.features)
        print(f"Number of features: {len(self.features)}")
        
        # Prepare features and target
        X = df[self.features].fillna(0)
        y = df['SepsisLabel']
        
        # Print original class distribution
        print("\nOriginal Class Distribution:")
        print(f"No Sepsis: {(y == 0).sum()} ({(1 - y.mean()):.1%})")
        print(f"Sepsis: {(y == 1).sum()} ({y.mean():.1%})")
        
        # First split off a small portion for test (1% of data)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.01, random_state=42, stratify=y
        )
        
        # Create balanced test set
        sepsis_idx = np.where(y_test == 1)[0]
        non_sepsis_idx = np.where(y_test == 0)[0]
        
        # Sample equal numbers of each class
        n_samples = min(len(sepsis_idx), len(non_sepsis_idx))
        balanced_idx = np.concatenate([
            np.random.choice(sepsis_idx, n_samples, replace=False),
            np.random.choice(non_sepsis_idx, n_samples, replace=False)
        ])
        
        X_test = X_test.iloc[balanced_idx]
        y_test = y_test.iloc[balanced_idx]
        
        # Use remaining data for training
        X_train = X_temp
        y_train = y_temp
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print("\nTest Set Class Distribution:")
        print(f"No Sepsis: {(y_test == 0).sum()} (50%)")
        print(f"Sepsis: {(y_test == 1).sum()} (50%)")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X, y):
        """Train the model"""
        print("\nTraining model...")
        self.model.fit(X, y)
        return self
    
    def get_tree_predictions(self, X):
        """Get raw predictions from each tree (scores from 0 to 1)"""
        dtrain = xgb.DMatrix(X)
        trees_pred = self.model.get_booster().predict(
            dtrain,
            pred_contribs=True,
            approx_contribs=False
        )
        
        # Convert scores to probabilities with very low temperature for extreme predictions
        temperature = 0.1  # Very low temperature makes predictions more extreme
        raw_scores = trees_pred[:, :-1]  # Exclude bias term
        scaled_scores = raw_scores / temperature
        tree_scores = 1 / (1 + np.exp(-scaled_scores))
        
        return np.clip(tree_scores, 0, 1)
    
    def predict(self, X, threshold=0.5):
        tree_scores = self.get_tree_predictions(X)
        mean_scores = np.mean(tree_scores, axis=1)  # Simple average of tree predictions
        return (mean_scores >= threshold).astype(int)
    
    def find_best_threshold(self, y_true, y_pred_proba):
        """Find best threshold balancing precision and recall"""
        thresholds = np.linspace(0.1, 0.9, 81)  # Try thresholds from 0.1 to 0.9
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Update if better F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        return best_metrics
    
    def evaluate(self, X_test, y_test, threshold=None):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Get predictions
        tree_scores = self.get_tree_predictions(X_test)
        mean_scores = np.mean(tree_scores, axis=1)
        
        # Find best threshold if not provided
        if threshold is None:
            print("\nFinding best threshold...")
            metrics = self.find_best_threshold(y_test, mean_scores)
            threshold = metrics['threshold']
            print(f"Best threshold: {threshold:.3f}")
            print(f"At best threshold - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
        
        # Make predictions with chosen threshold
        y_pred = (mean_scores >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate tree disagreement
        disagreement = np.std(tree_scores, axis=1)
        
        print("\nPrediction Distribution:")
        print(f"Mean: {tree_scores.mean():.3f}")
        print(f"Std: {tree_scores.std():.3f}")
        print(f"Min: {tree_scores.min():.3f}")
        print(f"Max: {tree_scores.max():.3f}")
        print(f"% Strong positive (>0.8): {(tree_scores > 0.8).mean() * 100:.1f}%")
        print(f"% Strong negative (<0.2): {(tree_scores < 0.2).mean() * 100:.1f}%")
        
        print(f"\nModel Performance (threshold={threshold:.3f}):")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
        
        print("\nTree Disagreement:")
        print(f"Mean: {disagreement.mean():.3f}")
        print(f"Max: {disagreement.max():.3f}")
        print(f"% High disagreement (std>0.2): {(disagreement > 0.2).mean() * 100:.1f}%")
        
        # First figure: Distribution plots with original style
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Overall prediction distribution
        plt.subplot(121)
        sns.histplot(tree_scores.flatten(), bins=50)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.title('Distribution of Tree Predictions')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 2: Mean predictions by true class
        plt.subplot(122)
        sns.histplot(data=pd.DataFrame({
            'Mean Score': mean_scores,
            'Actual': y_test
        }), x='Mean Score', hue='Actual', bins=50)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.title('Distribution of Mean Predictions by Class')
        plt.xlabel('Mean Prediction Score')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('assets/prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Second figure: Confusion Matrix with enhanced style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Create confusion matrix figure
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum()  # Normalize by total samples
        
        # Create the base heatmap without annotations
        sns.heatmap(cm, cmap='Blues', annot=False,
                   xticklabels=['No Sepsis', 'Sepsis'],
                   yticklabels=['No Sepsis', 'Sepsis'],
                   cbar_kws={'label': 'Count'})
        
        # Add annotations with both count and percentage
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Calculate percentage of total
                percentage = cm_norm[i, j] * 100
                
                # Add count and percentage
                text = f'{cm[i, j]}\n({percentage:.1f}% of total)'
                
                plt.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='black' if cm_norm[i, j] < 0.5 else 'white')
        
        plt.title('Confusion Matrix\nPredicted vs Actual', pad=20, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', labelpad=10)
        plt.ylabel('Actual Label', labelpad=10)
        
        # Add summary statistics
        stats_text = (
            f'Model Performance Metrics:\n'
            f'Precision: {precision:.3f}\n'
            f'Recall: {recall:.3f}\n'
            f'F1 Score: {f1:.3f}\n\n'
            f'Total Samples: {cm.sum()}\n'
            f'True Negatives: {cm[0,0]}\n'
            f'False Positives: {cm[0,1]}\n'
            f'False Negatives: {cm[1,0]}\n'
            f'True Positives: {cm[1,1]}'
        )
        
        # Add text box with statistics
        plt.text(2.5, 0.5, stats_text,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10,
                transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('assets/confusion_matrix.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        plt.show()
        
        # Print confusion matrix details
        print("\nConfusion Matrix Details:")
        print(f"Total Samples: {cm.sum()}")
        print(f"True Negatives: {cm[0,0]} ({cm[0,0]/cm.sum():.1%} of total)")
        print(f"False Positives: {cm[0,1]} ({cm[0,1]/cm.sum():.1%} of total)")
        print(f"False Negatives: {cm[1,0]} ({cm[1,0]/cm.sum():.1%} of total)")
        print(f"True Positives: {cm[1,1]} ({cm[1,1]/cm.sum():.1%} of total)")
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tree_scores': tree_scores,
            'disagreement': disagreement,
            'confusion_matrix': cm,
            'confusion_matrix_norm': cm_norm
        }
    
    def save(self, path):
        """Save model and scaler"""
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, path)
        print(f"\nModel saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load saved model"""
        data = joblib.load(path)
        model = cls()
        model.model = data['model']
        model.scaler = data['scaler']
        model.features = data['features']
        return model

if __name__ == "__main__":
    # Create directories
    os.makedirs('assets', exist_ok=True)
    
    # Create and train model
    model = TreeVotingModel(n_trees=100)
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data('./data/cleaned_dataset.csv')
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    results = model.evaluate(X_test, y_test, threshold=0.5)
    
    # Save model
    model.save('models/tree_voting_model.pkl') 