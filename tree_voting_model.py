import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class TreeVotingModel:
    def __init__(self, n_trees=100):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=6,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
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
            'Unnamed: 0', 'Unit1', 'Unit2'
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
        """Get predictions from each tree"""
        tree_predictions = []
        tree_scores = []
        
        # Get predictions from each tree in the forest
        for tree in self.model.estimators_:
            pred = tree.predict(X)[0]
            prob = tree.predict_proba(X)[0]
            tree_predictions.append(int(pred))
            tree_scores.append(prob[1])  # Probability of class 1 (Sepsis)
        
        return np.array(tree_predictions), np.array(tree_scores)
    
    def predict(self, X, threshold=0.5):
        """Make predictions using mean tree probabilities"""
        tree_scores = self.get_tree_predictions(X)[1]  # Get probabilities
        mean_scores = np.mean(tree_scores)  # Average probability across trees
        return 1 if mean_scores >= threshold else 0
    
    def evaluate(self, X_test, y_test, threshold=None):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Get predictions from all trees
        tree_preds, tree_scores = self.get_tree_predictions(X_test)
        mean_scores = np.mean(tree_scores)
        
        # Make prediction based on threshold
        if threshold is None:
            threshold = 0.5
        
        prediction = 1 if mean_scores >= threshold else 0
        
        # Calculate metrics
        precision = precision_score(y_test, [prediction])
        recall = recall_score(y_test, [prediction])
        f1 = f1_score(y_test, [prediction])
        
        # Calculate tree disagreement
        disagreement = np.std(tree_scores)
        
        print("\nPrediction Distribution:")
        print(f"Mean: {mean_scores:.3f}")
        print(f"Std: {disagreement:.3f}")
        print(f"Min: {np.min(tree_scores):.3f}")
        print(f"Max: {np.max(tree_scores):.3f}")
        print(f"% Strong positive (>0.8): {(tree_scores > 0.8).mean() * 100:.1f}%")
        print(f"% Strong negative (<0.2): {(tree_scores < 0.2).mean() * 100:.1f}%")
        
        print(f"\nModel Performance (threshold={threshold:.3f}):")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
        
        # Create confusion matrix visualization
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, [prediction])
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum()
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Sepsis', 'Sepsis'],
                   yticklabels=['No Sepsis', 'Sepsis'])
        
        plt.title('Confusion Matrix\nPredicted vs Actual')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        
        plt.tight_layout()
        plt.savefig('assets/confusion_matrix.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.show()
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tree_scores': tree_scores,
            'disagreement': disagreement
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
    model.save('models/random_forest_model.pkl') 