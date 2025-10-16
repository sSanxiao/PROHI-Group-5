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
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path):
        """Load and prepare the sepsis data"""
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        print("\nAll columns:", df.columns)
        
        # Define columns to keep
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
        
        # Prepare features and target
        X = df[self.features].fillna(0)
        y = df['SepsisLabel']
        
        # Print feature statistics
        print("\nFeature Statistics:")
        for feature in self.features:
            print(f"\n{feature}:")
            print(f"  Mean: {X[feature].mean():.3f}")
            print(f"  Std: {X[feature].std():.3f}")
            print(f"  Min: {X[feature].min():.3f}")
            print(f"  Max: {X[feature].max():.3f}")
            print(f"  Missing: {X[feature].isna().sum()} ({X[feature].isna().mean():.1%})")
        
        # Print class distribution
        print("\nClass Distribution:")
        print(f"No Sepsis: {(y == 0).sum()} ({(1 - y.mean()):.1%})")
        print(f"Sepsis: {(y == 1).sum()} ({y.mean():.1%})")
        
        # Create balanced training set
        sepsis_samples = X[y == 1]
        non_sepsis_samples = X[y == 0]
        sepsis_labels = y[y == 1]
        non_sepsis_labels = y[y == 0]
        
        # Undersample majority class to match minority class
        n_sepsis = len(sepsis_samples)
        non_sepsis_indices = np.random.choice(len(non_sepsis_samples), n_sepsis, replace=False)
        balanced_non_sepsis = non_sepsis_samples.iloc[non_sepsis_indices]
        balanced_non_sepsis_labels = non_sepsis_labels.iloc[non_sepsis_indices]
        
        # Combine balanced datasets
        X_balanced = pd.concat([sepsis_samples, balanced_non_sepsis])
        y_balanced = pd.concat([sepsis_labels, balanced_non_sepsis_labels])
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.05,
            random_state=42,
            stratify=y_balanced
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print("\nBalanced Dataset:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print("\nTest Set Class Distribution:")
        print(f"No Sepsis: {(y_test == 0).sum()} ({(y_test == 0).mean():.1%})")
        print(f"Sepsis: {(y_test == 1).sum()} ({(y_test == 1).mean():.1%})")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X, y):
        """Train the model"""
        print("\nTraining model...")
        self.model.fit(X, y)
        
        # Print feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importances:")
        for f in range(len(self.features)):
            print(f"{self.features[indices[f]]}: {importances[indices[f]]:.4f}")
        
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
    
    def evaluate(self, X_test, y_test, threshold=None, save_plots=False):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Get predictions from all trees for all test samples
        all_predictions = []
        all_tree_scores = []
        
        for i in range(len(X_test)):
            # Get predictions for this sample
            tree_preds, tree_scores = self.get_tree_predictions(X_test[i:i+1])
            mean_score = np.mean(tree_scores)
            
            # Make prediction based on threshold
            if threshold is None:
                threshold = 0.5
            
            prediction = 1 if mean_score >= threshold else 0
            all_predictions.append(prediction)
            all_tree_scores.append(tree_scores)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_tree_scores = np.array(all_tree_scores)
        
        # Calculate metrics
        precision = precision_score(y_test, all_predictions)
        recall = recall_score(y_test, all_predictions)
        f1 = f1_score(y_test, all_predictions)
        
        # Calculate tree disagreement (across all samples)
        disagreement = np.mean([np.std(scores) for scores in all_tree_scores])
        
        print("\nPrediction Distribution:")
        mean_scores = np.mean(all_tree_scores, axis=1)  # Mean score per sample
        print(f"Mean: {np.mean(mean_scores):.3f}")
        print(f"Std: {np.std(mean_scores):.3f}")
        print(f"Min: {np.min(mean_scores):.3f}")
        print(f"Max: {np.max(mean_scores):.3f}")
        print(f"% Strong positive (>0.8): {(mean_scores > 0.8).mean() * 100:.1f}%")
        print(f"% Strong negative (<0.2): {(mean_scores < 0.2).mean() * 100:.1f}%")
        
        print(f"\nModel Performance (threshold={threshold:.3f}):")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, all_predictions)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Only create and save plots if requested
        if save_plots:
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
                'Score': mean_scores,
                'Actual': ['Sepsis' if y == 1 else 'No Sepsis' for y in y_test]
            }), x='Score', hue='Actual', bins=30, stat='density')
            
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
            plt.title('Distribution of Predictions by Actual Class')
            plt.xlabel('Prediction Score')
            plt.ylabel('Density')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('assets/prediction_distribution.png', 
                        dpi=300, 
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none')
            plt.close()  # Close figure instead of showing it
            
            # Second figure: Confusion Matrix
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
            plt.savefig('assets/confusion_matrix.png', 
                        dpi=300, 
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none')
            plt.close()  # Close figure instead of showing it
        
        # Print detailed confusion matrix statistics
        print("\nConfusion Matrix Statistics:")
        print("\nCounts and Percentages of Actual Values:")
        df_stats = pd.DataFrame(
            [[f"{cm[i,j]} ({cm_norm[i,j]:.1f}%)" for j in range(2)] for i in range(2)],
            columns=['Pred No Sepsis', 'Pred Sepsis'],
            index=['True No Sepsis', 'True Sepsis']
        )
        print(df_stats)
        
        # Print row sums for verification
        print("\nTotal samples per class:")
        print(f"True No Sepsis: {cm[0,:].sum()}")
        print(f"True Sepsis: {cm[1,:].sum()}")
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tree_scores': all_tree_scores,
            'predictions': all_predictions,
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
    os.makedirs('models', exist_ok=True)
    
    # Prepare data once for all models
    base_model = TreeVotingModel()
    X_train, X_test, y_train, y_test = base_model.prepare_data('./data/cleaned_dataset.csv')
    
    # Store the features list for use with all models
    features = base_model.features
    
    # Define 9 different hyperparameter configurations to test
    model_configs = [
        {"n_trees": 50, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 4},
        {"n_trees": 50, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
        {"n_trees": 50, "max_depth": 15, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_trees": 100, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 4},
        {"n_trees": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
        {"n_trees": 100, "max_depth": 15, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_trees": 200, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 4},
        {"n_trees": 200, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
        {"n_trees": 200, "max_depth": 15, "min_samples_split": 2, "min_samples_leaf": 1}
    ]
    
    # Train and evaluate all models
    results = []
    for i, config in enumerate(model_configs):
        print(f"\n\n{'='*80}")
        print(f"Training Model {i+1}/9: {config}")
        print(f"{'='*80}")
        
        # Create model with current config
        model = TreeVotingModel(
            n_trees=config["n_trees"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"]
        )
        
        # Set the features attribute from the base model
        model.features = features.copy()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate without saving plots
        eval_results = model.evaluate(X_test, y_test, threshold=0.5, save_plots=False)
        
        # Save model with numbered name
        model_path = f'models/random_forest_model_{i+1}.pkl'
        model.save(model_path)
        
        # Store results
        results.append({
            "model_id": i+1,
            "config": config,
            "f1": eval_results["f1"],
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "path": model_path
        })
    
    # Print summary table of all models
    print("\n\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':^10} | {'Trees':^8} | {'Depth':^8} | {'Min Split':^10} | {'Min Leaf':^10} | {'F1':^8} | {'Precision':^10} | {'Recall':^8}")
    print("-"*90)
    
    # Sort results by F1 score (descending)
    sorted_results = sorted(results, key=lambda x: x["f1"], reverse=True)
    
    for res in sorted_results:
        print(f"{res['model_id']:^10} | {res['config']['n_trees']:^8} | {res['config']['max_depth']:^8} | "
              f"{res['config']['min_samples_split']:^10} | {res['config']['min_samples_leaf']:^10} | "
              f"{res['f1']:.3f}    | {res['precision']:.3f}      | {res['recall']:.3f}")
    
    # Find the best model based on F1 score
    best_model = sorted_results[0]
    
    print("\n\n" + "="*80)
    print(f"BEST MODEL: Model {best_model['model_id']}")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Trees: {best_model['config']['n_trees']}")
    print(f"  - Max Depth: {best_model['config']['max_depth']}")
    print(f"  - Min Samples Split: {best_model['config']['min_samples_split']}")
    print(f"  - Min Samples Leaf: {best_model['config']['min_samples_leaf']}")
    print(f"Performance:")
    print(f"  - F1 Score: {best_model['f1']:.3f}")
    print(f"  - Precision: {best_model['precision']:.3f}")
    print(f"  - Recall: {best_model['recall']:.3f}")
    print(f"Path: {best_model['path']}")
    print("="*80)
    
    # Save a copy of the best model as the default model
    best_model_data = joblib.load(best_model['path'])
    joblib.dump(best_model_data, 'models/random_forest.pkl')
    print(f"\nBest model copied to models/random_forest.pkl")
    
    # Generate and save plots for the best model only
    print("\nGenerating plots for the best model...")
    best_model_instance = TreeVotingModel(
        n_trees=best_model['config']['n_trees'],
        max_depth=best_model['config']['max_depth'],
        min_samples_split=best_model['config']['min_samples_split'],
        min_samples_leaf=best_model['config']['min_samples_leaf']
    )
    best_model_instance.model = best_model_data['model']
    best_model_instance.scaler = best_model_data['scaler']
    best_model_instance.features = best_model_data['features']
    
    # Re-evaluate the best model with plot saving enabled
    best_model_instance.evaluate(X_test, y_test, threshold=0.5, save_plots=True)
    print("\nPlots for the best model saved to assets/ directory") 