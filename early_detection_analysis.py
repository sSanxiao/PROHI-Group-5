import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tree_voting_model import TreeVotingModel
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# Load model and data
model = TreeVotingModel.load('models/random_forest.pkl')
df = pd.read_csv('./data/cleaned_dataset.csv')

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'Unit1', 'Unit2'], axis=1)

# Get features (all columns except Patient_ID and SepsisLabel)
features = [col for col in df.columns if col not in ['Patient_ID', 'SepsisLabel']]

# Prepare features and target for splitting
X = df[features].fillna(0)
y = df['SepsisLabel']

# Create balanced dataset (same as tree_voting_model.py)
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

# Split using same parameters as tree_voting_model.py
_, X_test, _, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.05,
    random_state=42,
    stratify=y_balanced
)

# Get test patient IDs
test_patients = df[df.index.isin(X_test.index)]['Patient_ID'].unique()
print(f"Analyzing {len(test_patients)} test set patients")

# Store earliest detection time for each patient
earliest_detections = {}  # patient_id -> earliest detection time
confidence_over_time = []

# Analyze each test patient with sepsis
sepsis_test_patients = []
for patient_id in test_patients:
    patient_data = df[df['Patient_ID'] == patient_id].copy()
    if 1 in patient_data['SepsisLabel'].values:
        sepsis_test_patients.append(patient_id)

print(f"Found {len(sepsis_test_patients)} sepsis patients in test set")

# Analyze each sepsis patient
for patient_id in tqdm(sepsis_test_patients, desc="Analyzing patients"):
    patient_data = df[df['Patient_ID'] == patient_id].copy()
    sepsis_onset = patient_data[patient_data['SepsisLabel'] == 1]['Hour'].iloc[0]
    
    # Analyze each hour in 24-hour window before sepsis
    patient_predictions = []
    earliest_detection = None
    
    for hour in sorted(patient_data['Hour'].unique()):
        hours_before = sepsis_onset - hour
        # Only look at 24-hour window before sepsis
        if hours_before > 24 or hours_before <= 0:
            continue
            
        # Get data for this hour
        hour_data = patient_data[patient_data['Hour'] == hour]
        X = hour_data[model.features].fillna(0)
        X_scaled = model.scaler.transform(X)
        
        # Get predictions from all trees
        _, tree_scores = model.get_tree_predictions(X_scaled)
        confidence = np.mean(tree_scores)
        
        patient_predictions.append({
            'hours_before_sepsis': hours_before,
            'confidence': confidence,
            'patient_id': patient_id
        })
        
        # Check if this would have been an early detection
        if confidence >= 0.55 and earliest_detection is None:
            earliest_detection = hours_before
    
    # Store earliest detection if found
    if earliest_detection is not None:
        earliest_detections[patient_id] = earliest_detection
            
    confidence_over_time.extend(patient_predictions)

# Convert earliest detections to list
detection_hours = list(earliest_detections.values())

# Count undetected patients
undetected_count = len(sepsis_test_patients) - len(detection_hours)

# Create detection time distribution plot
plt.figure(figsize=(12, 6))

# Plot detected cases in blue
plt.hist(detection_hours, bins=20, color='skyblue', edgecolor='black', 
         label=f'Early Detection ({len(detection_hours)} patients)')

# Add undetected cases as a single bar at 0 hours in red
if undetected_count > 0:
    plt.bar(0, undetected_count, color='red', width=1.2, 
            label=f'No Early Detection ({undetected_count} patients)')

plt.title('Distribution of Earliest Sepsis Detection Times (24h Window)')
plt.xlabel('Hours Before Actual Sepsis Onset')
plt.ylabel('Number of Patients')
plt.axvline(np.mean(detection_hours), color='green', linestyle='--', 
            label=f'Mean Detection Time: {np.mean(detection_hours):.1f} hours')
plt.legend()
plt.savefig('assets/early_detection_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create confidence progression plot
df_confidence = pd.DataFrame(confidence_over_time)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_confidence, x='hours_before_sepsis', y='confidence', 
            ci=95, color='blue')
plt.title('Average Prediction Confidence vs Hours Before Sepsis (24h Window)')
plt.xlabel('Hours Before Sepsis')
plt.ylabel('Model Confidence')
plt.axhline(y=0.55, color='red', linestyle='--', label='Detection Threshold (0.55)')
plt.xlim(24, 0)  # Reverse x-axis from 24 to 0
plt.legend()
plt.savefig('assets/confidence_progression.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print(f"\nEarly Detection Statistics (24h Window):")
print(f"Patients with early detection: {len(detection_hours)} out of {len(sepsis_test_patients)}")
print(f"Patients without early detection: {undetected_count}")
print(f"Average early detection: {np.mean(detection_hours):.1f} hours before onset")
print(f"Median early detection: {np.median(detection_hours):.1f} hours before onset")
print(f"Earliest detection: {np.max(detection_hours):.1f} hours before onset")
print(f"Latest detection: {np.min(detection_hours):.1f} hours before onset") 