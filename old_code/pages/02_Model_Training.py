import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Model Training & Results")
st.markdown("---")

# Introduction
st.markdown("""
This section presents our machine learning approach to sepsis prediction, comparing two distinct modeling strategies:
**Tabular Models** for individual time-point analysis and **Sequential Models** for temporal pattern recognition.
""")

# Model Approaches Section
st.header("🎯 Modeling Approaches")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Tabular Models")
    st.markdown("""
    **Approach**: Traditional machine learning on individual patient records
    
    **Models Implemented**:
    - **Random Forest**: Ensemble method with 100 decision trees
    - **Logistic Regression**: Linear classifier with L2 regularization
    
    **Data Format**: Each row represents a single time point
    - Shape: `(samples × features)` 
    - Features: 40 clinical variables (vital signs, lab values, demographics)
    
    **Advantages**:
    - ✅ Fast training and inference
    - ✅ High interpretability (feature importance)
    - ✅ Works well with structured data
    - ✅ Less memory intensive
    
    **Limitations**:
    - ❌ No temporal pattern recognition
    - ❌ Treats each time point independently
    """)

with col2:
    st.subheader("⏰ Sequential Models")
    st.markdown("""
    **Approach**: Deep learning on patient time series sequences
    
    **Model Implemented**:
    - **GRU (Gated Recurrent Unit)**: Advanced RNN architecture
      - 2-layer GRU (128 → 64 hidden units)
      - Attention mechanism for important time steps
      - Batch normalization and dropout regularization
    
    **Data Format**: Sequences of patient observations
    - Shape: `(samples × timesteps × features)`
    - Sequence length: 12 hours (12-hour window)
    - Architecture: 2-layer GRU (128→64 units) + Attention + Classification layers
    
    **Advantages**:
    - ✅ Captures temporal deterioration patterns
    - ✅ Learns from patient history trends  
    - ✅ Attention mechanism highlights critical periods
    - ✅ Better for early warning systems
    
    **Limitations**:
    - ❌ Computationally intensive
    - ❌ Requires sequential data
    - ❌ Less interpretable
    """)

# Training Details
st.header("🚀 Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Data Preparation")
    st.markdown("""
    - **Dataset Size**: 1.5M+ records, 40K+ patients
    - **Sequence Creation**: 12-hour windows per patient
    - **Feature Scaling**: StandardScaler normalization
    - **Train/Test Split**: 80/20 stratified split
    - **Class Balance**: Weighted loss functions (sepsis ~1.8%)
    """)

with col2:
    st.subheader("🔧 Implementation Pipelines")
    st.markdown("""
    **Tabular Data Pipeline**:
    ```
    Raw Data → Missing Value Imputation → Feature Scaling → 
    Train/Test Split → Model Training
    ```
    
    **Sequential Data Pipeline**:
    ```
    Raw Data → Patient Grouping → Sequence Creation (12h windows) → 
    Feature Scaling → Train/Val/Test Split → Model Training
    ```
    """)



# Results Section
st.header("📊 Training Results")

# Check if results are available
if os.path.exists('assets/metrics_comparison.png'):
    
    # Performance Overview
    st.subheader("🎯 Model Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image('assets/roc_curves_comparison.png', 
                caption="ROC Curves - Higher AUC indicates better discrimination between sepsis and non-sepsis cases",
                use_container_width=True)
    
    with col2:
        st.image('assets/auc_comparison.png',
                caption="AUC Comparison - Area Under the ROC Curve for each model",
                use_container_width=True)
    
    # Detailed Metrics
    st.subheader("📈 Detailed Performance Metrics")
    st.image('assets/metrics_comparison.png',
            caption="Comprehensive comparison of all evaluation metrics across models",
            use_container_width=True)
    
    # Training Progress
    st.subheader("📉 Training History")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image('assets/training_history.png',
                caption="Training and validation loss curves showing model learning progress",
                use_container_width=True)
    
    with col2:
        st.markdown("""
        **Training Insights**:
        
        🔍 **Loss Curves Analysis**:
        - Decreasing training loss shows learning
        - Validation loss tracks training loss
        - Early stopping prevents overfitting
        
        🎯 **Convergence**:
        - Stable convergence typically achieved
        - Learning rate scheduling helps optimization
        - Gradient clipping ensures stability
        
        ⚠️ **Monitoring**:
        - Watch for overfitting (val_loss >> train_loss)
        - Plateau indicates convergence
        - Early stopping saves best model
        """)
    
    # Confusion Matrices
    st.subheader("🔍 Model Confusion Matrices")
    st.markdown("Detailed analysis of prediction accuracy for sepsis detection:")
    
    # Check for confusion matrix files
    confusion_files = {
        'Random Forest': 'assets/random_forest_confusion_matrix.png',
        'Logistic Regression': 'assets/logistic_regression_confusion_matrix.png', 
        'GRU': 'assets/gru_confusion_matrix.png'
    }
    
    cols = st.columns(3)
    for i, (model_name, file_path) in enumerate(confusion_files.items()):
        if os.path.exists(file_path):
            with cols[i]:
                st.image(file_path, caption=f"{model_name} Confusion Matrix", 
                        use_container_width=True)
    
else:
    st.warning("⚠️ Model training results not found. Please run the model training notebook first.")
    st.markdown("""
    To generate training results:
    1. Open `sepsis_prediction_models.ipynb`
    2. Run all cells to train models
    3. Refresh this page to see results
    """)

# Key Findings Section
st.header("🔑 Key Findings from Model Training")

# Performance Results Table
st.subheader("📊 Model Performance Results")

results_data = {
    "Model": ["Random Forest", "Logistic Regression", "GRU"],
    "Type": ["Tabular", "Tabular", "Sequential"],
    "AUC": [0.999, 0.746, 0.911],
    "Precision": [0.948, 0.044, 0.356],
    "Recall": [0.731, 0.633, 0.601],
    "F1-Score": [0.826, 0.082, 0.447],
    "Data Shape": ["310442 × 40", "310442 × 40", "17334 × 12 × 40"]
}

results_df = pd.DataFrame(results_data)
st.dataframe(results_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Best Model: Random Forest")
    st.markdown("""
    **Performance Metrics**:
    - **AUC**: 0.999 (Near perfect discrimination)
    - **Precision**: 0.948 (94.8% accuracy when predicting sepsis)
    - **Recall**: 0.731 (73.1% of sepsis cases detected)
    - **F1-Score**: 0.826 (Excellent balance)
    
    **⚠️ Clinical Consideration - Recall Issue**:
    - **Recall of 73.1% means 26.9% of sepsis cases are missed**
    - In clinical settings, missing sepsis cases is more critical than false alarms
    - **Solution**: Experiment with thresholds lower than 0.5 to increase recall
    - Lower threshold = catch more sepsis cases but more false positives
    """)

with col2:
    st.subheader("📈 Model Comparison Insights")
    st.markdown("""
    **Sequential Model (GRU) Performance**:
    - **AUC**: 0.911 (Good discrimination)
    - **Recall**: 0.601 (60.1% sepsis detection)
    - **Challenge**: Lower precision (35.6%) - more false alarms
    
    **Key Observations**:
    - 🎯 Tabular models outperformed sequential for this dataset
    - 🔍 Random Forest achieved near-perfect AUC
    - ⚖️ Trade-off between precision and recall varies by model
    - 📊 Feature engineering may be more important than temporal modeling
    """)



st.markdown("---")
st.markdown("*Navigate to other pages to explore data preprocessing and feature importance analysis.*") 