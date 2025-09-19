import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Feature Importance", page_icon="🔬", layout="wide")

st.title("🔬 Feature Importance Analysis")
st.markdown("---")

# Introduction
st.markdown("""
This section analyzes which clinical features are most important for sepsis prediction and provides clinical interpretation 
of why these features are significant in identifying patients at risk of developing sepsis.
""")

# Feature Importance Visualization
st.header("📊 Most Important Features")

if os.path.exists('assets/feature_importance.png'):
    st.image('assets/feature_importance.png', 
            caption="Top 15 most important features identified by Random Forest model for sepsis prediction",
            use_container_width=True)
    
    st.markdown("""
    The feature importance plot above shows the relative importance of different clinical variables in predicting sepsis. 
    Higher values indicate features that contribute more to the model's decision-making process.
    """)
else:
    st.warning("⚠️ Feature importance plot not found. Please run the model training notebook first.")

# Model-Based Feature Analysis
st.header("🩺 Model-Based Feature Analysis")

def get_clinical_significance(feature_code, importance_value):
    """Get clinical significance - mostly empty for manual writing"""
    
    # Only keep a few key features with descriptions, leave most empty
    significance_map = {
        'ICULOS': f"Model importance: {importance_value:.3f}. [Add clinical description here]",
        'Temp': f"Model importance: {importance_value:.3f}. [Add clinical description here]", 
        'WBC': f"Model importance: {importance_value:.3f}. [Add clinical description here]",
        'Lactate': f"Model importance: {importance_value:.3f}. [Add clinical description here]",
        'HR': f"Model importance: {importance_value:.3f}. [Add clinical description here]"
    }
    
    # For all other features, return just the importance value
    return significance_map.get(feature_code, f"Model importance: {importance_value:.3f}. [Add clinical description here]")

# Load the trained Random Forest model and extract feature importances
@st.cache_data
def load_feature_importances():
    """Load feature importances from saved data files (bypasses model loading issues)"""
    
    # Method 1: Try to load from CSV (most reliable)
    if os.path.exists('assets/feature_importance_data.csv'):
        try:
            feature_importance_df = pd.read_csv('assets/feature_importance_data.csv')
            st.success("✅ Feature importances loaded from saved CSV data")
            return feature_importance_df, True
        except Exception as e:
            st.warning(f"CSV loading failed: {str(e)}")
    
    # Method 2: Try to load from JSON backup
    if os.path.exists('assets/feature_importance_raw.json'):
        try:
            import json
            with open('assets/feature_importance_raw.json', 'r') as f:
                importance_dict = json.load(f)
            
            # Convert to dataframe
            feature_importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in importance_dict.items()
            ]).sort_values('importance', ascending=False)
            
            st.success("✅ Feature importances loaded from JSON backup")
            return feature_importance_df, True
        except Exception as e:
            st.warning(f"JSON loading failed: {str(e)}")
    
    # If we reach here, both CSV and JSON loading failed
    st.error("❌ Could not load feature importance data from any source")
    st.info("💡 Make sure to run the notebook first to generate the feature importance files")
    return None, False

# Load feature importances
feature_importance_df, model_loaded = load_feature_importances()

if model_loaded and feature_importance_df is not None:
    st.success("✅ Loaded actual feature importances from trained Random Forest model")
    
    # Define importance thresholds based on actual data (5 categories)
    max_importance = feature_importance_df['importance'].max()
    very_high_threshold = max_importance * 0.5   # Top 50% of max importance
    high_threshold = max_importance * 0.25       # Top 25% of max importance  
    medium_threshold = max_importance * 0.12     # Top 12% of max importance
    low_threshold = max_importance * 0.05        # Top 5% of max importance
    
    st.markdown(f"""
    The following features are ranked by their **actual importance** in the trained Random Forest model. 
    All **{len(feature_importance_df)} features** are shown with color-coded importance levels:
    
    - **Very High (🔴)** ≥{very_high_threshold:.3f}
    - **High (🟠)** ≥{high_threshold:.3f} 
    - **Medium (🟡)** ≥{medium_threshold:.3f}
    - **Low (🟢)** ≥{low_threshold:.3f}
    - **Very Low (🔵)** <{low_threshold:.3f}
    """)
    
    # Create model features list with actual importances (ALL FEATURES)
    model_features = []
    for _, row in feature_importance_df.iterrows():  # ALL features, not just top 15
        feature_code = row['feature']
        importance_value = row['importance']
        
        # Determine importance level based on thresholds (5 categories)
        if importance_value >= very_high_threshold:
            importance_level = "🔴 Very High"
        elif importance_value >= high_threshold:
            importance_level = "🟠 High"
        elif importance_value >= medium_threshold:
            importance_level = "🟡 Medium"
        elif importance_value >= low_threshold:
            importance_level = "🟢 Low"
        else:
            importance_level = "🔵 Very Low"
        
        # Map feature codes to readable names (ALL 41 features)
        feature_names = {
            # Vital Signs
            'HR': 'Heart Rate',
            'O2Sat': 'Oxygen Saturation',
            'Temp': 'Body Temperature',
            'SBP': 'Systolic Blood Pressure',
            'MAP': 'Mean Arterial Pressure',
            'DBP': 'Diastolic Blood Pressure',
            'Resp': 'Respiratory Rate',
            'EtCO2': 'End-tidal CO2',
            
            # Blood Gas & Acid-Base
            'BaseExcess': 'Base Excess',
            'HCO3': 'Bicarbonate',
            'FiO2': 'Fraction of Inspired Oxygen',
            'pH': 'Blood pH',
            'PaCO2': 'Arterial CO2 Pressure',
            'SaO2': 'Arterial Oxygen Saturation',
            
            # Organ Function
            'AST': 'Aspartate Aminotransferase',
            'BUN': 'Blood Urea Nitrogen',
            'Alkalinephos': 'Alkaline Phosphatase',
            'Calcium': 'Serum Calcium',
            'Chloride': 'Serum Chloride',
            'Creatinine': 'Serum Creatinine',
            'Bilirubin_direct': 'Direct Bilirubin',
            'Bilirubin_total': 'Total Bilirubin',
            
            # Metabolic & Electrolytes
            'Glucose': 'Blood Glucose',
            'Lactate': 'Serum Lactate',
            'Magnesium': 'Serum Magnesium',
            'Phosphate': 'Serum Phosphate',
            'Potassium': 'Serum Potassium',
            
            # Hematology & Cardiac
            'TroponinI': 'Troponin I',
            'Hct': 'Hematocrit',
            'Hgb': 'Hemoglobin',
            'PTT': 'Partial Thromboplastin Time',
            'WBC': 'White Blood Cell Count',
            'Fibrinogen': 'Fibrinogen',
            'Platelets': 'Platelet Count',
            
            # Demographics & Temporal
            'Age': 'Patient Age',
            'Gender': 'Patient Gender',
            'Unit1': 'ICU Unit Type 1',
            'Unit2': 'ICU Unit Type 2',
            'HospAdmTime': 'Hospital Admission Time',
            'ICULOS': 'ICU Length of Stay'
        }
        
        feature_name = feature_names.get(feature_code, feature_code)
        
        # Clinical significance based on what the model learned
        clinical_significance = get_clinical_significance(feature_code, importance_value)
        
        model_features.append({
            'feature': feature_code,
            'name': feature_name,
            'importance': importance_level,
            'importance_value': importance_value,
            'clinical_significance': clinical_significance
        })
        
else:
    st.warning("⚠️ Could not load trained Random Forest model.")
    
    with st.expander("🔧 Troubleshooting Steps"):
        st.markdown("""
        **If you see a ThreadpoolController import error:**
        1. Update your packages: `pip install --upgrade scikit-learn joblib threadpoolctl`
        2. Or try: `conda update scikit-learn joblib`
        
        **To generate feature importances:**
        1. Run `sepsis_prediction_models.ipynb`
        2. Ensure models are saved to `models/` directory
        3. Refresh this page
        
        **Alternative approach:**
        - The notebook can save feature importance data directly to `assets/feature_importance_data.csv`
        - This bypasses model loading issues
        """)
    
    # Fallback - show empty state
    model_features = []



# Display features in order of importance
st.subheader("📋 Top Features Ranked by Model Importance")

# Create two columns for better layout
col1, col2 = st.columns(2)

for idx, feature in enumerate(model_features):
    # Alternate between columns
    current_col = col1 if idx % 2 == 0 else col2
    
    with current_col:
        # Color-coded importance indicator
        importance_emoji = feature['importance'].split()[0]  # Get the emoji
        importance_text = feature['importance'].split()[1]   # Get High/Medium/Low
        
        with st.expander(f"{importance_emoji} **{feature['name']}** ({feature['feature']})"):
            st.markdown(f"""
            **Model Importance**: {feature['importance']}
            
            **Clinical Significance**: 
            {feature['clinical_significance']}
            """)

st.markdown("---")
