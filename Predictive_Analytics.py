import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'THRESHOLD_SEPSIS': 0.55,      # Above this = Sepsis
    'THRESHOLD_NO_SEPSIS': 0.45,   # Below this = No Sepsis  
    'MODEL_PATH': './models/random_forest.pkl',  # Updated to use random forest model
    'LOGO_PATH': './assets/project-logo.jpg',
    'CONSENSUS_STRONG': 80,
    'CONSENSUS_MODERATE': 60,
    'TOP_FEATURES_COUNT': 10
}

# Global variable for sample patients
SAMPLE_PATIENTS = None

# Feature columns and categories
FEATURE_COLUMNS = [
    'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS'
]

FEATURE_CATEGORIES = {
    "Stay Duration": ['HospAdmTime', 'ICULOS'],
    "Vital Signs": ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2'],
    "Blood Gas": ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2'],
    "Organ Function": ['AST', 'BUN', 'Alkalinephos', 'Creatinine'],
    "Metabolic": ['Calcium', 'Chloride', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium'],
    "Hematology": ['Bilirubin_direct', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'],
    "Demographics": ['Age', 'Gender', 'Hour']  # Moved Hour to Demographics as it's a temporal marker
}

# Normal ranges for features
NORMAL_RANGES = {
    'HR': (60, 100), 'O2Sat': (95, 100), 'Temp': (36.1, 37.2), 'SBP': (90, 140),
    'MAP': (70, 100), 'DBP': (60, 90), 'Resp': (12, 20), 'EtCO2': (35, 45),
    'BaseExcess': (-2, 2), 'HCO3': (22, 26), 'FiO2': (21, 100), 'pH': (7.35, 7.45),
    'PaCO2': (35, 45), 'SaO2': (95, 100), 'AST': (10, 40), 'BUN': (7, 20),
    'Alkalinephos': (44, 147), 'Calcium': (8.5, 10.5), 'Chloride': (96, 106),
    'Creatinine': (0.6, 1.3), 'Bilirubin_direct': (0, 0.3), 'Glucose': (70, 140),
    'Lactate': (0.5, 2.2), 'Magnesium': (1.7, 2.2), 'Phosphate': (2.5, 4.5),
    'Potassium': (3.5, 5.0), 'Bilirubin_total': (0.2, 1.2), 'TroponinI': (0, 0.04),
    'Hct': (38, 52), 'Hgb': (12, 18), 'PTT': (25, 35), 'WBC': (4.0, 11.0),
    'Fibrinogen': (200, 400), 'Platelets': (150, 450), 'Age': (18, 100),
    'Hour': (0, 23), 'HospAdmTime': (0, 240), 'ICULOS': (0, 240)
}

@st.cache_resource
def load_sample_patients():
    """Load and prepare a balanced set of sample patients"""
    try:
        # Load data
        df = pd.read_csv('./data/cleaned_dataset.csv')
        
        # Drop specified columns
        columns_drop = {
            'Unnamed: 0', 'Unit1', 'Unit2'
        }
        df = df.drop(columns=[col for col in columns_drop if col in df.columns])
        
        # Separate sepsis and non-sepsis cases
        sepsis_cases = df[df['SepsisLabel'] == 1]
        non_sepsis_cases = df[df['SepsisLabel'] == 0]
        
        # Sample equal numbers from each class (100 from each)
        n_samples = min(100, len(sepsis_cases))
        balanced_sepsis = sepsis_cases.sample(n=n_samples, random_state=42)
        balanced_non_sepsis = non_sepsis_cases.sample(n=n_samples, random_state=42)
        
        # Combine and shuffle
        balanced_df = pd.concat([balanced_sepsis, balanced_non_sepsis])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_df
        
    except Exception as e:
        st.error(f"Error loading sample patients: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        data = joblib.load(CONFIG['MODEL_PATH'])
        return data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_input_constraints(feature):
    """Get very permissive min/max constraints for input fields"""
    return -100000.0, 100000.0  # Very wide range for all features


def generate_random_patient_data():
    """Select a random real patient from our balanced sample set"""
    global SAMPLE_PATIENTS
    
    # Load sample patients if not already loaded
    if SAMPLE_PATIENTS is None:
        SAMPLE_PATIENTS = load_sample_patients()
        if SAMPLE_PATIENTS is None:
            return None
    
    # Select a random patient
    random_patient = SAMPLE_PATIENTS.sample(n=1).iloc[0]
    
    # Convert to dictionary with ALL features in exact order
    patient_data = {}
    for feature in FEATURE_COLUMNS:
        if feature in random_patient.index:
            patient_data[feature] = random_patient[feature]
        else:
            # Use default value for missing features
            if feature == 'Gender':
                patient_data[feature] = 0  # Default to Female
            elif feature == 'Hour':
                patient_data[feature] = 0  # Default to start of day
            elif feature in NORMAL_RANGES:
                min_val, max_val = NORMAL_RANGES[feature]
                patient_data[feature] = (min_val + max_val) / 2  # Use middle of normal range
            else:
                patient_data[feature] = 0  # Default to 0 for unknown features
    
    return patient_data

def get_individual_tree_predictions(model, X):
    """Get predictions from each tree in the random forest"""
    tree_predictions = []
    tree_scores = []
    
    # Get predictions from each tree in the forest
    for tree in model.estimators_:
        pred = tree.predict(X)[0]
        prob = tree.predict_proba(X)[0]
        tree_predictions.append(int(pred))
        tree_scores.append(prob[1])  # Probability of class 1 (Sepsis)
    
    return np.array(tree_predictions), np.array(tree_scores)

def create_advisory_board_visualization(tree_predictions, tree_scores):
    """Create visualization of tree predictions"""
    threshold_sepsis = CONFIG['THRESHOLD_SEPSIS']
    threshold_no_sepsis = CONFIG['THRESHOLD_NO_SEPSIS']
    
    # Separate predictions into three groups
    sepsis_indices = []
    no_sepsis_indices = []
    uncertain_indices = []
    
    for i, score in enumerate(tree_scores):
        if score >= threshold_sepsis:
            sepsis_indices.append(i)
        elif score <= threshold_no_sepsis:
            no_sepsis_indices.append(i)
        else:
            uncertain_indices.append(i)
    
    sepsis_count = len(sepsis_indices)
    no_sepsis_count = len(no_sepsis_indices)
    uncertain_count = len(uncertain_indices)
    
    fig = go.Figure()
    
    # Calculate positions for each group
    def create_column_positions(indices, col_x, spacing=2.0):
        if not indices:
            return [], []
        n = len(indices)
        x = []
        y = []
        for i in range(n):
            x.append(col_x + (i % 5) * 2.0)  # 5 doctors per row
            y.append(-(i // 5) * spacing)  # New row every 5 doctors
        return x, y
    
    # Create positions for each group with extreme spacing
    x_no_sepsis, y_no_sepsis = create_column_positions(no_sepsis_indices, col_x=0)
    x_uncertain, y_uncertain = create_column_positions(uncertain_indices, col_x=20)
    x_sepsis, y_sepsis = create_column_positions(sepsis_indices, col_x=40)
    
    # Add background circles for no sepsis predictions
    if no_sepsis_indices:
        # Add background circles
        fig.add_trace(go.Scatter(
            x=x_no_sepsis,
            y=y_no_sepsis,
            mode='markers',
            marker=dict(
                size=45,
                color='rgba(76, 175, 80, 0.2)',  # Light green background
                line=dict(color='#4caf50', width=2)  # Green border
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        # Add doctor emojis
        fig.add_trace(go.Scatter(
            x=x_no_sepsis,
            y=y_no_sepsis,
            mode='text',
            text=["👨‍⚕️"] * len(no_sepsis_indices),
            textposition="middle center",
            textfont=dict(size=30),
            hovertext=[f"Doctor {i+1}<br>Prediction: No Sepsis<br>Confidence: {tree_scores[i]:.3f}" 
                      for i in no_sepsis_indices],
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add background circles for uncertain predictions
    if uncertain_indices:
        # Add background circles
        fig.add_trace(go.Scatter(
            x=x_uncertain,
            y=y_uncertain,
            mode='markers',
            marker=dict(
                size=45,
                color='rgba(255, 152, 0, 0.2)',  # Light orange background
                line=dict(color='#ff9800', width=2)  # Orange border
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        # Add doctor emojis
        fig.add_trace(go.Scatter(
            x=x_uncertain,
            y=y_uncertain,
            mode='text',
            text=["👨‍⚕️"] * len(uncertain_indices),
            textposition="middle center",
            textfont=dict(size=30),
            hovertext=[f"Doctor {i+1}<br>Prediction: Uncertain<br>Confidence: {tree_scores[i]:.3f}" 
                      for i in uncertain_indices],
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add background circles for sepsis predictions
    if sepsis_indices:
        # Add background circles
        fig.add_trace(go.Scatter(
            x=x_sepsis,
            y=y_sepsis,
            mode='markers',
            marker=dict(
                size=45,
                color='rgba(244, 67, 54, 0.2)',  # Light red background
                line=dict(color='#f44336', width=2)  # Red border
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        # Add doctor emojis
        fig.add_trace(go.Scatter(
            x=x_sepsis,
            y=y_sepsis,
            mode='text',
            text=["👨‍⚕️"] * len(sepsis_indices),
            textposition="middle center",
            textfont=dict(size=30),
            hovertext=[f"Doctor {i+1}<br>Prediction: Sepsis<br>Confidence: {tree_scores[i]:.3f}" 
                      for i in sepsis_indices],
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add column labels with much more space from doctors
    fig.add_annotation(x=4, y=2.0,
                      text="No Sepsis Prediction", 
                      showarrow=False,
                      font=dict(size=16, color="#4caf50"),
                      align="center")
    fig.add_annotation(x=24, y=2.0,
                      text="Uncertain", 
                      showarrow=False,
                      font=dict(size=16, color="#ff9800"),
                      align="center")
    fig.add_annotation(x=44, y=2.0,
                      text="Sepsis Prediction", 
                      showarrow=False,
                      font=dict(size=16, color="#f44336"),
                      align="center")
    
    # Calculate maximum number of rows needed
    max_rows = max(
        (len(sepsis_indices) + 4) // 5,  # 5 doctors per row
        (len(no_sepsis_indices) + 4) // 5,
        (len(uncertain_indices) + 4) // 5
    ) if any([sepsis_indices, no_sepsis_indices, uncertain_indices]) else 1
    
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-2, 50]  # Much wider range for extreme spacing
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-(max_rows * 2.0), 3.0],  # Much more vertical space
            scaleanchor='x',
            scaleratio=1
        ),
        plot_bgcolor='white',
        height=600,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig, sepsis_count, no_sepsis_count, uncertain_count

def predictive_analytics():
    st.info("This page allows you to input patient data and receive sepsis risk predictions from a pre-trained Random Forest model.")
    st.markdown("---")
    
    # Create directory for assets if it doesn't exist
    os.makedirs("assets/priority", exist_ok=True)
    
    # Use cached data from the main dashboard
    if 'data' in st.session_state:
        # Use cached balanced sample
        if st.session_state.data['balanced_sample'] is not None:
            SAMPLE_PATIENTS = st.session_state.data['balanced_sample']
        else:
            # If no cached balanced sample, try to use the main dataframe
            if st.session_state.data['df'] is not None:
                df = st.session_state.data['df']
                # Create a balanced sample on the fly
                sepsis_cases = df[df['SepsisLabel'] == 1]
                non_sepsis_cases = df[df['SepsisLabel'] == 0]
                n_samples = min(100, len(sepsis_cases))
                balanced_sepsis = sepsis_cases.sample(n=n_samples, random_state=42)
                balanced_non_sepsis = non_sepsis_cases.sample(n=n_samples, random_state=42)
                SAMPLE_PATIENTS = pd.concat([balanced_sepsis, balanced_non_sepsis]).sample(frac=1, random_state=42)
            else:
                # Fallback to loading directly
                SAMPLE_PATIENTS = load_sample_patients()
        
        # Use cached model
        data = st.session_state.data['model_data']
    else:
        # Fallback to loading directly if not in session state
        SAMPLE_PATIENTS = load_sample_patients()
        data = load_model()
    
    if SAMPLE_PATIENTS is None:
        st.error("Could not load sample patients. Please check the data file.")
        return
    
    if data is None:
        st.warning("Model file not found. Using a demo mode with simulated predictions.")
        # Create a simple demo model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(np.random.rand(100, len(FEATURE_COLUMNS)), np.random.randint(0, 2, 100))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.rand(100, len(FEATURE_COLUMNS)))
        model_features = FEATURE_COLUMNS
    else:
        # Extract model from loaded data
        model = data.get('model')
        scaler = data.get('scaler')
        model_features = data.get('features', FEATURE_COLUMNS)  # Get the exact features from the model
    
    st.markdown('<h3 class="sub-header">📋 Patient Information Input</h3>', unsafe_allow_html=True)
    
    # Add refresh and reset buttons
    col_refresh, col_info, col_reset = st.columns([1, 2, 1])
    
    with col_refresh:
        if st.button("🎲 Select Random Patient for demonstration purposes", type="secondary", 
                    help="Select a random real patient from our dataset",
                    disabled=st.session_state.get('inputs_locked', False)):
            st.session_state.refresh_values = True
            st.session_state.inputs_locked = True
    
    with col_info:
        # Initialize the show_status in session state if not present
        if 'show_status' not in st.session_state:
            st.session_state.show_status = False
            
        # Set show_status to True when random patient is selected
        if st.session_state.get('refresh_values', False) and st.session_state.get('inputs_locked', False):
            st.session_state.show_status = True
            
            # Store current patient data in session state
            try:
                random_patient = SAMPLE_PATIENTS.sample(n=1).iloc[0]
                st.session_state.current_patient = {
                    'id': random_patient.name,
                    'hour': random_patient['Hour'],
                    'status': random_patient['SepsisLabel'],
                    'patient_id': random_patient['Patient_ID']
                }
            except Exception as e:
                st.error("Error selecting random patient")
                
        # Show status if it's enabled
        if st.session_state.get('show_status', False) and hasattr(SAMPLE_PATIENTS, 'iloc'):
            try:
                if hasattr(st.session_state, 'current_patient'):
                    current_patient = st.session_state.current_patient
                    
                    # Get all records for this patient from the full dataset
                    if 'data' in st.session_state and st.session_state.data['df'] is not None:
                        full_df = st.session_state.data['df']
                    else:
                        full_df = pd.read_csv("./data/cleaned_dataset.csv")
                    
                    # Get patient ID and all their records
                    patient_id = current_patient['patient_id']
                    all_patient_records = full_df[full_df['Patient_ID'] == patient_id].sort_values('Hour')
                    
                    # Get current hour status
                    current_hour_status = current_patient['status']
                    current_hour = current_patient['hour']
                    
                    # Get future records and check if sepsis develops later
                    future_records = all_patient_records[all_patient_records['Hour'] > current_hour]
                    will_develop_sepsis = (future_records['SepsisLabel'] == 1).any() if not future_records.empty else False
                    
                    # Get past records and check if had sepsis before
                    past_records = all_patient_records[all_patient_records['Hour'] < current_hour]
                    had_sepsis_before = (past_records['SepsisLabel'] == 1).any() if not past_records.empty else False
                    
                    # Display status header
                    st.markdown("### Patient Sepsis Status")
                    
                    if current_hour_status == 1:
                        # Currently has sepsis
                        st.error(f"🚨 Patient currently HAS SEPSIS at hour {current_hour}")
                    elif will_develop_sepsis:
                        # Currently no sepsis but will develop later
                        next_sepsis_hour = future_records[future_records['SepsisLabel'] == 1]['Hour'].min()
                        st.info(f"⚠️ Patient does NOT have sepsis at current hour {current_hour}")
                        st.warning(f"⚠️ Alert: Patient will develop sepsis at hour {next_sepsis_hour}")
                    elif had_sepsis_before:
                        # Had sepsis before but not now and not in future
                        st.info(f"ℹ️ Patient does NOT have sepsis at current hour {current_hour} (had sepsis in earlier hours)")
                    else:
                        # Never had/has/will have sepsis
                        st.success(f"✅ Patient does NOT develop sepsis at any time (current hour: {current_hour})")
            except Exception as e:
                st.error(f"Error displaying patient status: {str(e)}")
    
    with col_reset:
        if st.button("🔄 Reset", type="secondary",
                    help="Reset fields and enable editing",
                    disabled=not st.session_state.get('inputs_locked', False)):
            # Clear all session state flags and values
            st.session_state.inputs_locked = False
            st.session_state.refresh_values = False
            st.session_state.show_status = False  # Explicitly set show_status to False
            
            # Clear all feature values
            for feature in FEATURE_COLUMNS:
                if feature in st.session_state:
                    del st.session_state[feature]
            
            # Clear any stored patient data
            if 'current_patient' in st.session_state:
                del st.session_state.current_patient
            
            st.rerun()
    
    # Generate random data if refresh button was pressed
    if st.session_state.get('refresh_values', False):
        random_data = generate_random_patient_data()
        if random_data is not None:
            st.session_state.update(random_data)
        st.session_state.refresh_values = False
        st.rerun()
    
    input_data = {}
    
    # Define category icons
    category_icons = {
        "Stay Duration": "⌛",
        "Vital Signs": "❤️",
        "Blood Gas": "💨",
        "Organ Function": "🫁",
        "Metabolic": "⚗️",
        "Hematology": "🩸",
        "Demographics": "👤"
    }
    
    # Add custom CSS for expander styling
    st.markdown("""
        <style>
        /* Target the expander header */
        div[data-testid="stExpander"] div[role="button"] {
            background: linear-gradient(to right, #f0f2f6, #e0e5ec) !important;
            border-radius: 8px !important;
            padding: 15px 20px !important;
            border: none !important;
            box-shadow: 3px 3px 6px #b8b9be, -3px -3px 6px #ffffff !important;
            transition: all 0.3s ease !important;
        }
        
        /* Hover effect */
        div[data-testid="stExpander"] div[role="button"]:hover {
            background: linear-gradient(to right, #e0e5ec, #d1d6dd) !important;
            box-shadow: 2px 2px 4px #b8b9be, -2px -2px 4px #ffffff !important;
        }
        
        /* Style the text inside expander */
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            color: #2E86AB !important;
            text-shadow: 1px 1px 1px rgba(255,255,255,0.5) !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
        }
        
        /* Make emojis larger */
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.2rem !important;
        }
        
        /* Active state */
        div[data-testid="stExpander"] div[role="button"][aria-expanded="true"] {
            background: linear-gradient(to right, #e0e5ec, #d1d6dd) !important;
            box-shadow: inset 2px 2px 5px #b8b9be, inset -2px -2px 5px #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create expanders for each category
    for category, features in FEATURE_CATEGORIES.items():
        icon = category_icons.get(category, "📋")
        
        # Special case for Stay Duration to avoid emoji duplication
        display_category = category
        display_icon = icon
        
        with st.expander(f"{display_icon}  {display_category}", expanded=True):

            
            # Create a grid layout for features
            cols = st.columns(3)
            
            for j, feature in enumerate(features):
                col_idx = j % 3
                
                with cols[col_idx]:
     
                    # st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
                    # Use session state value if available, otherwise use default
                    session_key = f"input_{feature}"
                    
                    if feature == 'Gender':
                        default_gender = int(st.session_state.get(feature, 0))  # Ensure integer
                        value = st.selectbox(
                            f"{feature}", 
                            options=[0, 1], 
                            index=default_gender if default_gender in [0, 1] else 0,
                            format_func=lambda x: "Female" if x == 0 else "Male",
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    elif feature == 'Hour':
                        default_hour = int(st.session_state.get(feature, 0))  # Ensure integer
                        value = st.selectbox(
                            f"{feature}", 
                            options=list(range(24)), 
                            index=default_hour if 0 <= default_hour < 24 else 0,
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    elif feature == 'ICULOS':
                        default_iculos = int(st.session_state.get(feature, 0))  # Ensure integer
                        value = st.number_input(
                            f"{feature}", 
                            min_value=0,
                            max_value=1000,
                            value=default_iculos if 0 <= default_iculos <= 1000 else 0,
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    else:
                        # Get appropriate constraints for this feature
                        min_val, max_val = get_input_constraints(feature)
                        
                        if feature in NORMAL_RANGES:
                            normal_min, normal_max = NORMAL_RANGES[feature]
                            default_val = (normal_min + normal_max) / 2
                        else:
                            default_val = min_val
                        
                        # Use session state value if available, otherwise use default
                        current_val = st.session_state.get(feature, default_val)
                        
                        # Ensure current value is within constraints
                        current_val = np.clip(float(current_val), min_val, max_val)
                        
                        # Determine appropriate step size
                        if feature in ['HR', 'Age', 'Resp', 'FiO2', 'MAP', 'WBC', 'Platelets']:
                            step = 1.0  # Integer values
                        elif max_val > 1000:
                            step = 1.0  # Large ranges use integer steps
                        else:
                            step = 0.1  # Decimal values
                        
                        value = st.number_input(
                            f"{feature}", 
                            min_value=float(min_val), 
                            max_value=float(max_val),
                            value=float(current_val),
                            step=step,
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    
                    input_data[feature] = value
                    st.markdown('</div>', unsafe_allow_html=True)  # Close feature-card div
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close feature-category div
    
    if st.button("🔍 Get Sepsis Prediction", type="primary", use_container_width=True):
        # Create DataFrame with features in exact order
        X_input = pd.DataFrame([input_data])[FEATURE_COLUMNS]
        X_input = X_input.fillna(0)  # Fill any missing values with 0
        
        # Scale the input data using the loaded scaler
        X_input_scaled = scaler.transform(X_input)
        
        # Get predictions from all trees
        tree_preds, tree_scores = get_individual_tree_predictions(model, X_input_scaled)
        
        # Calculate mean probability
        mean_prob = np.mean(tree_scores)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🏛️ Random Forest Advisory Board Decision</h3>', unsafe_allow_html=True)
        
        # Add distribution plot of tree predictions
        fig_dist = go.Figure()
        
        # Create scatter plot of tree predictions
        fig_dist.add_trace(go.Scatter(
            x=tree_scores,
            y=np.zeros_like(tree_scores),  # All points on same y-level
            mode='markers',
            marker=dict(
                size=15,
                color=['#f44336' if s >= CONFIG['THRESHOLD_SEPSIS'] else  # Red for sepsis
                       '#4caf50' if s <= CONFIG['THRESHOLD_NO_SEPSIS'] else  # Green for no sepsis
                       '#ff9800' for s in tree_scores],  # Orange for uncertain
                line=dict(width=1, color='white')
            ),
            text=[f"Tree {i+1}<br>Score: {score:.3f}" for i, score in enumerate(tree_scores)],
            hovertemplate='%{text}<extra></extra>',
            name='Tree Predictions'
        ))
        
        # Add threshold lines
        fig_dist.add_vline(x=CONFIG['THRESHOLD_SEPSIS'], 
                          line_dash="dash", line_color="red",
                          annotation=dict(
                              text="Sepsis Threshold",
                              font=dict(color="red")
                          ))
        fig_dist.add_vline(x=CONFIG['THRESHOLD_NO_SEPSIS'], 
                          line_dash="dash", line_color="green",
                          annotation=dict(
                              text="No Sepsis Threshold",
                              font=dict(color="green")
                          ))
        
        # Update layout
        fig_dist.update_layout(
            xaxis=dict(
                title="Prediction Score",
                range=[0, 1],  # Fix range from 0 to 1
                tickformat='.1f',  # Show as decimal
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            showlegend=False,
            plot_bgcolor='white',
            height=200,
            margin=dict(l=50, r=50, t=20, b=50)  # Reduced top margin since no title
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Calculate votes and consensus first
        sepsis_votes = sum(s >= CONFIG['THRESHOLD_SEPSIS'] for s in tree_scores)
        no_sepsis_votes = sum(s <= CONFIG['THRESHOLD_NO_SEPSIS'] for s in tree_scores)
        uncertain_votes = sum(CONFIG['THRESHOLD_NO_SEPSIS'] < s < CONFIG['THRESHOLD_SEPSIS'] for s in tree_scores)
        
        # Calculate consensus percentage
        consensus_pct = max(sepsis_votes, no_sepsis_votes) / len(tree_scores) * 100 if len(tree_scores) > 0 else 0

        # Show summary statistics in Streamlit
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean Score", f"{np.mean(tree_scores):.3f}")
        with col2:
            st.markdown(f"""
            <div style='text-align: center'>
                <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Trees for Sepsis</p>
                <p style='margin: 0; color: #f44336; font-size: 2rem; font-weight: 600'>{sepsis_votes}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='text-align: center'>
                <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Trees for No Sepsis</p>
                <p style='margin: 0; color: #4caf50; font-size: 2rem; font-weight: 600'>{no_sepsis_votes}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style='text-align: center'>
                <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Uncertain Trees</p>
                <p style='margin: 0; color: #ff9800; font-size: 2rem; font-weight: 600'>{uncertain_votes}</p>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div style='text-align: center'>
                <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Consensus</p>
                <p style='margin: 0; font-size: 2rem; font-weight: 600'>{consensus_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Create and show advisory board visualization
        fig, _, _, _ = create_advisory_board_visualization(tree_scores, tree_scores)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clinical Recommendation")
        
        if mean_prob >= CONFIG['THRESHOLD_SEPSIS']:
            if consensus_pct >= 70:
                st.error("🚨 **HIGH PRIORITY**: Strong consensus for sepsis risk. Immediate clinical evaluation and intervention recommended.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    # Load the image from the assets/priority folder
                    try:
                        st.image("assets/priority/5.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
            else:
                st.warning("⚠️ **MODERATE PRIORITY**: Sepsis risk detected with some uncertainty. Close monitoring and clinical assessment advised.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/4.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
        elif mean_prob <= CONFIG['THRESHOLD_NO_SEPSIS']:
            if consensus_pct >= 70:
                st.success("✅ **LOW PRIORITY**: Strong consensus for low sepsis risk. Continue routine monitoring per protocol.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/1.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
            else:
                st.info("ℹ️ **ROUTINE MONITORING**: Low sepsis risk indicated but maintain standard care vigilance.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/2.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
        else:
            st.warning(f"🤔 **UNCERTAIN PREDICTION**: Model is uncertain (Sepsis: {mean_prob:.1%}). Consider additional clinical assessment, laboratory tests, and expert consultation. Monitor closely for clinical deterioration.")
            col1, col2, col3 = st.columns([2,3,2])
            with col2:
                try:
                    st.image("assets/priority/3.png", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load priority image: {e}")