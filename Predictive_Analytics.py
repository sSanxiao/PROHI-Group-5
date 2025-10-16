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
    'THRESHOLD_SEPSIS': 0.44,      # Above this = Sepsis (from temporal_sepsis_model.ipynb)
    'THRESHOLD_NO_SEPSIS': 0.34,   # Below this = No Sepsis (adjusted to maintain 0.1 gap)  
    'RF_MODEL_PATH': './models/random_forest.pkl',  # Random Forest model
    'GRU_MODEL_PATH': './models/gru_temporal_best.pth',  # GRU model from temporal_sepsis_model.ipynb
    'LOGO_PATH': './assets/project-logo.jpg',
    'CONSENSUS_STRONG': 80,
    'CONSENSUS_MODERATE': 60,
    'TOP_FEATURES_COUNT': 10,
    'MAX_SEQUENCE_LENGTH': 24  # Maximum number of hours to show for GRU input
}

# Global variable for sample patients
SAMPLE_PATIENTS = None

# Feature columns and categories - limited to the specified features
FEATURE_COLUMNS = [
    'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
    'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender'
]

FEATURE_CATEGORIES = {
    "Stay Duration": ['ICULOS'],
    "Vital Signs": ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp'],
    "Blood Gas": ['HCO3', 'pH', 'PaCO2'],
    "Organ Function": ['Creatinine'],
    "Hematology": ['Bilirubin_direct', 'WBC', 'Platelets'],
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
def load_rf_model():
    """Load the Random Forest model"""
    try:
        data = joblib.load(CONFIG['RF_MODEL_PATH'])
        return data
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")
        return None

@st.cache_resource
def load_gru_model():
    """Load the GRU model"""
    try:
        import torch
        import torch.nn as nn
        import sys
        import joblib
        
        # Define the GRU model class from the notebook
        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
                super(GRUModel, self).__init__()
                
                # GRU layers
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True
                )
                
                # Attention mechanism
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
                
                # Output layers
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x, lengths=None):
                # x shape: (batch_size, seq_len, input_size)
                
                # If lengths are provided, use packed sequence
                if lengths is not None:
                    # Pack padded sequence for more efficient computation
                    packed_input = nn.utils.rnn.pack_padded_sequence(
                        x, lengths.cpu(), batch_first=True, enforce_sorted=False
                    )
                    
                    # Apply GRU
                    packed_output, _ = self.gru(packed_input)
                    
                    # Unpack the sequence
                    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                else:
                    # If no lengths provided, just run the GRU directly
                    output, _ = self.gru(x)
                
                # output shape: (batch_size, seq_len, hidden_size * 2)
                
                # Apply attention
                batch_size, seq_len, hidden_size = output.size()
                
                # Calculate attention scores
                attention_scores = self.attention(output).squeeze(-1)
                # attention_scores shape: (batch_size, seq_len)
                
                if lengths is not None:
                    # Create mask for padding
                    mask = torch.zeros_like(attention_scores)
                    for i, length in enumerate(lengths):
                        mask[i, :length] = 1
                    
                    # Apply mask (set padding attention scores to -inf)
                    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
                
                # Apply softmax to get attention weights
                attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
                # attention_weights shape: (batch_size, seq_len, 1)
                
                # Apply attention weights to get context vector
                context = torch.sum(output * attention_weights, dim=1)
                # context shape: (batch_size, hidden_size * 2)
                
                # Apply output layers
                output = self.fc(context)
                # output shape: (batch_size, 1)
                
                return torch.sigmoid(output.squeeze(-1))
        
        # Create a wrapper class similar to GRUSequenceModel
        class GRUSequenceWrapper:
            def __init__(self, model, features, scaler=None):
                self.model = model
                self.features = features
                self.scaler = scaler
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            def predict_sequence(self, X):
                """Predict on a sequence"""
                self.model.eval()
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
                
                # Move to device
                X_tensor = X_tensor.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    output = self.model(X_tensor)
                
                # Convert to numpy
                predictions = (output >= 0.5).cpu().numpy()
                probabilities = output.cpu().numpy()
                
                return predictions, probabilities
        
        # Set the path to the temporal model
        CONFIG['GRU_MODEL_PATH'] = './models/gru_temporal_best.pth'
        
        # Try to load metadata first
        try:
            metadata = joblib.load('./models/gru_temporal_best_metadata.pkl')
            input_size = metadata.get('input_size', len(FEATURE_COLUMNS))
            hidden_size = metadata.get('hidden_size', 128)
            num_layers = metadata.get('num_layers', 2)
            dropout = metadata.get('dropout', 0.3)
            features = metadata.get('features', FEATURE_COLUMNS)
            print(f"Loaded model metadata with input_size={input_size}")
        except Exception as e:
            # If metadata not available, use default values
            print(f"Metadata not available: {e}. Using default values.")
            input_size = len(FEATURE_COLUMNS)
            hidden_size = 128
            num_layers = 2
            dropout = 0.3
            features = FEATURE_COLUMNS
        
        # Create the model
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Load the model weights
        try:
            # Try to load the model state dict
            checkpoint = torch.load(CONFIG['GRU_MODEL_PATH'], map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
            model.eval()  # Set to evaluation mode
            
            # Create the wrapper
            wrapper = GRUSequenceWrapper(
                model=model,
                features=features
            )
            
            print(f"Successfully loaded model from {CONFIG['GRU_MODEL_PATH']}")
            return wrapper
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            return None
    except Exception as e:
        st.error(f"Error loading GRU model: {e}")
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
    st.info("This page allows you to input patient data and receive sepsis risk predictions from pre-trained machine learning models.")
    st.markdown("---")
    
    # Create directory for assets if it doesn't exist
    os.makedirs("assets/priority", exist_ok=True)
    
    # Model selection
    model_type = st.radio(
        "Select prediction model:",
        ["Random Forest", "GRU (Temporal)"],
        horizontal=True,
        help="Random Forest uses single timepoint data. GRU uses sequence data over time."
    )
    
    # Initialize session state for sequence data if not exists
    if 'sequence_data' not in st.session_state:
        st.session_state.sequence_data = []
    
    # Use cached data from the main dashboard
    if 'data' in st.session_state:
        # Use cached balanced sample
        if st.session_state.data['balanced_sample'] is not None:
            global SAMPLE_PATIENTS
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
    else:
        # Fallback to loading directly if not in session state
        SAMPLE_PATIENTS = load_sample_patients()
    
        if SAMPLE_PATIENTS is None:
            st.error("Could not load sample patients. Please check the data file.")
            return
    
    # Load appropriate model based on selection
    if model_type == "Random Forest":
        # Load Random Forest model
        if 'data' in st.session_state and st.session_state.data.get('model_data') is not None:
            data = st.session_state.data['model_data']
        else:
            data = load_rf_model()
        model = data.get('model')
        scaler = data.get('scaler')
        model_features = data.get('features', FEATURE_COLUMNS)  # Get the exact features from the model
    else:
        # Load GRU model
        gru_model = load_gru_model()
        
        if gru_model is None:
            st.warning("GRU model file not found. Using a demo mode with simulated predictions.")
            # Create a simple demo model using Random Forest as fallback
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(np.random.rand(100, len(FEATURE_COLUMNS)), np.random.randint(0, 2, 100))
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(np.random.rand(100, len(FEATURE_COLUMNS)))
            model_features = FEATURE_COLUMNS
        else:
            # Use the GRU model
            model = gru_model
            scaler = model.scaler
            model_features = model.features
            
    # Different input sections based on model type
    if model_type == "Random Forest":
        st.markdown('<h3 class="sub-header">📋 Single Timepoint Patient Information</h3>', unsafe_allow_html=True)
    
        # Add refresh and reset buttons
        col_refresh, col_info, col_reset = st.columns([1, 2, 1])
        
        with col_refresh:
            if st.button("🎲 Select Random Patient", type="secondary", 
                        help="Select a random real patient from our dataset",
                        disabled=st.session_state.get('inputs_locked', False)):
                st.session_state.refresh_values = True
                st.session_state.inputs_locked = True
                # Clear sequence data when switching to random forest
                st.session_state.sequence_data = []
        
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
    else:
        # GRU model input - temporal sequence
        st.markdown('<h3 class="sub-header">📋 Temporal Sequence Patient Information</h3>', unsafe_allow_html=True)
        
        # Explanation of GRU input
        st.info("""
        The GRU model requires a sequence of measurements over time to make predictions. 
        You can add multiple timepoints (hours) for a patient, and the model will analyze 
        the temporal patterns to predict sepsis risk.
        """)
        
        # Controls for sequence data
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("🎲 Select Random Patient Sequence", type="secondary", 
                        help="Select a random patient sequence from our dataset"):
                # Get a random patient
                try:
                    if 'data' in st.session_state and st.session_state.data['df'] is not None:
                        full_df = st.session_state.data['df']
                    else:
                        full_df = pd.read_csv("./data/cleaned_dataset.csv")
                    
                    # Get random patient ID
                    random_patient_id = np.random.choice(full_df['Patient_ID'].unique())
                    
                    # Get all records for this patient
                    patient_records = full_df[full_df['Patient_ID'] == random_patient_id].sort_values('Hour')
                    
                    # Filter to only include the features we want
                    patient_records = patient_records[FEATURE_COLUMNS + ['Patient_ID', 'SepsisLabel']]
                    
                    # Limit to max sequence length if needed
                    max_seq = CONFIG['MAX_SEQUENCE_LENGTH']
                    if len(patient_records) > max_seq:
                        patient_records = patient_records.iloc[:max_seq]
                    
                    # Store in session state
                    st.session_state.sequence_data = patient_records.to_dict('records')
                    st.rerun()
                except Exception as e:
                    st.error(f"Error selecting random patient sequence: {str(e)}")
        
        with col3:
            if st.button("🔄 Clear Sequence", type="secondary",
                        help="Clear the current sequence data"):
                st.session_state.sequence_data = []
                st.rerun()
                
        # Display current sequence
        if st.session_state.sequence_data:
            # Convert to DataFrame for display
            seq_df = pd.DataFrame(st.session_state.sequence_data)
            
            # Show sequence length and patient ID
            patient_id = seq_df['Patient_ID'].iloc[0] if 'Patient_ID' in seq_df.columns else "Unknown"
            
            # Show sepsis status prominently
            col_id, col_status = st.columns([1, 2])
            
            with col_id:
                st.markdown("### Current Patient Sequence")
                st.markdown(f"**Patient ID:** {patient_id} | **Sequence Length:** {len(seq_df)} hours")
            
            with col_status:
                if 'SepsisLabel' in seq_df.columns:
                    has_sepsis = seq_df['SepsisLabel'].max() == 1
                    if has_sepsis:
                        first_sepsis_hour = seq_df[seq_df['SepsisLabel'] == 1]['Hour'].min()
                        st.warning(f"⚠️ This patient develops sepsis at hour {first_sepsis_hour}")
                    else:
                        st.success("✅ This patient does not develop sepsis in this sequence")
            
            # Display the sequence data in a table
            st.dataframe(
                seq_df[FEATURE_COLUMNS],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No sequence data available. Please select a random patient sequence or add timepoints manually.")
            
            # Option to manually add a timepoint
            st.markdown("### Add a Timepoint Manually")
            
            # Create a form for adding a new timepoint
            with st.form("add_timepoint_form"):
                st.markdown("Enter values for a new timepoint:")
                
                # Create columns for the form fields
                form_cols = st.columns(3)
                
                # Initialize input data dictionary
                new_timepoint = {}
                
                # Add Hour field first
                new_timepoint['Hour'] = form_cols[0].number_input(
                    "Hour",
                    min_value=0,
                    max_value=240,
                    value=len(st.session_state.sequence_data),  # Next hour in sequence
                    step=1
                )
                
                # Add other fields
                col_idx = 1
                for feature in [f for f in FEATURE_COLUMNS if f != 'Hour']:
                    # Get appropriate constraints for this feature
                    min_val, max_val = get_input_constraints(feature)
                    
                    if feature in NORMAL_RANGES:
                        normal_min, normal_max = NORMAL_RANGES[feature]
                        default_val = (normal_min + normal_max) / 2
                    else:
                        default_val = min_val
                    
                    # Determine appropriate step size
                    if feature in ['HR', 'Age', 'Resp', 'FiO2', 'MAP', 'WBC', 'Platelets']:
                        step = 1.0  # Integer values
                    elif max_val > 1000:
                        step = 1.0  # Large ranges use integer steps
                    else:
                        step = 0.1  # Decimal values
                    
                    # Special handling for Gender
                    if feature == 'Gender':
                        new_timepoint[feature] = form_cols[col_idx].selectbox(
                            feature,
                            options=[0, 1],
                            format_func=lambda x: "Female" if x == 0 else "Male"
                        )
                    else:
                        new_timepoint[feature] = form_cols[col_idx].number_input(
                            feature,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default_val),
                            step=step
                        )
                    
                    # Update column index
                    col_idx = (col_idx + 1) % 3
                
                # Add submit button
                submit_button = st.form_submit_button("Add Timepoint")
                
                if submit_button:
                    # Add the new timepoint to the sequence
                    st.session_state.sequence_data.append(new_timepoint)
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
    
    # Only show feature input expanders for Random Forest model
    if model_type == "Random Forest":
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
    
    # Add consent checkbox
    consent = st.checkbox(
        "I understand that this prediction is for educational purposes only and may be inaccurate. Clinical decisions should always be made by qualified healthcare professionals.",
        key="consent_checkbox"
    )
    
    # Different prediction button based on model type
    if model_type == "Random Forest":
        prediction_button = st.button("🔍 Get Random Forest Prediction", type="primary", use_container_width=True, disabled=not consent)
    else:
        prediction_button = st.button("🔍 Get GRU Temporal Prediction", type="primary", use_container_width=True, disabled=not consent or not st.session_state.sequence_data)
        if not st.session_state.sequence_data and consent:
            st.warning("Please add at least one timepoint to the sequence before making a prediction.")
    
    if prediction_button:
        if not consent:
            st.warning("Please acknowledge the consent notice before proceeding with the prediction.")
        else:
            # Display a prominent disclaimer
            st.warning("""
            **IMPORTANT DISCLAIMER**: This prediction is based on a machine learning model and is not a substitute for professional medical diagnosis. 
            The model may be inaccurate in individual cases. Always consult with qualified healthcare professionals for clinical decisions.
            """)
            
            # Different prediction logic based on model type
            if model_type == "Random Forest":
                # Create DataFrame with features in exact order
                X_input = pd.DataFrame([input_data])[FEATURE_COLUMNS]
                X_input = X_input.fillna(0)  # Fill any missing values with 0
                
                # Scale the input data using the loaded scaler
                X_input_scaled = scaler.transform(X_input)
                
                # Get predictions from all trees
                tree_preds, tree_scores = get_individual_tree_predictions(model, X_input_scaled)
                
                # Calculate mean probability
                mean_prob = np.mean(tree_scores)
                
                # Set up for visualization
                show_tree_visualization = True
                
            else:  # GRU model
                if not st.session_state.sequence_data:
                    st.error("No sequence data available. Please add timepoints to the sequence.")
                    return
                
                try:
                    # Convert sequence data to DataFrame
                    seq_df = pd.DataFrame(st.session_state.sequence_data)
                    
                    # Make sure we have all required features
                    for feature in FEATURE_COLUMNS:
                        if feature not in seq_df.columns:
                            seq_df[feature] = 0  # Default value
                    
                    # Sort by Hour to ensure temporal order
                    seq_df = seq_df.sort_values('Hour').reset_index(drop=True)
                    
                    # Make sure we only use the features that the GRU model expects
                    # These are the features from the columns_to_keep list in gru_model.py
                    gru_features = [
                        'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
                        'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender'
                    ]
                    
                    # Extract features in the correct order
                    X_sequence = seq_df[gru_features].values
                    
                    # Make prediction using GRU model
                    with st.spinner("Running GRU prediction on temporal sequence..."):
                        try:
                            # Use the model's predict_sequence method which handles all the preprocessing
                            # This method returns predictions and probabilities for each timestep
                            _, probabilities = model.predict_sequence(X_sequence)
                            
                            # Get the prediction for the last timestep
                            mean_prob = probabilities[-1]
                            
                        except Exception as e:
                            st.error(f"Error in GRU prediction: {str(e)}")
                            
                            # Fallback to a simpler approach - simulate a prediction
                            st.warning("Using simulated prediction as fallback")
                            
                            # Generate a probability based on clinical heuristics
                            # These are simplified rules based on sepsis criteria
                            
                            # Extract key vital signs if available
                            hr_values = seq_df['HR'].values if 'HR' in seq_df.columns else np.array([80])
                            temp_values = seq_df['Temp'].values if 'Temp' in seq_df.columns else np.array([37])
                            resp_values = seq_df['Resp'].values if 'Resp' in seq_df.columns else np.array([16])
                            wbc_values = seq_df['WBC'].values if 'WBC' in seq_df.columns else np.array([8])
                            
                            # Look for trends - increasing values may indicate developing sepsis
                            hr_trend = 0 if len(hr_values) < 2 else (hr_values[-1] - hr_values[0]) / max(1, len(hr_values))
                            temp_trend = 0 if len(temp_values) < 2 else (temp_values[-1] - temp_values[0]) / max(1, len(temp_values))
                            
                            # Calculate risk factors based on latest values and trends
                            hr_risk = min(1.0, max(0.0, (hr_values[-1] - 60) / 60))  # HR > 90 is concerning
                            temp_risk = min(1.0, max(0.0, abs(temp_values[-1] - 37) / 2))  # Fever or hypothermia
                            resp_risk = min(1.0, max(0.0, (resp_values[-1] - 12) / 10))  # Tachypnea
                            wbc_risk = min(1.0, max(0.0, abs(wbc_values[-1] - 7.5) / 7.5))  # Abnormal WBC
                            
                            # Add trend factors
                            hr_trend_risk = min(0.5, max(0.0, hr_trend / 10))
                            temp_trend_risk = min(0.5, max(0.0, abs(temp_trend) / 0.5))
                            
                            # Combine all factors with weights
                            mean_prob = min(1.0, max(0.0, 
                                hr_risk * 0.25 + 
                                temp_risk * 0.25 + 
                                resp_risk * 0.15 + 
                                wbc_risk * 0.15 +
                                hr_trend_risk * 0.1 +
                                temp_trend_risk * 0.1))
                            
                            # Add some randomness to simulate model uncertainty
                            mean_prob = min(1.0, max(0.0, mean_prob + np.random.normal(0, 0.05)))
                        
                        # For visualization compatibility
                        # Create dummy tree scores that approximate the GRU confidence
                        n_trees = 100  # Number of dummy trees
                        tree_scores = np.random.normal(mean_prob, 0.1, n_trees)
                        tree_scores = np.clip(tree_scores, 0, 1)  # Ensure values are between 0 and 1
                        
                        # Set up for visualization
                        show_tree_visualization = False
                except Exception as e:
                    st.error(f"Error making GRU prediction: {str(e)}")
                    return
        
        st.markdown("---")
            
        # Different visualization based on model type
        if model_type == "Random Forest":
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
        else:
            # GRU model visualization
            st.markdown('<h3 class="sub-header">🧠 GRU Temporal Prediction</h3>', unsafe_allow_html=True)
            
            # Create a temporal visualization of the sequence and prediction
            if 'sequence_data' in st.session_state and st.session_state.sequence_data:
                seq_df = pd.DataFrame(st.session_state.sequence_data)
                
                # Show sequence statistics
                st.markdown("### Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sequence Length", f"{len(seq_df)} hours")
                with col2:
                    st.metric("Sepsis Risk", f"{mean_prob:.1%}")
                with col3:
                    risk_level = "High" if mean_prob >= 0.64 else "Medium" if mean_prob >= 0.24 else "Low"
                    risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Risk Level</p>
                        <p style='margin: 0; color: {risk_color}; font-size: 2rem; font-weight: 600'>{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create predictions for each subsequence length
                st.markdown("### Prediction Evolution Over Time")
                st.markdown("This chart shows how the prediction changes as more data points are added:")
                
                hours = seq_df['Hour'].values
                predictions_over_time = []
                
                # Try to make predictions with increasing sequence lengths
                try:
                    # Get predictions for the entire sequence using the model's predict_sequence method
                    _, all_probabilities = model.predict_sequence(X_sequence)
                    
                    # Each probability at index i corresponds to the prediction after seeing i+1 timesteps
                    predictions_over_time = all_probabilities.tolist()
                except Exception as e:
                    # If prediction fails, create simulated predictions
                    st.error(f"Error generating sequential predictions: {str(e)}")
                    predictions_over_time = [mean_prob * (i/len(seq_df)) for i in range(1, len(seq_df) + 1)]
                
                # Create the visualization
                fig = go.Figure()
                
                # Add the prediction line
                fig.add_trace(go.Scatter(
                    x=hours[:len(predictions_over_time)],
                    y=predictions_over_time,
                    mode='lines+markers',
                    name='Sepsis Risk',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                # Add threshold lines
                fig.add_shape(
                    type="line",
                    x0=hours[0],
                    y0=0.64,
                    x1=hours[-1],
                    y1=0.64,
                    line=dict(color="red", width=2, dash="dash"),
                    name="High Risk"
                )
                
                fig.add_shape(
                    type="line",
                    x0=hours[0],
                    y0=0.44,
                    x1=hours[-1],
                    y1=0.44,
                    line=dict(color="orange", width=2, dash="dash"),
                    name="Sepsis Threshold"
                )
                
                fig.add_shape(
                    type="line",
                    x0=hours[0],
                    y0=0.24,
                    x1=hours[-1],
                    y1=0.24,
                    line=dict(color="green", width=2, dash="dash"),
                    name="Low Risk"
                )
                
                # Add annotations for thresholds
                fig.add_annotation(
                    x=hours[-1],
                    y=0.64,
                    text="High Risk",
                    showarrow=False,
                    xshift=10,
                    font=dict(color="red")
                )
                
                fig.add_annotation(
                    x=hours[-1],
                    y=0.44,
                    text="Sepsis Threshold",
                    showarrow=False,
                    xshift=10,
                    font=dict(color="orange")
                )
                
                fig.add_annotation(
                    x=hours[-1],
                    y=0.24,
                    text="Low Risk",
                    showarrow=False,
                    xshift=10,
                    font=dict(color="green")
                )
                
                # Update layout
                fig.update_layout(
                    title="Sepsis Risk Prediction with Increasing Data Points",
                    xaxis_title="Hour",
                    yaxis_title="Sepsis Risk Probability",
                    yaxis=dict(range=[0, 1]),
                    height=400,
                    plot_bgcolor='white'
                )
                        
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Clinical Recommendation")
            
                # Display recommendation based on prediction probability
        if model_type == "Random Forest":
                    # Random Forest recommendation with consensus
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
        else:
            # GRU model recommendation without consensus
            if mean_prob >= 0.64:
                st.error("🚨 **HIGH PRIORITY**: High sepsis risk detected from temporal patterns. Immediate clinical evaluation and intervention recommended.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/5.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
            elif mean_prob >= 0.44:
                st.warning("⚠️ **MODERATE PRIORITY**: Sepsis risk detected from temporal patterns. Close monitoring and clinical assessment advised.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/4.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
            elif mean_prob >= 0.24:
                st.warning(f"🤔 **UNCERTAIN PREDICTION**: Model shows some risk patterns (Sepsis: {mean_prob:.1%}). Consider additional clinical assessment and monitoring.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/3.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
            else:
                st.success("✅ **LOW PRIORITY**: Low sepsis risk based on temporal patterns. Continue routine monitoring per protocol.")
                col1, col2, col3 = st.columns([2,3,2])
                with col2:
                    try:
                        st.image("assets/priority/1.png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load priority image: {e}")
            
            # Add sequence-specific recommendations
            if len(seq_df) < 5:
                st.warning("⚠️ **Short Sequence Warning**: This prediction is based on a limited number of timepoints. Longer sequences provide more reliable predictions.")
            elif 'SepsisLabel' in seq_df.columns and seq_df['SepsisLabel'].sum() > 0:
                st.error("🚨 **Alert**: This patient has already shown sepsis indicators in the sequence data.")