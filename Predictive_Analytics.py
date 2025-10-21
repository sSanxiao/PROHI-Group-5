import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'THRESHOLD_SEPSIS': 0.44,      # Above this = Sepsis
    'THRESHOLD_NO_SEPSIS': 0.34,   # Below this = No Sepsis
    'RF_MODEL_PATH': './models/random_forest.pkl',  
    'GRU_MODEL_PATH': './models/gru_sliding_model.pth',
    'GRU_METADATA_PATH': './models/gru_sliding_model_metadata.pkl',
    'PRIORITY_IMAGES_PATH': './assets/priority/'
}

# Global variables
SAMPLE_PATIENTS = None
RF_MODEL = None
RF_SCALER = None

# Feature columns - these are the only features we'll use (original names for data processing)
FEATURE_COLUMNS = [
    'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
    'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender'
]

# Normal ranges for features (using original names without units for keys)
NORMAL_RANGES = {
    'HR': (60, 100), 'O2Sat': (95, 100), 'Temp': (36.1, 37.2), 'SBP': (90, 140),
    'MAP': (70, 100), 'Resp': (12, 20), 'HCO3': (22, 26), 'pH': (7.35, 7.45),
    'PaCO2': (35, 45), 'Creatinine': (0.6, 1.3), 'Bilirubin_direct': (0, 0.3),
    'WBC': (4.0, 11.0), 'Platelets': (150, 450), 'Age': (18, 100),
    'Hour': (0, 23), 'ICULOS': (0, 240)
}

# Feature categories for UI organization (using original names without units for matching)
FEATURE_CATEGORIES = {
    "Vital Signs": ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp'],
    "Blood Gas": ['HCO3', 'pH', 'PaCO2'],
    "Labs": ['Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets'],
    "Patient Info": ['Age', 'Gender', 'Hour', 'ICULOS']
}

# Feature display names with units
FEATURE_DISPLAY_NAMES = {
    'HR': 'HR (bpm)',
    'O2Sat': 'O2Sat (%)',
    'Temp': 'Temp (°C)',
    'SBP': 'SBP (mm Hg)',
    'MAP': 'MAP (mm Hg)',
    'Resp': 'Resp (breaths/min)',
    'HCO3': 'HCO3 (mmol/L)',
    'pH': 'pH',
    'PaCO2': 'PaCO2 (mm Hg)',
    'Creatinine': 'Creatinine (mg/dL)',
    'Bilirubin_direct': 'Bilirubin_direct (mg/dL)',
    'WBC': 'WBC (count×10³/µL)',
    'Platelets': 'Platelets (count×10³/µL)',
    'ICULOS': 'ICULOS (Intensive Care Unit Length of Stay in hours)',
    'Age': 'Age (years)',
    'Gender': 'Gender',
    'Hour': 'Hour'
}

# GRU Model Definition (Improved Architecture)
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

# GRU Sequence Wrapper
class GRUSequenceWrapper:
    def __init__(self, model, features, scaler=None):
        self.model = model.cpu()  # Ensure model is on CPU
        self.features = features
        self.scaler = scaler
        self.device = 'cpu'  # Force CPU usage for consistency
        
    def predict_sequence(self, X):
        """Predict on a sequence with proper scaling"""
        self.model.eval()
        
        # Scale the input if scaler is available
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                st.warning(f"Error scaling input: {e}. Using unscaled data.")
                X_scaled = X
        else:
            X_scaled = X
        
        # Convert to tensor and ensure it's on CPU
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)  # Add batch dimension
        
        # Make sure model is on CPU
        self.model = self.model.cpu()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(X_tensor)
        
        # Convert to numpy
        predictions = (output >= 0.5).cpu().numpy()
        probabilities = output.cpu().numpy()
        
        return predictions, probabilities

# Load Random Forest model
@st.cache_resource
def load_rf_model():
    """Load the Random Forest model"""
    
    data = joblib.load(CONFIG['RF_MODEL_PATH'])
    print(f"Successfully loaded Random Forest model from {CONFIG['RF_MODEL_PATH']}")
    
    # Verify the scaler is fitted
    if 'scaler' in data:
        try:
            # Test the scaler with a dummy input
            test_input = np.zeros((1, len(FEATURE_COLUMNS)))
            data['scaler'].transform(test_input)
            print("Scaler verification successful")
        except Exception as e:
            print(f"Scaler verification failed: {e}")
            # Replace with a new scaler fitted on actual data
            from sklearn.preprocessing import StandardScaler
            print("Creating new scaler with actual dataset...")
            try:
                # Load the actual dataset to fit the scaler properly
                df_temp = pd.read_csv('./data/cleaned_dataset.csv')
                X_temp = df_temp[FEATURE_COLUMNS].fillna(0)
                data['scaler'] = StandardScaler()
                data['scaler'].fit(X_temp)
                print("Scaler fitted with actual dataset successfully!")
            except Exception as fit_error:
                print(f"Could not fit scaler with actual data: {fit_error}")
                print("Using dummy data as fallback (not recommended for production)")
                X_dummy = np.random.rand(100, len(FEATURE_COLUMNS))
                data['scaler'] = StandardScaler()
                data['scaler'].fit(X_dummy)
    else:
        # Create a scaler if missing
        from sklearn.preprocessing import StandardScaler
        print("Scaler missing, creating new one with actual dataset...")
        try:
            # Load the actual dataset to fit the scaler properly
            df_temp = pd.read_csv('./data/cleaned_dataset.csv')
            X_temp = df_temp[FEATURE_COLUMNS].fillna(0)
            data['scaler'] = StandardScaler()
            data['scaler'].fit(X_temp)
            print("Scaler fitted with actual dataset successfully!")
        except Exception as fit_error:
            print(f"Could not fit scaler with actual data: {fit_error}")
            print("Using dummy data as fallback (not recommended for production)")
            X_dummy = np.random.rand(100, len(FEATURE_COLUMNS))
            data['scaler'] = StandardScaler()
            data['scaler'].fit(X_dummy)
        
    return data


# Load GRU model
def load_gru_model():
    """Load the improved GRU model"""
    try:
        # Load metadata first
        metadata = joblib.load(CONFIG['GRU_METADATA_PATH'])
        
        # Create a new model instance with the improved architecture
        input_size = len(FEATURE_COLUMNS)
        model = GRUModel(
            input_size=input_size,
            dropout=metadata.get('dropout', 0.6)
        )
        
        # Load the model weights - explicitly map to CPU
        checkpoint = torch.load(CONFIG['GRU_MODEL_PATH'], map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.cpu()  # Ensure model is on CPU
        model.eval()  # Set to evaluation mode
        
        # Create the wrapper with scaler
        wrapper = GRUSequenceWrapper(
            model=model,
            features=FEATURE_COLUMNS,
            scaler=metadata.get('scaler', None)
        )
        
        return wrapper
    except Exception as e:
        st.warning(f"Error loading GRU model: {e}")
        # Create a dummy model with improved architecture
        model = GRUModel(
            input_size=len(FEATURE_COLUMNS),
            dropout=0.6
        )
        return GRUSequenceWrapper(model=model, features=FEATURE_COLUMNS)

# Get input constraints for features
def get_input_constraints(feature):
    """Get very permissive min/max constraints for input fields"""
    return -100000.0, 100000.0  # Very wide range for all features

# Get predictions from individual trees
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

# Create a simple visualization of the prediction
def create_prediction_visualization(probability, threshold_sepsis, threshold_no_sepsis):
    """Create a gauge chart for the prediction probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sepsis Risk", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold_no_sepsis * 100], 'color': 'green'},
                {'range': [threshold_no_sepsis * 100, threshold_sepsis * 100], 'color': 'yellow'},
                {'range': [threshold_sepsis * 100, 100], 'color': 'red'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Create a distribution plot of tree predictions
def create_tree_distribution_plot(tree_scores, threshold_sepsis, threshold_no_sepsis):
    """Create a distribution plot of tree predictions"""
    fig = go.Figure()
    
    # Count occurrences of each unique score value
    unique_scores, counts = np.unique(np.round(tree_scores, 3), return_counts=True)
    score_counts = dict(zip(unique_scores, counts))
    
    # Create a dictionary to store counts for each score
    score_positions = {}
    for score in np.round(tree_scores, 3):
        if score not in score_positions:
            score_positions[score] = 0
        score_positions[score] += 1
    
    # Create scatter plot of tree predictions
    fig.add_trace(go.Scatter(
        x=tree_scores,
        y=np.zeros_like(tree_scores),  # All points on same y-level
        mode='markers',
        marker=dict(
            size=15,
            color=['#f44336' if s >= threshold_sepsis else  # Red for sepsis
                  '#4caf50' if s <= threshold_no_sepsis else  # Green for no sepsis
                  '#ff9800' for s in tree_scores],  # Orange for uncertain
            line=dict(width=1, color='white')
        ),
        text=[f"Tree {i+1}<br>Score: {score:.3f}" for i, score in enumerate(tree_scores)],
        hovertemplate='%{text}<extra></extra>',
        name='Tree Predictions'
    ))
    
    # Add count annotations for scores with count > 1
    for score, count in score_counts.items():
        if count > 1:
            fig.add_annotation(
                x=float(score),
                y=0.05,  # Slightly above the dots
                yanchor="bottom",
                text=f"{count}",
                showarrow=False,
                font=dict(size=12, color="black", family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="black",
                borderwidth=1,
                borderpad=2,
                opacity=0.8
            )
    
    # Calculate average score
    avg_score = np.mean(tree_scores)
    
    # Add threshold lines
    fig.add_vline(x=threshold_sepsis, 
                  line_dash="dash", line_color="red",
                  annotation=dict(
                              text="Sepsis Threshold",
                              font=dict(color="red")
                          ))
    fig.add_vline(x=threshold_no_sepsis, 
                  line_dash="dash", line_color="green",
                  annotation=dict(
                      text="No Sepsis Threshold",
                      font=dict(color="green")
                  ))
    
    # Add average score line
    fig.add_vline(x=avg_score,
                  line_dash="dash", line_color="blue", line_width=2,
                  annotation=dict(
                      text=f"Average: {avg_score:.3f}",
                      font=dict(color="blue", size=10, family="Arial Black"),
                      bgcolor="rgba(255, 255, 255, 0.7)",
                      bordercolor="blue",
                      borderwidth=1,
                      borderpad=4,
                      y=0.05,            # <--- lower this value to move it further down (range: 0 to 1)
                      yanchor="bottom"         # <--- this anchors the annotation to the bottom of the plot   
                  ))
        
    # Update layout
    fig.update_layout(
        xaxis=dict(
            title="Prediction Score",
            range=[-0.1, 1.1],  # Expanded range for better visualization
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
    
    return fig

# Load sample patients
@st.cache_resource
def load_sample_patients():
    """Load and prepare a balanced set of sample patients"""
    try:
        # Check if data file exists
        data_path = './data/cleaned_dataset.csv'
        if not os.path.exists(data_path):
            st.error(f"Data file not found: {data_path}")
            return None
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"DEBUG: Loaded data shape: {df.shape}")
        
        # Check if required columns exist
        required_columns = ['Patient_ID', 'SepsisLabel']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Drop specified columns
        columns_drop = {
            'Unnamed: 0', 'Unit1', 'Unit2'
        }
        df = df.drop(columns=[col for col in columns_drop if col in df.columns])
        
        # Separate sepsis and non-sepsis cases
        sepsis_cases = df[df['SepsisLabel'] == 1]
        non_sepsis_cases = df[df['SepsisLabel'] == 0]
        
        print(f"DEBUG: Sepsis cases: {len(sepsis_cases)}, Non-sepsis cases: {len(non_sepsis_cases)}")
        
        # Sample equal numbers from each class (100 from each)
        n_samples = min(100, len(sepsis_cases))
        if n_samples == 0:
            st.error("No sepsis cases found in the data")
            return None
        
        balanced_sepsis = sepsis_cases.sample(n=n_samples, random_state=42)
        balanced_non_sepsis = non_sepsis_cases.sample(n=n_samples, random_state=42)
        
        # Combine and shuffle
        balanced_df = pd.concat([balanced_sepsis, balanced_non_sepsis])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"DEBUG: Final balanced sample shape: {balanced_df.shape}")
        return balanced_df
        
    except Exception as e:
        st.error(f"Error loading sample patients: {e}")
        print(f"DEBUG: Error details: {str(e)}")
        return None

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

# Create a temporal prediction evolution chart
def create_prediction_evolution_chart(hours, predictions, threshold_sepsis, threshold_no_sepsis):
    """Create a chart showing how predictions evolve over time"""
    fig = go.Figure()
    
    # Add the prediction line
    fig.add_trace(go.Scatter(
        x=hours[:len(predictions)],
        y=predictions,
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
    
    return fig

# Main function for the Streamlit app
def predictive_analytics():
    st.info("This page allows you to input patient data and receive sepsis risk predictions from pre-trained machine learning models.")
    st.markdown("---")
    
    # Create directory for assets if it doesn't exist
    os.makedirs("assets/priority", exist_ok=True)
    
    # Initialize global variables
    global SAMPLE_PATIENTS
    if SAMPLE_PATIENTS is None:
        # Use cached data from the main dashboard
        if 'data' in st.session_state:
            # Use cached balanced sample
            if st.session_state.data.get('balanced_sample') is not None:
                SAMPLE_PATIENTS = st.session_state.data['balanced_sample']
            else:
                # If no cached balanced sample, try to use the main dataframe
                if st.session_state.data.get('df') is not None:
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
    
    # Model selection
    model_type = st.radio(
        "Select prediction approach:",
        ["Single Timepoint Analysis", "Sequential History Analysis"],
        horizontal=True,
        help="Single Timepoint Analysis uses data from one hour only. Sequential History Analysis uses multiple hours of patient data to detect temporal patterns."
    )
    
    # Initialize session state for sequence data if not exists
    if 'sequence_data' not in st.session_state:
        st.session_state.sequence_data = []
    
    # Different input sections based on model type
    if model_type == "Single Timepoint Analysis":
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
        
        # Initialize input data dictionary
        input_data = {}
    
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
                            FEATURE_DISPLAY_NAMES.get(feature, feature), 
                            options=[0, 1], 
                            index=default_gender if default_gender in [0, 1] else 0,
                            format_func=lambda x: "Female" if x == 0 else "Male",
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    elif feature == 'Hour':
                        default_hour = int(st.session_state.get(feature, 0))  # Ensure integer
                        value = st.selectbox(
                            FEATURE_DISPLAY_NAMES.get(feature, feature), 
                            options=list(range(24)), 
                            index=default_hour if 0 <= default_hour < 24 else 0,
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    elif feature == 'ICULOS':
                        default_iculos = int(st.session_state.get(feature, 0))  # Ensure integer
                        value = st.number_input(
                            FEATURE_DISPLAY_NAMES.get(feature, feature), 
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
                            FEATURE_DISPLAY_NAMES.get(feature, feature), 
                            min_value=float(min_val), 
                            max_value=float(max_val),
                            value=float(current_val),
                            step=step,
                            key=session_key,
                            disabled=st.session_state.get('inputs_locked', False)
                        )
                    
                    input_data[feature] = value
    
        # Add consent checkbox
        st.markdown("---")
        consent_rf = st.checkbox(
            "I understand that this prediction is for educational purposes only and may be inaccurate. Clinical decisions should always be made by qualified healthcare professionals.",
            key="consent_checkbox_rf"
        )
        
        # Prediction button
        submitted = st.button("🔍 Get Single Timepoint Prediction", type="primary", use_container_width=True, disabled=not consent_rf)
        
        if submitted:
            if not consent_rf:
                st.warning("Please acknowledge the consent notice before proceeding with the prediction.")
            else:
                # Save input data to session state for use in Prescriptive Analytics
                st.session_state.predictive_input = input_data.copy()
                # Display a prominent disclaimer
                st.warning("""
                **IMPORTANT DISCLAIMER**: This prediction is based on a machine learning model and is not a substitute for professional medical diagnosis. 
                The model may be inaccurate in individual cases. Always consult with qualified healthcare professionals for clinical decisions.
                """)
                
                # Load the Random Forest model if not already loaded
                global RF_MODEL, RF_SCALER
                try:
                    data = load_rf_model()
                    if data is not None:
                        RF_MODEL = data['model']
                        RF_SCALER = data['scaler']
                        print("Successfully loaded Random Forest model and scaler")
                    else:
                        st.error("Failed to load Random Forest model. Please check if the model file exists.")
                        return
                except Exception as e:
                    st.error(f"Error loading Random Forest model: {e}")
                    return
                
                # Create DataFrame with features in exact order
                X_input = pd.DataFrame([input_data])[FEATURE_COLUMNS]
                X_input = X_input.fillna(0)  # Fill any missing values with 0
                
                # Scale the input data with robust fallback
                X_input_scaled = None
                
                # First try the loaded scaler
                if RF_SCALER is not None:
                    try:
                        X_input_scaled = RF_SCALER.transform(X_input)
                        print("Successfully scaled input with loaded scaler")
                    except Exception as e:
                        print(f"Error with loaded scaler: {e}")
                        
                # If that fails, create a new scaler
                if X_input_scaled is None:
                    try:
                        from sklearn.preprocessing import StandardScaler
                        print("Creating new scaler for input")
                        temp_scaler = StandardScaler()
                        X_input_scaled = temp_scaler.fit_transform(X_input)
                        st.warning("Using a temporary scaler as fallback.")
                    except Exception as e:
                        print(f"Error with temporary scaler: {e}")
                        # Last resort: no scaling
                        X_input_scaled = X_input.values
                        st.warning("Using unscaled input as fallback.")
        
                # Get predictions from all trees
                tree_preds, tree_scores = get_individual_tree_predictions(RF_MODEL, X_input_scaled)
                
                # Calculate mean probability
                probability = np.mean(tree_scores)
                
                # Display results
                st.markdown("## Prediction Results")
                
                # Display gauge chart
                fig = create_prediction_visualization(
                    probability, 
                    CONFIG['THRESHOLD_SEPSIS'], 
                    CONFIG['THRESHOLD_NO_SEPSIS']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add distribution plot of tree predictions
                st.markdown('<h3 class="sub-header">🏛️ Virtual Doctors Advisory Board Decision</h3>', unsafe_allow_html=True)
                fig_dist = create_tree_distribution_plot(tree_scores, CONFIG['THRESHOLD_SEPSIS'], CONFIG['THRESHOLD_NO_SEPSIS'])
                st.plotly_chart(fig_dist, use_container_width=True)
        
                # Calculate votes and consensus
                sepsis_votes = sum(s >= CONFIG['THRESHOLD_SEPSIS'] for s in tree_scores)
                no_sepsis_votes = sum(s <= CONFIG['THRESHOLD_NO_SEPSIS'] for s in tree_scores)
                uncertain_votes = sum(CONFIG['THRESHOLD_NO_SEPSIS'] < s < CONFIG['THRESHOLD_SEPSIS'] for s in tree_scores)
                
                # Calculate consensus percentage
                consensus_pct = max(sepsis_votes, no_sepsis_votes) / len(tree_scores) * 100 if len(tree_scores) > 0 else 0

                # Show summary statistics in Streamlit
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                            st.metric("Mean Score", f"{probability:.3f}")
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Virtual Doctors Predicting Sepsis</p>
                        <p style='margin: 0; color: #f44336; font-size: 2rem; font-weight: 600'>{sepsis_votes}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Virtual Doctors Predicting No Sepsis</p>
                        <p style='margin: 0; color: #4caf50; font-size: 2rem; font-weight: 600'>{no_sepsis_votes}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Uncertain Virtual Doctors</p>
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

                st.markdown("### Clinical Recommendation")
        
                # Display recommendation based on prediction probability
                if probability >= CONFIG['THRESHOLD_SEPSIS']:
                    if consensus_pct >= 70:
                        st.error("🚨 **HIGH PRIORITY**: Strong consensus for sepsis risk. Immediate clinical evaluation and intervention recommended.")
                        col1, col2, col3 = st.columns([2,3,2])
                        with col2:
                                    try:
                                        st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}5.png", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not load priority image: {e}")
                    else:
                        st.warning("⚠️ **MODERATE PRIORITY**: Sepsis risk detected with some uncertainty. Close monitoring and clinical assessment advised.")
                        col1, col2, col3 = st.columns([2,3,2])
                        with col2:
                                    try:
                                        st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}4.png", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not load priority image: {e}")
                elif probability <= CONFIG['THRESHOLD_NO_SEPSIS']:
                    if consensus_pct >= 70:
                        st.success("✅ **LOW PRIORITY**: Strong consensus for low sepsis risk. Continue routine monitoring per protocol.")
                        col1, col2, col3 = st.columns([2,3,2])
                        with col2:
                                    try:
                                        st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}1.png", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not load priority image: {e}")
                    else:
                        st.info("ℹ️ **ROUTINE MONITORING**: Low sepsis risk indicated but maintain standard care vigilance.")
                        col1, col2, col3 = st.columns([2,3,2])
                        with col2:
                                    try:
                                        st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}2.png", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not load priority image: {e}")
                else:
                    st.warning(f"🤔 **UNCERTAIN PREDICTION**: Model is uncertain (Sepsis: {probability:.1%}). Consider additional clinical assessment, laboratory tests, and expert consultation. Monitor closely for clinical deterioration.")
                    col1, col2, col3 = st.columns([2,3,2])
                    with col2:
                                try:
                                    st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}3.png", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not load priority image: {e}")
                        
                # Display disclaimer at the bottom
                st.info("**DISCLAIMER**: This prediction is for educational purposes only and may not be accurate. Clinical decisions should be made by qualified healthcare professionals.")

    else:  # GRU model
        st.markdown('<h3 class="sub-header">📊 Sequential History Patient Information</h3>', unsafe_allow_html=True)
        
        # Explanation of sequential input
        st.info("""
        This approach analyzes a patient's history over time to detect patterns that may indicate developing sepsis.
        You can add multiple timepoints (hours) for a patient, and the model will analyze 
        the temporal patterns and trends to predict sepsis risk.
        """)
        
        # Controls for sequence data
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎲 Select Random Patient", help="Load a random patient's sequence data"):
                # Clear existing sequence and status
                st.session_state.sequence_data = []
                st.session_state.patient_sepsis_status = None
                st.session_state.patient_id = None
                
                # Get a random patient ID - try to load from full dataset for better sequences
                if SAMPLE_PATIENTS is not None and len(SAMPLE_PATIENTS) > 0:
                    # Try to load from full dataset for longer sequences
                    try:
                        full_df = pd.read_csv('./data/cleaned_dataset.csv')
                        print(f"DEBUG: Full dataset shape: {full_df.shape}")
                        
                        # Get patient counts from full dataset
                        full_patient_counts = full_df['Patient_ID'].value_counts()
                        print(f"DEBUG: Full dataset has {len(full_patient_counts)} unique patients")
                        
                        # Find patients with longer sequences (prefer >10 records)
                        patients_with_long_sequences = full_patient_counts[full_patient_counts > 10]
                        if len(patients_with_long_sequences) == 0:
                            # Fallback to patients with >5 records
                            patients_with_long_sequences = full_patient_counts[full_patient_counts > 5]
                        if len(patients_with_long_sequences) == 0:
                            # Last resort: patients with >1 records
                            patients_with_long_sequences = full_patient_counts[full_patient_counts > 1]
                        
                        if len(patients_with_long_sequences) == 0:
                            raise ValueError("No patients found in full dataset")
                        
                        print(f"DEBUG: Found {len(patients_with_long_sequences)} patients with long sequences")
                        
                        # Check sepsis distribution in the available patients
                        sepsis_patients = []
                        non_sepsis_patients = []
                        
                        for patient_id in patients_with_long_sequences.index[:50]:  # Check first 50 for efficiency
                            patient_data = full_df[full_df['Patient_ID'] == patient_id]
                            has_sepsis = (patient_data['SepsisLabel'] == 1).any()
                            if has_sepsis:
                                sepsis_patients.append(patient_id)
                            else:
                                non_sepsis_patients.append(patient_id)
                        
                        print(f"DEBUG: Sepsis patients available: {len(sepsis_patients)}")
                        print(f"DEBUG: Non-sepsis patients available: {len(non_sepsis_patients)}")
                        
                        # Prefer sepsis patients (70% chance) to get more interesting cases
                        if len(sepsis_patients) > 0 and np.random.random() < 0.7:
                            random_patient_id = np.random.choice(sepsis_patients)
                            print(f"DEBUG: Selected sepsis patient: {random_patient_id}")
                        else:
                            # Select from all available patients
                            random_patient_id = patients_with_long_sequences.sample(1).index[0]
                            print(f"DEBUG: Selected random patient: {random_patient_id}")
                        
                        # Get all records for this patient from full dataset
                        patient_records = full_df[full_df['Patient_ID'] == random_patient_id].sort_values('Hour')
                        
                        print(f"DEBUG: Using full dataset - Patient {random_patient_id} has {len(patient_records)} records")
                        
                    except Exception as e:
                        print(f"DEBUG: Could not load from full dataset: {e}")
                        print("DEBUG: Falling back to SAMPLE_PATIENTS...")
                        # Fallback to SAMPLE_PATIENTS
                        full_df = SAMPLE_PATIENTS
                        patient_counts = SAMPLE_PATIENTS['Patient_ID'].value_counts()
                        
                        # Try to find patients with multiple records (prefer > 3, fallback to > 1)
                        patients_with_multiple = patient_counts[patient_counts > 3]
                        if len(patients_with_multiple) == 0:
                            # Fallback to patients with at least 2 records
                            patients_with_multiple = patient_counts[patient_counts > 1]
                        
                        if len(patients_with_multiple) == 0:
                            # Last resort: use any patient
                            patients_with_multiple = patient_counts
                        
                        if len(patients_with_multiple) == 0:
                            raise ValueError("No patients found in sample data")
                        
                        # Select a random patient
                        random_patient_id = patients_with_multiple.sample(1).index[0]
                        
                        # Get all records for this patient
                        patient_records = SAMPLE_PATIENTS[SAMPLE_PATIENTS['Patient_ID'] == random_patient_id].sort_values('Hour')
                        
                        print(f"DEBUG: Using SAMPLE_PATIENTS - Patient {random_patient_id} has {len(patient_records)} records")
                    
                    # Debug: Print patient record information
                    print(f"DEBUG: Patient {random_patient_id} has {len(patient_records)} records")
                    print(f"DEBUG: Hour range: {patient_records['Hour'].min()} to {patient_records['Hour'].max()}")
                    print(f"DEBUG: Available hours: {sorted(patient_records['Hour'].unique())}")
                    
                    # Check sepsis status for this patient
                    has_sepsis = (patient_records['SepsisLabel'] == 1).any()
                    sepsis_hours = patient_records[patient_records['SepsisLabel'] == 1]['Hour'].tolist()
                    
                    # Store sepsis status in session state
                    st.session_state.patient_sepsis_status = {
                        'has_sepsis': has_sepsis,
                        'sepsis_hours': sepsis_hours,
                        'patient_id': random_patient_id
                    }
                    
                    # Debug: Print sepsis status (remove this in production)
                    print(f"DEBUG: Patient {random_patient_id} - Has Sepsis: {has_sepsis}, Hours: {sepsis_hours}")
                    
                    # Convert to list of dictionaries for sequence_data
                    print(f"DEBUG: Converting {len(patient_records)} records to sequence data...")
                    for i, (_, row) in enumerate(patient_records.iterrows()):
                        timepoint = {}
                        for feature in FEATURE_COLUMNS:
                            if feature in row:
                                timepoint[feature] = row[feature]
                        # Also include SepsisLabel for status display
                        if 'SepsisLabel' in row:
                            timepoint['SepsisLabel'] = row['SepsisLabel']
                        # Include Hour for debugging
                        if 'Hour' in row:
                            timepoint['Hour'] = row['Hour']
                        st.session_state.sequence_data.append(timepoint)
                        print(f"DEBUG: Added timepoint {i+1}: Hour {row.get('Hour', 'N/A')}, SepsisLabel {row.get('SepsisLabel', 'N/A')}")
                    
                    st.success(f"✅ Loaded {len(st.session_state.sequence_data)} hours of data for Patient ID: {random_patient_id}")

                else:
                    if SAMPLE_PATIENTS is None:
                        st.error("Sample patient data not loaded. Please check the data file.")
                    elif len(SAMPLE_PATIENTS) == 0:
                        st.error("Sample patient data is empty. Please check the data file.")
                    else:
                        st.error("Sample patient data not available")
                
                st.rerun()
        
        with col2:
            if st.button("🔄 Clear Sequence", help="Clear the current sequence data"):
                st.session_state.sequence_data = []
                st.session_state.patient_sepsis_status = None
                st.session_state.patient_id = None
                st.rerun()
        
        
        # Display current sequence
        if st.session_state.sequence_data:
            # Convert to DataFrame for display
            seq_df = pd.DataFrame(st.session_state.sequence_data)
            
            # Show sequence length
            st.markdown(f"**Current Sequence Length:** {len(seq_df)} hours")
            
            # Display sepsis status from session state
            if hasattr(st.session_state, 'patient_sepsis_status') and st.session_state.patient_sepsis_status is not None:
                status = st.session_state.patient_sepsis_status
                patient_id = status['patient_id']
                has_sepsis = status['has_sepsis']
                sepsis_hours = status['sepsis_hours']
                
                st.markdown(f"**Patient ID:** {patient_id}")
                
                # Display sepsis status
                if has_sepsis:
                    if len(sepsis_hours) == 1:
                        st.warning(f"🚨 **Patient HAS SEPSIS** at hour {sepsis_hours[0]}")
                    else:
                        st.warning(f"🚨 **Patient HAS SEPSIS** at hours: {', '.join(map(str, sepsis_hours))}")
                else:
                    st.success("✅ **Patient does NOT develop sepsis**")
            
            # Check if we have sepsis information in the sequence data (fallback)
            elif 'SepsisLabel' in seq_df.columns:
                has_sepsis = (seq_df['SepsisLabel'] == 1).any()
                sepsis_hours = seq_df[seq_df['SepsisLabel'] == 1]['Hour'].tolist()
                
                # Display sepsis status
                if has_sepsis:
                    if len(sepsis_hours) == 1:
                        st.warning(f"🚨 **Patient HAS SEPSIS** at hour {sepsis_hours[0]}")
                    else:
                        st.warning(f"🚨 **Patient HAS SEPSIS** at hours: {', '.join(map(str, sepsis_hours))}")
                else:
                    st.success("✅ **Patient does NOT develop sepsis**")
            
            # Display the sequence data in a table (exclude SepsisLabel from display)
            display_columns = [col for col in FEATURE_COLUMNS if col in seq_df.columns]
            st.dataframe(
                seq_df[display_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No sequence data available. Please add timepoints below.")
        
        # Form for adding a timepoint
        st.subheader("Add a Timepoint")
        with st.form("add_timepoint_form"):
            # Create columns for the form fields
            form_cols = st.columns(3)
            
            # Initialize input data dictionary
            new_timepoint = {}
            
            # Add Hour field first
            next_hour = len(st.session_state.sequence_data)
            if st.session_state.sequence_data:
                # If we have sequence data, get the max hour + 1
                seq_df = pd.DataFrame(st.session_state.sequence_data)
                if 'Hour' in seq_df.columns:
                    next_hour = int(seq_df['Hour'].max() + 1)
                    # Warn if the sequence is getting very long
                    if next_hour > 500:
                        st.warning(f"⚠️ Sequence is very long ({next_hour} hours). Consider starting a new sequence.")
                        next_hour = 0  # Reset to 0 for new sequence
            
            new_timepoint['Hour'] = form_cols[0].number_input(
                FEATURE_DISPLAY_NAMES.get('Hour', 'Hour'),
                min_value=0,
                max_value=500,  # Increased max_value to accommodate longer sequences
                value=next_hour,
                step=1,
                help="Hour of the timepoint (0-500)"
            )
            
            # Add other fields
            col_idx = 1
            for feature in [f for f in FEATURE_COLUMNS if f != 'Hour']:
                # Get normal range if available
                if feature in NORMAL_RANGES:
                    min_val, max_val = NORMAL_RANGES[feature]
                    default_val = (min_val + max_val) / 2
                else:
                    min_val, max_val = 0, 100
                    default_val = 0
                
                # Determine appropriate step size
                step = 1.0 if feature in ['HR', 'Age', 'Resp', 'ICULOS'] else 0.1
                
                # Special handling for Gender
                if feature == 'Gender':
                    new_timepoint[feature] = form_cols[col_idx % 3].selectbox(
                        FEATURE_DISPLAY_NAMES.get(feature, feature),
                        options=[0, 1],
                        format_func=lambda x: "Female" if x == 0 else "Male"
                    )
                else:
                    new_timepoint[feature] = form_cols[col_idx % 3].number_input(
                        FEATURE_DISPLAY_NAMES.get(feature, feature),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_val),
                        step=step
                    )
                
                col_idx += 1
            
            # Submit button
            submit_timepoint = st.form_submit_button("Add Timepoint")
            
            if submit_timepoint:
                # Add the new timepoint to the sequence
                st.session_state.sequence_data.append(new_timepoint)
                st.rerun()
        
        # Add consent checkbox
        consent = st.checkbox(
            "I understand that this prediction is for educational purposes only and may be inaccurate. Clinical decisions should always be made by qualified healthcare professionals.",
            key="consent_checkbox_gru"
        )
        
        # Prediction button
        if st.button("🔍 Get Sequential History Prediction", type="primary", use_container_width=True, disabled=not st.session_state.sequence_data or not consent):
            if not consent:
                st.warning("Please acknowledge the consent notice before proceeding with the prediction.")
            elif not st.session_state.sequence_data:
                st.error("No sequence data available. Please add timepoints to the sequence.")
            else:
                # Display a prominent disclaimer
                st.warning("""
                **IMPORTANT DISCLAIMER**: This prediction is based on a machine learning model and is not a substitute for professional medical diagnosis. 
                The model may be inaccurate in individual cases. Always consult with qualified healthcare professionals for clinical decisions.
                """)
                
                # Convert sequence data to DataFrame
                seq_df = pd.DataFrame(st.session_state.sequence_data)
                
                # Make sure we have all required features
                for feature in FEATURE_COLUMNS:
                    if feature not in seq_df.columns:
                        seq_df[feature] = 0  # Default value
                
                # Sort by Hour to ensure temporal order
                seq_df = seq_df.sort_values('Hour').reset_index(drop=True)
                
                # Extract features in the correct order
                X_sequence = seq_df[FEATURE_COLUMNS].values
                
                # Load GRU model
                model = load_gru_model()
                
                # Make prediction
                with st.spinner("Running GRU prediction on temporal sequence..."):
                    try:
                        # Use the model's predict_sequence method
                        _, probabilities = model.predict_sequence(X_sequence)
                        
                        # Get the prediction for the last timestep
                        probability = probabilities[-1]
                        
                        # Store all probabilities for temporal visualization
                        all_probabilities = probabilities
                    except Exception as e:
                        st.error(f"Error in GRU prediction: {str(e)}")
                        
                        # Fallback to a simpler approach - simulate a prediction
                        st.warning("Using simulated prediction as fallback")
                        
                        # Generate a probability based on clinical heuristics
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
                        probability = min(1.0, max(0.0, 
                            hr_risk * 0.25 + 
                            temp_risk * 0.25 + 
                            resp_risk * 0.15 + 
                            wbc_risk * 0.15 +
                            hr_trend_risk * 0.1 +
                            temp_trend_risk * 0.1))
                        
                        # Add some randomness to simulate model uncertainty
                        probability = min(1.0, max(0.0, probability + np.random.normal(0, 0.05)))
                        
                        # Create simulated probabilities for all timesteps
                        all_probabilities = np.linspace(probability * 0.5, probability, len(seq_df))
                
                # Display results
                st.markdown("## Prediction Results")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sequence Length", f"{len(seq_df)} hours")
                with col2:
                    st.metric("Sepsis Risk", f"{probability:.1%}")
                with col3:
                    risk_level = "High" if probability >= CONFIG['THRESHOLD_SEPSIS'] else "Low" if probability <= CONFIG['THRESHOLD_NO_SEPSIS'] else "Moderate"
                    risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Moderate" else "green"
                    st.markdown(f"""
                    <div style='text-align: center'>
                        <p style='margin-bottom: 0px; color: gray; font-size: 14px'>Risk Level</p>
                        <p style='margin: 0; color: {risk_color}; font-size: 2rem; font-weight: 600'>{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display gauge chart
                fig = create_prediction_visualization(
                    probability, 
                    CONFIG['THRESHOLD_SEPSIS'], 
                    CONFIG['THRESHOLD_NO_SEPSIS']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display clinical recommendation
                st.markdown("### Clinical Recommendation")
                
                # Display recommendation based on prediction probability
                if probability >= 0.64:  # Higher threshold for high risk
                    st.error("🚨 **HIGH PRIORITY**: High sepsis risk detected from temporal patterns. Immediate clinical evaluation and intervention recommended.")
                    col1, col2, col3 = st.columns([2,3,2])
                    with col2:
                        try:
                            st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}5.png", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load priority image: {e}")
                elif probability >= CONFIG['THRESHOLD_SEPSIS']:
                    st.warning("⚠️ **MODERATE PRIORITY**: Sepsis risk detected from temporal patterns. Close monitoring and clinical assessment advised.")
                    col1, col2, col3 = st.columns([2,3,2])
                    with col2:
                        try:
                            st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}4.png", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load priority image: {e}")
                elif probability >= 0.24:  # Lower threshold for uncertain
                    st.warning(f"🤔 **UNCERTAIN PREDICTION**: Model shows some risk patterns (Sepsis: {probability:.1%}). Consider additional clinical assessment and monitoring.")
                    col1, col2, col3 = st.columns([2,3,2])
                    with col2:
                        try:
                            st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}3.png", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load priority image: {e}")
                else:
                    st.success("✅ **LOW PRIORITY**: Low sepsis risk based on temporal patterns. Continue routine monitoring per protocol.")
                    col1, col2, col3 = st.columns([2,3,2])
                    with col2:
                        try:
                            st.image(f"{CONFIG['PRIORITY_IMAGES_PATH']}1.png", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load priority image: {e}")
                
                # Add sequence-specific recommendations
                if len(seq_df) < 5:
                    st.warning("⚠️ **Short Sequence Warning**: This prediction is based on a limited number of timepoints. Longer sequences provide more reliable predictions.")
                
                # Display disclaimer
                st.info("**DISCLAIMER**: This prediction is for educational purposes only and may not be accurate. Clinical decisions should be made by qualified healthcare professionals.")

# Run the app
if __name__ == "__main__":
    predictive_analytics()