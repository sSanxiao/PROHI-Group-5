import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Set page config
st.set_page_config(
    page_title="PROHI Sepsis Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directory structure if it doesn't exist
os.makedirs("assets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define CSS for styling
st.markdown("""
<style>
/* Main header styling */
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #2E86AB;
    margin-bottom: 1rem;
    padding: 1rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Sub header styling */
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #A23B72;
    margin: 1rem 0;
    padding-left: 0.5rem;
    border-left: 4px solid #A23B72;
}

/* Feature category styling */
.feature-category {
    margin-bottom: 1rem;
    padding: 1rem;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.feature-category-header {
    font-weight: bold;
    color: #2E86AB;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
}

.feature-category-icon {
    margin-right: 8px;
}

/* Card styling for feature inputs */
.feature-card {
    background-color: white;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.feature-card:hover {
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
}

.feature-label {
    font-weight: 500;
    color: #495057;
    margin-bottom: 5px;
}

/* Make the Streamlit components look nicer */
div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem;
}

/* Button styling */
.stButton > button {
    border-radius: 20px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Metric styling */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

/* Alert styling */
div.stAlert {
    border-radius: 10px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #e9ecef;
}

/* Sidebar nav styling */
.sidebar-nav {
    padding: 1rem 0.5rem;
}

.nav-item {
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
    background-color: white;
    display: flex;
    align-items: center;
}

.nav-item:hover {
    background-color: #e9ecef;
    transform: translateX(5px);
}

.nav-item.active {
    background-color: #e6f0ff;
    border-left: 4px solid #2E86AB;
    color: #2E86AB;
    font-weight: bold;
}

.nav-icon {
    margin-right: 0.75rem;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# Centralized data loading functions with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataset():
    """Load and preprocess the main dataset"""
    try:
        data_path = "./data/cleaned_dataset.csv"
        df = pd.read_csv(data_path)
        
        # Drop specified columns
        columns_drop = {
            'Unnamed: 0', 'Unit1', 'Unit2'
        }
        df = df.drop(columns=[col for col in columns_drop if col in df.columns])
        
        # Convert SepsisLabel to category if it exists
        if 'SepsisLabel' in df.columns:
            df['SepsisLabel'] = df['SepsisLabel'].astype('category')
            
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Return a small dummy dataset for demonstration
        return create_dummy_dataset()

def create_dummy_dataset():
    """Create a dummy dataset for demonstration when real data is unavailable"""
    # Create a small dataset with the expected columns
    np.random.seed(42)
    n_samples = 100
    
    # Create common features
    data = {
        'Patient_ID': range(1, n_samples + 1),
        'Hour': np.random.randint(0, 24, n_samples),
        'HR': np.random.normal(80, 15, n_samples),
        'O2Sat': np.random.normal(95, 5, n_samples),
        'Temp': np.random.normal(37, 1, n_samples),
        'MAP': np.random.normal(85, 10, n_samples),
        'Resp': np.random.normal(18, 5, n_samples),
        'Age': np.random.normal(65, 15, n_samples),
        'Gender': np.random.randint(0, 2, n_samples),
        'SepsisLabel': np.random.randint(0, 2, n_samples)
    }
    
    # Add some additional features
    for feature in ['Glucose', 'Lactate', 'WBC', 'Creatinine', 'Platelets', 'HospAdmTime', 'ICULOS']:
        data[feature] = np.random.normal(100, 30, n_samples)
    
    df = pd.DataFrame(data)
    
    # Convert SepsisLabel to category
    df['SepsisLabel'] = df['SepsisLabel'].astype('category')
    
    return df

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = './models/random_forest.pkl'
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            return data
        else:
            st.warning("Model file not found. Using demo mode.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def get_balanced_sample(df, n_samples=100):
    """Get a balanced sample of sepsis and non-sepsis cases"""
    if df is None:
        return None
        
    # Separate sepsis and non-sepsis cases
    sepsis_cases = df[df['SepsisLabel'] == 1]
    non_sepsis_cases = df[df['SepsisLabel'] == 0]
    
    # Sample equal numbers from each class
    n_samples = min(n_samples, len(sepsis_cases), len(non_sepsis_cases))
    balanced_sepsis = sepsis_cases.sample(n=n_samples, random_state=42)
    balanced_non_sepsis = non_sepsis_cases.sample(n=n_samples, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([balanced_sepsis, balanced_non_sepsis])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

# Import the modules for each tab
from Descriptive_Analytics import descriptive_analytics
from Diagnostic_Analytics import diagnostic_analytics
from Predictive_Analytics import predictive_analytics
from Prescriptive_Analytics import prescriptive_analytics
from About import about_page

def main():
    # Define tab data
    tabs = [
        {"id": "descriptive", "icon": "📊", "label": "Descriptive Analytics", "function": descriptive_analytics},
        {"id": "diagnostic", "icon": "🔬", "label": "Diagnostic Analytics", "function": diagnostic_analytics},
        {"id": "predictive", "icon": "🔮", "label": "Predictive Analytics", "function": predictive_analytics},
        {"id": "prescriptive", "icon": "💡", "label": "Prescriptive Analytics", "function": prescriptive_analytics},
        {"id": "about", "icon": "ℹ️", "label": "About", "function": about_page}
    ]
    
    # Load data once for all tabs
    with st.spinner("Loading data..."):
        df = load_dataset()
        model_data = load_model()
        balanced_sample = get_balanced_sample(df)
    
    # Store data in session state for access across tabs
    if 'data' not in st.session_state:
        st.session_state.data = {
            'df': df,
            'model_data': model_data,
            'balanced_sample': balanced_sample
        }
    
    # Initialize active tab if not set
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = tabs[0]["id"]
    
    # Create sidebar navigation
    with st.sidebar:
        st.image("assets/project-logo.jpg", width=200, use_container_width=True)
        st.markdown("## PROHI Sepsis Dashboard")
        st.markdown("---")
        
        # Create navigation items with beautiful styling
        for tab in tabs:
            # Determine if this tab is active
            is_active = st.session_state.active_tab == tab["id"]
            button_style = "primary" if is_active else "secondary"
            
            # Create a direct, visible button with icon and label
            if st.button(f"{tab['icon']} {tab['label']}", 
                         key=f"btn_{tab['id']}", 
                         use_container_width=True,
                         type=button_style):
                st.session_state.active_tab = tab["id"]
                st.rerun()
    
    # Main content area
    st.markdown('<h1 class="main-header">🩺 PROHI Sepsis Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Get the active tab and call its function
    active_tab = next((tab for tab in tabs if tab["id"] == st.session_state.active_tab), tabs[0])
    active_tab["function"]()

if __name__ == "__main__":
    main()