import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from google.oauth2 import service_account

GCS_FILE_PATH = "gs://prohi_group_5/cleaned_dataset.csv"

# ----------------------------------------------------
# 函数：获取 GCS 认证选项
# ----------------------------------------------------
def get_gcs_storage_options():
    # 1. 检查 Streamlit Cloud Secrets 中是否有密钥 (用于部署环境)
    if "gcp_service_account" in st.secrets:
        # Streamlit Secrets 将 JSON 密钥存储为字符串
        # 我们需要将其解析为 Python 字典
        key_dict = json.loads(st.secrets["gcp_service_account"])
        
        # 从字典创建服务账号凭证对象
        credentials = service_account.Credentials.from_service_account_info(key_dict)
        
        # 返回 gcsfs 所需的 token 认证参数
        return {'token': credentials}
        
    # 2. 如果 Secrets 中没有密钥 (用于本地环境)
    else:
        # gcsfs 将自动尝试使用 gcloud auth application-default login
        # 或 GOOGLE_APPLICATION_CREDENTIALS 环境变量
        # 本地验证步骤将依赖于此
        return {}


@st.cache_data(show_spinner="Loading Dataset From Google Cloud Storage ...")
def load_large_data_from_gcs(gcs_path, _storage_options):
    try:
        # pandas 和 gcsfs 协同工作，通过 storage_options 进行认证
        df = pd.read_csv(gcs_path, storage_options=_storage_options)
        return df
    except Exception as e:
        st.error(f"Cannot load dataset from GCS. Please check keys,GCS file PATH or permission.Detalied Information: {e}")
        return None


# Set page config
st.set_page_config(
    page_title="PROHI Sepsis Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force white theme
st.markdown("""
<style>
/* Force white background for the entire app */
.stApp {
    background-color: white;
}

/* Force white background for main content area */
.main .block-container {
    background-color: white;
}

/* Force white background for sidebar */
section[data-testid="stSidebar"] {
    background-color: white;
}

/* Force white background for all containers */
div[data-testid="stVerticalBlock"] {
    background-color: white;
}

/* Override any dark theme styles */
.stApp > header {
    background-color: white;
}

.stApp > div {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

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
def preprocess_gcs_data(df):
    """Apply necessary preprocessing (like column filtering) after GCS load"""
    if df is None:
        return None
        
    try:   
        # Keep only the 17 features we need plus ID and label
        columns_to_keep = [
            'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
            'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender',
            'Patient_ID', 'SepsisLabel'
        ]
        # Only keep columns that exist in the dataframe
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep]
        
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
    with st.spinner("Loading data..."): # <--- 重新添加 with 块
        # 1. 获取认证选项
        storage_options = get_gcs_storage_options() 
        
        # 2. 使用 GCS 加载数据 (使用 Dashboard.py 开头的 GCS_FILE_PATH)
        df = load_large_data_from_gcs(GCS_FILE_PATH, _storage_options) 
        
        # 3. 数据预处理
        df_processed = preprocess_gcs_data(df) # 调用新的数据处理函数
        
        model_data = load_model()
        balanced_sample = get_balanced_sample(df_processed) # 使用处理后的数据
    
    # Store data in session state for access across tabs
    if 'data' not in st.session_state:
        st.session_state.data = {
            'df': df_processed,
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
