import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="📊",
    layout="wide"
)

st.sidebar.image("./assets/project-logo.jpg")

st.title("📊 Data Preprocessing")
st.markdown("Analysis of missing data patterns and imputation strategies for sepsis prediction dataset.")

# Load and display basic dataset info
@st.cache_data
def load_dataset_info():
    try:
        df = pd.read_csv('./data/Dataset.csv')
        return df
    except:
        return None

df = load_dataset_info()

if df is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("Total Columns", df.shape[1])
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")

    # Dataset overview
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.write(f"**Time Range:** {df['Hour'].min()} to {df['Hour'].max()} hours")
    st.write(f"**Patients:** {df['Patient_ID'].nunique():,}")
    
    # Missing data analysis
    st.subheader("Missing Data Analysis")
    
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    # Show top missing columns
    st.write("**Top 15 columns with missing data:**")
    st.dataframe(missing_df.head(15), use_container_width=True)
    
    # Display missing data visualization
    # st.subheader("Missing Data Visualization")
    
    try:
        image = Image.open('./assets/missing_data.png')
        st.image(image, caption="Missing Data Pattern Analysis", use_container_width=True)
    except:
        st.warning("Missing data visualization image not found.")
    
    # Imputation strategy
    st.subheader("Imputation Strategy")
    
    st.info("""
    **Forward-Fill + Backward-Fill Strategy:**
    1. **Forward-fill**: Carry last known values forward (clinical persistence)
    2. **Backward-fill**: Fill early gaps with first measurements  
    3. **Zero-fill**: Only for completely unmeasured variables (rare)
    """)
    
    # Strategy explanation
    with st.expander("Why This Strategy Works for Medical Data"):
        st.write("""
        **Clinical Rationale:**
        - Vital signs persist until new measurements
        - Lab values remain valid for hours/days
        - Missing often means "not retested" rather than "unknown"
        - Preserves temporal relationships crucial for sepsis prediction
        
        **Example:**
        - Hour 0: HR=NaN, Temp=NaN → Backward-fill from Hour 2
        - Hour 1: HR=NaN, Temp=NaN → Backward-fill from Hour 2  
        - Hour 2: HR=80, Temp=98.6 → Original values
        - Hour 3: HR=NaN, Temp=NaN → Forward-fill from Hour 2
        """)
    
    # Sample data transformation
    st.subheader("Sample Data Transformation")
    
    sample_patient = df[df['Patient_ID'] == df['Patient_ID'].iloc[0]].sort_values('Hour').head(8)
    patient_id = sample_patient['Patient_ID'].iloc[0]
    
    st.write(f"**Example Patient:** {patient_id}")
    
    # Column descriptions
    with st.expander("Column Descriptions"):
        st.write("""
        - **Patient_ID**: Unique identifier for each patient
        - **Hour**: Time elapsed since ICU admission (hours)
        - **HR**: Heart Rate (beats per minute)
        - **Temp**: Body Temperature (°F)
        - **SBP**: Systolic Blood Pressure (mmHg)
        - **DBP**: Diastolic Blood Pressure (mmHg)
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Imputation:**")
        st.dataframe(sample_patient[['Patient_ID', 'Hour', 'HR', 'Temp', 'SBP', 'DBP']], use_container_width=True)
    
    with col2:
        st.write("**After Forward-Fill + Backward-Fill:**")
        # Simulate the imputation result
        sample_clean = sample_patient[['Patient_ID', 'Hour', 'HR', 'Temp', 'SBP', 'DBP']].copy()
        sample_clean.iloc[:, 2:] = sample_clean.iloc[:, 2:].ffill().bfill().fillna(0)  # Only impute vital signs, not Patient_ID and Hour
        st.dataframe(sample_clean, use_container_width=True)

else:
    st.error("Dataset not found. Please ensure Dataset.csv is in the data folder.")
    