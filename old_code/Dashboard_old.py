import streamlit as st

st.set_page_config(
    page_title="PROHI Dashboard",
    page_icon="👋",
    layout="wide"
)

st.sidebar.image("./assets/project-logo.jpg")
st.sidebar.success("Select a page above.")

st.write("# Welcome to PROHI Dashboard! 👋")

st.markdown("""
## Alarm System Prevention of Sepsis

This dashboard demonstrates early sepsis prediction for ICU patients using machine learning.

### The Problem
Sepsis is a life-threatening condition that caused 11 million deaths globally in 2017 - nearly 20% of all deaths. In the ICU, mortality reaches 42%. Early detection is critical because for every hour of delayed treatment, mortality risk increases significantly.

### Dataset Overview
**Source:** Prediction of Sepsis (Kaggle)
- **44 columns** with longitudinal clinical data from ICU patients
- **Vital Signs (8):** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- **Lab Values (26):** Blood gas, organ function, metabolic, and hematology markers
- **Demographics (4):** Age, Gender, hospital admission time, ICU length of stay
- **Target:** SepsisLabel (sepsis/no sepsis)

### Project Goals
- **Descriptive:** Analyze sepsis patterns across demographics and vital signs
- **Diagnostic:** Identify most important predictive features
- **Predictive:** Develop early warning system for sepsis detection

### Navigation
1. **Data Preprocessing** - Missing data analysis and imputation strategies
2. **Model Training** - Machine learning approaches and training results
3. **Feature Importance** - Clinical interpretation of predictive features
""")



st.markdown("---")
st.markdown("""
**Team Project - PROHI Course - Group 5**

**Team Members:**
- Max Altez Linhardt
- Khachatur Dallakyan  
- Pratibha Rustogi
- Qilu Wang
- Xue Wu
""")
