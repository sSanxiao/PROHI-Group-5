import streamlit as st

def about_page():
    st.info("This page provides information about the sepsis prediction project, dataset, team members, and references.")
    st.markdown("---")
    
    st.subheader("About the Project")
    st.markdown("""
    The ASPoS aims to develop an early warning decision support system to assist healthcare professionals in the intensive care unit (ICU) with the timely identification of patients at risk of sepsis. Sepsis is a severe and life-threatening condition with high mortality rates and significant healthcare costs [1]. The project utilizes machine learning techniques to continuously analyze vital signs and laboratory results, thereby identifying subtle patterns that may indicate the early stages of sepsis before they become clinically evident [2].
    
    The dashboard includes:
    
    1. **Descriptive Analytics**: Explore and understand the dataset through visualizations and summaries
    2. **Diagnostic Analytics**: Analyze relationships between variables and identify factors correlated with sepsis
    3. **Predictive Analytics**: Generate sepsis risk predictions for individual patients using a pre-trained model, based on vital sign information and the current hour
    4. **Prescriptive Analytics**: Understand model predictions through SHAP values and get clinical recommendations
    
    This tool is intended to support clinical decision-making, not replace it. All predictions should be considered in the context of clinical expertise and patient-specific factors.
    """)
    
    st.subheader("Dataset Information")
    st.markdown("""
    ### Sepsis Prediction Dataset
    
    **Source**: [Kaggle - Prediction of Sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)
    
    **Description**:  
    This dataset contains patient measurements from the ICU of two separate hospitals. The data has been anonymized, and the task is to predict sepsis 6 hours before the clinical prediction.
    
    **Features**:
    - Vital signs (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2)
    - Laboratory values (BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, etc.)
    - Demographics (Age, Gender)
    - ICU length of stay (ICULOS)
    
    **Target Variable**:
    - SepsisLabel: Indicates the onset of sepsis according to Sepsis-3 clinical criteria
    
    The data has been cleaned and preprocessed for this application.
    """)
    
    st.subheader("Team Members")
    
    # Create columns for team members
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Max Altez Linhardt**
        - **Khachatur Dallakyan**
        """)
    
    with col2:
        st.markdown("""
        - **Pratibha Rustogi**
        - **Qilu Wang**
        - **Xue Wu**
        """)
    
    st.subheader("References")
    st.markdown("""
    1. Fleischmann-Struzek C, Mellhammar L, Rose N, Cassini A, Rudd KE, Schlattmann P, et al. Incidence and mortality of hospital- and ICU-treated sepsis: results from an updated and expanded systematic review and meta-analysis. Intensive Care Med. 2020;46(8):1552-1562. doi: 10.1007/s00134-020-06151-x.
    
    2. Rawat S, Shanmugam H, Airen L. Machine Learning and Deep Learning Models for Early Sepsis Prediction: A Scoping Review. Indian journal of critical care medicine. 2025;29(6):516–524. doi: 10.5005/jp-journals-10071-24986
    
    3. Kaggle Dataset: [Prediction of Sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)
    """)
    
    st.subheader("Contact Information")
    st.markdown("""
    For questions, feedback, or support, please contact any of our team members:
    
    - **Xue Wu**: wu.xue@stud.ki.se
    - **Khachatur Dallakyan**: khachatur.dallakyan@stud.ki.se
    - **Qilu Wang**: qilu.wang@stud.ki.se
    - **Pratibha Rustogi**: rustogi.pratibha@gmail.com
    - **Max Altez Linhardt**: Altez.power@gmail.com
    
    **Project Repository**: [GitHub - PROHI-Group-5](https://github.com/sSanxiao/PROHI-Group-5)
    """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        © 2025 PROHI Group 5. All rights reserved.<br>
        This project was developed for educational purposes only.
    </div>
    """, unsafe_allow_html=True)
