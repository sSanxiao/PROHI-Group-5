import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

def descriptive_analytics():
    # Use the cached data from the main dashboard
    if 'data' in st.session_state and st.session_state.data['df'] is not None:
        df = st.session_state.data['df']
    else:
        # Fallback to loading data directly if session_state doesn't have it
        DATA_PATH = "./data/cleaned_dataset.csv"
        try:
            df = pd.read_csv(DATA_PATH)
            # Assuming SepsisLabel exists
            if 'SepsisLabel' in df.columns:
                df['SepsisLabel'] = df['SepsisLabel'].astype('category')
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Sample data for faster visualization if dataset is large
    if df.shape[0] > 10000:
        df_sample = df.sample(n=10000, random_state=42)
        st.info(f"Dataset is large ({df.shape[0]} records). Using a sample of 10,000 records for visualizations.")
    else:
        df_sample = df
        
    NUMERICAL_COLS = df.select_dtypes(include=np.number).columns.tolist()
    if 'Patient_ID' in NUMERICAL_COLS:
        NUMERICAL_COLS.remove('Patient_ID') 
    
    # ----------------------------------------------------
    # B. Page Content
    # ----------------------------------------------------
    st.info(f"This page provides a summary overview and distribution analysis of the dataset.")
    st.markdown("---")

    # 1. Basic Statistics
    st.subheader("1. Data Statistical Summary")
    st.markdown(f"**Total Records:** {df.shape[0]} | **Total Features:** {df.shape[1]}")
    
    # Use a more efficient approach for descriptive statistics
    with st.spinner("Calculating statistics..."):
        stats_df = df[NUMERICAL_COLS].describe().T
    st.dataframe(stats_df)

    # 2. Sepsis Prevalence
    if 'SepsisLabel' in df.columns:
        st.subheader("2. Sepsis Label Distribution")
        
        # Calculate counts once and reuse
        sepsis_counts = df['SepsisLabel'].value_counts().reset_index()
        sepsis_counts.columns = ['Sepsis Status', 'Count']
        sepsis_counts['Sepsis Status'] = sepsis_counts['Sepsis Status'].map({0: 'Non-Sepsis', 1: 'Sepsis'})

        # Create pie chart with optimized settings
        fig_pie = px.pie(
            sepsis_counts, 
            values='Count', 
            names='Sepsis Status', 
            title='Sepsis Label Distribution',
            color='Sepsis Status', 
            color_discrete_map={'Sepsis':'red', 'Non-Sepsis':'green'}
        )
        # Optimize the chart for performance
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True, config={"staticPlot": False})

    # 3. Variable Distribution (Interactive)
    st.subheader("3. Individual Feature Distribution Exploration")
    
    # Remove SepsisLabel from feature selection
    feature_cols = [col for col in NUMERICAL_COLS if col != 'SepsisLabel'] 
    
    selected_col = st.selectbox("Select a variable to view its distribution:", feature_cols)

    if selected_col:
        # Use the sampled dataframe for histogram to improve performance
        fig_hist = px.histogram(
            df_sample, 
            x=selected_col, 
            color='SepsisLabel',
            marginal="box", 
            title=f'Distribution of {selected_col}, Grouped by Sepsis Status',
            barmode='overlay', 
            opacity=0.7,
            nbins=30  # Limit number of bins for better performance
        )
        # Optimize the chart
        fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # 4. Patient Demographics
    st.subheader("4. Patient Demographics")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        if 'Age' in df.columns:
            # Use sampled data for age histogram
            fig_age = px.histogram(
                df_sample, 
                x='Age', 
                color='SepsisLabel',
                title='Age Distribution by Sepsis Status',
                barmode='overlay', 
                opacity=0.7,
                color_discrete_map={0: 'green', 1: 'red'},
                nbins=20  # Fewer bins for better performance
            )
            fig_age.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_age, use_container_width=True)
            
    with demo_col2:
        if 'Gender' in df.columns:
            # Pre-aggregate data for the gender chart to improve performance
            gender_counts = df.groupby(['Gender', 'SepsisLabel']).size().reset_index(name='Count')
            gender_counts['Gender'] = gender_counts['Gender'].map({0: 'Female', 1: 'Male'})
            
            fig_gender = px.bar(
                gender_counts, 
                x='Gender', 
                y='Count', 
                color='SepsisLabel',
                title='Gender Distribution by Sepsis Status',
                barmode='group',
                color_discrete_map={0: 'green', 1: 'red'}
            )
            fig_gender.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gender, use_container_width=True)