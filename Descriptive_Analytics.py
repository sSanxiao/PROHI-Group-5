import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def descriptive_analytics():
    # Use the cached data from the main dashboard
    if 'data' in st.session_state and st.session_state.data['df'] is not None:
        df = st.session_state.data['df']
    else:
        # Fallback to loading data directly if session_state doesn't have it
        DATA_PATH = "./data/cleaned_dataset.csv"
        df = pd.read_csv(DATA_PATH)
        # Assuming SepsisLabel exists
        if 'SepsisLabel' in df.columns:
            df['SepsisLabel'] = df['SepsisLabel'].astype('category')
    

    # Instead of sampling, we'll use the full dataset but focus on patient-level aggregations
    # for visualizations to improve performance while maintaining data integrity
    
    # Create a patient-level summary dataframe for visualizations
    with st.spinner("Preparing patient-level data for analysis..."):
        # Get unique patient IDs
        unique_patient_ids = df['Patient_ID'].unique()
        
        # We'll use the full dataset
        df_sample = df
        
    NUMERICAL_COLS = df.select_dtypes(include=np.number).columns.tolist()
    if 'Patient_ID' in NUMERICAL_COLS:
        NUMERICAL_COLS.remove('Patient_ID') 
    
    # ----------------------------------------------------
    # B. Page Content
    # ----------------------------------------------------
    st.info("This page provides a patient-centric analysis of the temporal dataset, focusing on how measurements change over time and how sepsis develops across patient stays.")
    st.markdown("---")

    # 1. Temporal Data Overview
    st.subheader("1. Temporal Data Overview")
    
    # Calculate patient-level statistics
    with st.spinner("Analyzing temporal data structure..."):
        # Count unique patients
        unique_patients = df['Patient_ID'].nunique()
        
        # Calculate average hours per patient
        hours_per_patient = df.groupby('Patient_ID').size().mean()
        
        # Calculate max hours for any patient
        max_hours = df.groupby('Patient_ID').size().max()
        
        # Get patient distribution by hours recorded
        patient_hours_dist = df.groupby('Patient_ID').size().value_counts().sort_index()
        
    # Display temporal overview
    st.markdown(f"""
    This dataset contains temporal measurements from **{unique_patients:,}** unique patients, with each patient having 
    multiple hourly records. On average, each patient has **{hours_per_patient:.1f}** hours of data, with the longest 
    patient record spanning **{max_hours}** hours.
    
    The temporal nature of this data is critical for sepsis prediction, as it allows us to track how patient 
    measurements change over time before sepsis onset.
    """)

    # 2. Sepsis Prevalence by Patient
    if 'SepsisLabel' in df.columns:
        st.subheader("2. Sepsis Prevalence by Patient")
        
        with st.spinner("Calculating patient-level sepsis statistics..."):
            # For sepsis prevalence, we'll use all patients to get accurate statistics
            # Convert SepsisLabel to numeric before grouping to avoid categorical issues
            if df['SepsisLabel'].dtype.name == 'category':
                df_temp = df.copy()
                df_temp['SepsisLabel'] = df_temp['SepsisLabel'].astype(int)
                # Identify patients who developed sepsis (any record with SepsisLabel=1)
                patient_sepsis = df_temp.groupby('Patient_ID')['SepsisLabel'].max().reset_index()
            else:
                # Identify patients who developed sepsis (any record with SepsisLabel=1)
                patient_sepsis = df.groupby('Patient_ID')['SepsisLabel'].max().reset_index()
            
            # Calculate sepsis prevalence at patient level
            sepsis_patients = patient_sepsis[patient_sepsis['SepsisLabel'] == 1].shape[0]
            non_sepsis_patients = patient_sepsis[patient_sepsis['SepsisLabel'] == 0].shape[0]
            total_patients = sepsis_patients + non_sepsis_patients
            sepsis_percentage = (sepsis_patients / total_patients) * 100
            
            # Create dataframe for visualization
            patient_sepsis_counts = pd.DataFrame({
                'Sepsis Status': ['Sepsis Patients', 'Non-Sepsis Patients'],
                'Count': [sepsis_patients, non_sepsis_patients]
            })
        
        # Display patient-level statistics
        st.markdown(f"""
        Among the **{total_patients:,}** patients in this dataset:
        - **{sepsis_patients:,}** patients (**{sepsis_percentage:.1f}%**) developed sepsis
        - **{non_sepsis_patients:,}** patients (**{(100-sepsis_percentage):.1f}%**) did not develop sepsis
        
        This patient-level analysis is more meaningful than looking at individual records, as each patient has multiple hourly measurements.
        """)

        # Create pie chart with optimized settings
        fig_pie = px.pie(
            patient_sepsis_counts, 
            values='Count', 
            names='Sepsis Status', 
            title='Patient-Level Sepsis Distribution',
            color='Sepsis Status', 
            color_discrete_map={'Sepsis Patients':'red', 'Non-Sepsis Patients':'green'}
        )
        # Optimize the chart for performance
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True, config={"staticPlot": False})
        
        # Add a visualization of sepsis onset timing
        if sepsis_patients > 0:
            st.subheader("2.1 Sepsis Onset Timing")
            
            with st.spinner("Analyzing sepsis onset timing..."):
                # For each sepsis patient, find the first hour when sepsis was detected
                # Handle categorical SepsisLabel if needed
                if df['SepsisLabel'].dtype.name == 'category':
                    sepsis_patients_df = df[df['SepsisLabel'].astype(int) == 1]
                else:
                    sepsis_patients_df = df[df['SepsisLabel'] == 1]
                sepsis_onset = sepsis_patients_df.groupby('Patient_ID')['Hour'].min().reset_index()
                sepsis_onset.columns = ['Patient_ID', 'Onset Hour']
                
                # Create a more beautiful visualization for sepsis onset timing
                # First, bin the onset hours into meaningful clinical intervals
                bins = [0, 6, 12, 24, 48, 72, 96, 120, 144, 168, float('inf')]
                labels = ['0-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72-96h', '96-120h', '120-144h', '144-168h', '168h+']
                
                sepsis_onset['Onset Period'] = pd.cut(sepsis_onset['Onset Hour'], bins=bins, labels=labels, right=False)
                period_counts = sepsis_onset['Onset Period'].value_counts().sort_index().reset_index()
                period_counts.columns = ['Onset Period', 'Number of Patients']
                
                # Calculate cumulative percentage
                total_patients = period_counts['Number of Patients'].sum()
                period_counts['Cumulative Patients'] = period_counts['Number of Patients'].cumsum()
                period_counts['Cumulative Percentage'] = (period_counts['Cumulative Patients'] / total_patients) * 100
                
                # Create a combined bar and line chart
                fig_onset = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add bar chart for patient counts
                fig_onset.add_trace(
                    go.Bar(
                        x=period_counts['Onset Period'],
                        y=period_counts['Number of Patients'],
                        name='Patients',
                        marker_color='rgba(220, 53, 69, 0.7)',
                        hovertemplate='%{y} patients<extra></extra>'
                    ),
                    secondary_y=False
                )
                
                # Add line chart for cumulative percentage
                fig_onset.add_trace(
                    go.Scatter(
                        x=period_counts['Onset Period'],
                        y=period_counts['Cumulative Percentage'],
                        name='Cumulative %',
                        line=dict(color='rgba(0, 123, 255, 1)', width=3),
                        mode='lines+markers',
                        marker=dict(size=8),
                        hovertemplate='%{y:.1f}% of cases<extra></extra>'
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig_onset.update_layout(
                    title='Sepsis Onset Timing Distribution',
                    xaxis_title='Time to Sepsis Onset',
                    yaxis_title='Number of Patients',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=60, b=40),
                    hovermode="x unified",
                    plot_bgcolor='white',
                    bargap=0.2
                )
                
                # Update y-axes titles
                fig_onset.update_yaxes(title_text="Number of Patients", secondary_y=False)
                fig_onset.update_yaxes(title_text="Cumulative Percentage (%)", secondary_y=True)
                
                # Add grid lines for better readability
                fig_onset.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                fig_onset.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                
                st.plotly_chart(fig_onset, use_container_width=True)
                
                # Add clinical interpretation
                st.markdown("""
                **Clinical Interpretation**: This chart shows when sepsis was first detected in patients. 
                The bars represent the number of patients whose sepsis was detected within each time period, 
                while the blue line shows the cumulative percentage of sepsis cases detected over time.
                Early detection is critical for improving patient outcomes.
                """)

    # 3. Temporal Feature Analysis
    st.subheader("3. Temporal Feature Analysis")
    
    # Remove SepsisLabel from feature selection
    feature_cols = [col for col in NUMERICAL_COLS if col != 'SepsisLabel' and col != 'Hour'] 
    
    selected_col = st.selectbox("Select a feature to analyze its temporal trends:", feature_cols)

    if selected_col:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create patient-level aggregation for the selected feature
            with st.spinner(f"Creating patient-level aggregation for {selected_col}..."):
                # Randomly select a subset of patients to improve performance
                random_patient_ids = np.random.choice(
                    df['Patient_ID'].unique(), 
                    size=min(1000, len(df['Patient_ID'].unique())), 
                    replace=False
                )
                
                # Filter data to include only the selected patients
                subset_df = df[df['Patient_ID'].isin(random_patient_ids)]
                
                # For each patient in the subset, calculate mean value of the selected feature
                patient_feature_agg = subset_df.groupby('Patient_ID').agg({
                    selected_col: 'mean',
                    'SepsisLabel': lambda x: 1 if x.astype(int).max() == 1 else 0
                }).reset_index()
                
                # Create violin plot with patient-level data (more visually appealing than histogram)
                fig_violin = px.violin(
                    patient_feature_agg, 
                    x='SepsisLabel', 
                    y=selected_col,
                    color='SepsisLabel',
                    box=True, 
                    points="all",
                    title=f'Distribution of Mean {selected_col} by Patient Sepsis Status',
                    color_discrete_map={0: 'green', 1: 'red'},
                    labels={
                        selected_col: f"Mean {selected_col} per Patient",
                        'SepsisLabel': 'Patient Status'
                    },
                    category_orders={'SepsisLabel': [0, 1]},
                    hover_data={'Patient_ID': True}
                )
                
                # Update x-axis labels
                fig_violin.update_xaxes(
                    ticktext=["Non-Sepsis Patients", "Sepsis Patients"],
                    tickvals=[0, 1]
                )
                
                # Improve styling
                fig_violin.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.1)',
                        title_font=dict(size=12)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.1)',
                        title_font=dict(size=12)
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12
                    ),
                    legend_title_text="Patient Status"
                )
                
                st.plotly_chart(fig_violin, use_container_width=True)
                
                # Add clinical interpretation and note about sampling
                st.markdown(f"""
                **Clinical Interpretation**: This violin plot shows the distribution of the *average* {selected_col} values for a random sample of 1,000 patients, 
                comparing those who developed sepsis (red) with those who did not (green). The width of each violin represents the 
                density of patients at that value, while the box plot inside shows the median and interquartile range. 
                Individual points represent individual patients.
                
                *Note: For performance reasons, this visualization uses a random sample of 1,000 patients.*
                """)
        
        with col2:
            # Temporal trend analysis - show how the feature changes over time
            with st.spinner("Analyzing temporal trends..."):
                if 'Patient_ID' in df.columns:
                    # Get unique patient IDs for selection
                    # Randomly select a subset of patients to improve performance
                    all_patient_ids = sorted(np.random.choice(
                        df['Patient_ID'].unique(), 
                        size=min(1000, len(df['Patient_ID'].unique())), 
                        replace=False
                    ))
                    
                    # Create patient selection options
                    st.markdown("### Patient Selection")
                    st.info(f"For performance reasons, only a random subset of {len(all_patient_ids)} patients is available for selection.")
                    
                    selection_method = st.radio(
                        "How would you like to select patients?",
                        ["Sample of patients", "Specific patient ID"],
                        horizontal=True
                    )
                    
                    if selection_method == "Sample of patients":
                        # Get some sepsis and non-sepsis patients from our subset
                        subset_df = df[df['Patient_ID'].isin(all_patient_ids)]
                        
                        # Limit to 3 patients of each type for better performance and readability
                        if subset_df['SepsisLabel'].dtype.name == 'category':
                            sepsis_patients = subset_df[subset_df['SepsisLabel'].astype(int) == 1]['Patient_ID'].unique()[:3]
                            non_sepsis_patients = subset_df[subset_df['SepsisLabel'].astype(int) == 0]['Patient_ID'].unique()[:3]
                        else:
                            sepsis_patients = subset_df[subset_df['SepsisLabel'] == 1]['Patient_ID'].unique()[:3]
                            non_sepsis_patients = subset_df[subset_df['SepsisLabel'] == 0]['Patient_ID'].unique()[:3]
                        sample_patients = np.concatenate([sepsis_patients, non_sepsis_patients])
                        
                        st.info(f"Showing a sample of {len(sample_patients)} patients (mix of sepsis and non-sepsis)")
                    else:
                        # Allow user to select specific patient ID
                        selected_patient_id = st.selectbox(
                            "Select a patient ID to analyze:",
                            all_patient_ids,
                            format_func=lambda x: f"Patient {x}"
                        )
                        
                        # Check if this patient developed sepsis
                        patient_records = df[df['Patient_ID'] == selected_patient_id]
                        has_sepsis = (patient_records['SepsisLabel'].astype(int).max() == 1)
                        sepsis_status = "developed sepsis" if has_sepsis else "did not develop sepsis"
                        
                        st.info(f"Showing data for Patient {selected_patient_id} who {sepsis_status}")
                        sample_patients = [selected_patient_id]
                    
                    # Filter data for selected patients
                    patient_data = df[df['Patient_ID'].isin(sample_patients)]
                    
                    # Create temporal trend plot with improved styling
                    fig_trend = px.line(
                        patient_data, 
                        x='Hour', 
                        y=selected_col,
                        color='Patient_ID',
                        title=f'Temporal Trend of {selected_col}',
                        labels={'Hour': 'Time (hours)', selected_col: selected_col},
                        line_shape='linear',
                        render_mode='svg'
                    )
                    
                    # Improve the plot styling
                    fig_trend.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        plot_bgcolor='white',
                        legend_title_text='Patient ID',
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            title_font=dict(size=12)
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            title_font=dict(size=12)
                        ),
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=12
                        )
                    )
                    
                    # If we have sepsis patients, mark the onset time
                    if selection_method == "Specific patient ID" and has_sepsis:
                        # Find sepsis onset hour for this patient
                        onset_hour = patient_records[patient_records['SepsisLabel'].astype(int) == 1]['Hour'].min()
                        
                        # Add a vertical line at sepsis onset
                        fig_trend.add_vline(
                            x=onset_hour, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text="Sepsis Onset",
                            annotation_position="top right"
                        )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Add clinical interpretation
                    if selection_method == "Sample of patients":
                        st.markdown("""
                        **Clinical Interpretation**: This chart shows how the selected feature changes over time for a sample of patients.
                        The temporal patterns can reveal important trends leading up to sepsis onset.
                        """)
                    else:
                        if has_sepsis:
                            st.markdown(f"""
                            **Clinical Interpretation**: This chart shows how {selected_col} changes over time for Patient {selected_patient_id}.
                            The dashed red line indicates when sepsis was first detected. Note any significant changes in the measurement
                            leading up to this point, as they may be early indicators of developing sepsis.
                            """)
                        else:
                            st.markdown(f"""
                            **Clinical Interpretation**: This chart shows how {selected_col} changes over time for Patient {selected_patient_id}.
                            This patient did not develop sepsis during their stay. Their measurements can serve as a reference
                            for normal temporal patterns.
                            """)
        
    # 4. Patient Demographics
    st.subheader("4. Patient Demographics")
    
    demo_col1, demo_col2 = st.columns(2)
    
    # Create patient-level demographics dataframe
    with st.spinner("Preparing patient demographics data..."):
        # For age visualization, use a random sample for better performance
        random_patient_ids = np.random.choice(
            df['Patient_ID'].unique(), 
            size=min(1000, len(df['Patient_ID'].unique())), 
            replace=False
        )
        
        # Filter data to include only the selected patients for age visualization
        subset_df = df[df['Patient_ID'].isin(random_patient_ids)]
        
        # For age visualization: use the subset of patients
        age_demographics = subset_df.groupby('Patient_ID').agg({
            'Age': 'first',
            'SepsisLabel': lambda x: 1 if x.astype(int).max() == 1 else 0
        }).reset_index()
        
        # For gender visualization: use all patients
        with st.spinner("Calculating gender statistics for all patients..."):
            gender_demographics = df.groupby('Patient_ID').agg({
                'Gender': 'first',
                'SepsisLabel': lambda x: 1 if x.astype(int).max() == 1 else 0
            }).reset_index()
        
        # Add note about sampling for age only
        st.info("For performance reasons, age distribution visualization uses a random sample of 1,000 patients. Gender distribution uses all patients.")
    
    with demo_col1:
        if 'Age' in df.columns:
            # Create age groups for better visualization
            age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
            
            # Add age group to age demographics
            age_demographics['Age Group'] = pd.cut(
                age_demographics['Age'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
            
            # Count patients by age group and sepsis status
            age_counts = age_demographics.groupby(['Age Group', 'SepsisLabel']).size().reset_index(name='Count')
            
            # Create a grouped bar chart (more visually appealing than histogram)
            fig_age = px.bar(
                age_counts,
                x='Age Group',
                y='Count',
                color='SepsisLabel',
                barmode='group',
                title='Age Distribution by Patient Sepsis Status',
                color_discrete_map={0: 'green', 1: 'red'},
                labels={'Count': 'Number of Patients', 'SepsisLabel': 'Patient Status'},
                category_orders={'Age Group': age_labels}
            )
            
            # Update legend labels
            fig_age.update_traces(
                selector=dict(name='0'),
                name='Non-Sepsis Patients'
            )
            fig_age.update_traces(
                selector=dict(name='1'),
                name='Sepsis Patients'
            )
            
            # Improve styling
            fig_age.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='white',
                xaxis=dict(
                    title='Age Group (years)',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    title_font=dict(size=12)
                ),
                yaxis=dict(
                    title='Number of Patients',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    title_font=dict(size=12)
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12
                ),
                legend_title_text="Patient Status"
            )
            
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Add clinical interpretation
            st.markdown("""
            **Clinical Interpretation**: This chart shows the age distribution of patients by decade, 
            comparing those who developed sepsis (red) with those who did not (green). 
            Age is an important risk factor for sepsis, with older patients generally at higher risk.
            """)
            
    with demo_col2:
        if 'Gender' in df.columns:
            # Pre-aggregate patient-level data for the gender chart (using all patients)
            gender_counts = gender_demographics.groupby(['Gender', 'SepsisLabel']).size().reset_index(name='Count')
            gender_counts['Gender'] = gender_counts['Gender'].map({0: 'Female', 1: 'Male'})
            
            # Calculate total counts and percentages for each gender
            gender_totals = gender_counts.groupby('Gender')['Count'].sum().reset_index()
            gender_counts = gender_counts.merge(gender_totals, on='Gender', suffixes=('', '_total'))
            gender_counts['Percentage'] = (gender_counts['Count'] / gender_counts['Count_total'] * 100).round(1)
            
            # Create a more visually appealing pie chart with subplots
            fig_gender = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=("Female Patients", "Male Patients")
            )
            
            # Add pie charts for each gender
            for i, gender in enumerate(['Female', 'Male']):
                gender_data = gender_counts[gender_counts['Gender'] == gender]
                
                # Create labels with percentages
                labels = [
                    f"Sepsis: {gender_data[gender_data['SepsisLabel'] == 1]['Percentage'].values[0]}%" if 1 in gender_data['SepsisLabel'].values else "Sepsis: 0%",
                    f"Non-Sepsis: {gender_data[gender_data['SepsisLabel'] == 0]['Percentage'].values[0]}%" if 0 in gender_data['SepsisLabel'].values else "Non-Sepsis: 0%"
                ]
                
                values = [
                    gender_data[gender_data['SepsisLabel'] == 1]['Count'].values[0] if 1 in gender_data['SepsisLabel'].values else 0,
                    gender_data[gender_data['SepsisLabel'] == 0]['Count'].values[0] if 0 in gender_data['SepsisLabel'].values else 0
                ]
                
                fig_gender.add_trace(
                    go.Pie(
                        labels=["Sepsis", "Non-Sepsis"],
                        values=values,
                        textinfo='label+percent',
                        hoverinfo='label+value+percent',
                        marker=dict(colors=['red', 'green']),
                        hole=0.4,
                        textfont=dict(size=12),
                        pull=[0.05, 0],
                        name=gender
                    ),
                    row=1, col=i+1
                )
            
            # Add annotations with total counts
            for i, gender in enumerate(['Female', 'Male']):
                total = gender_totals[gender_totals['Gender'] == gender]['Count'].values[0]
                fig_gender.add_annotation(
                    text=f"n = {total}",
                    x=0.5 if i == 0 else 1.5,
                    y=0.5,
                    font=dict(size=14, color="black"),
                    showarrow=False
                )
            
            # Improve styling
            fig_gender.update_layout(
                title='Gender Distribution and Sepsis Prevalence',
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                height=350
            )
            
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Add clinical interpretation with note about using all patients
            st.markdown("""
            **Clinical Interpretation**: These charts show the proportion of sepsis cases within each gender group,
            based on data from **all patients** in the dataset. Gender differences in sepsis prevalence and outcomes 
            have been reported in clinical literature, with some studies suggesting that biological sex may influence 
            immune responses and sepsis pathophysiology.
            """)