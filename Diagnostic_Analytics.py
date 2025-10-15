import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

def diagnostic_analytics():
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
        df_sample = df.sample(n=5000, random_state=42)
        st.info(f"Dataset is large ({df.shape[0]} records). Using a sample of 5,000 records for visualizations.")
    else:
        df_sample = df

    NUMERICAL_COLS = df.select_dtypes(include=np.number).columns.tolist()
    if 'Patient_ID' in NUMERICAL_COLS:
        NUMERICAL_COLS.remove('Patient_ID') 

    # ----------------------------------------------------
    # B. Page Content
    # ----------------------------------------------------
    st.info("This page aims to explore relationships between variables, identifying factors most correlated with Sepsis onset to answer the 'why'.")
    st.markdown("---")

    # 1. Correlation Heatmap
    if 'SepsisLabel' in df.columns:
        st.subheader("1. Variable Correlation Heatmap")
        st.markdown("The heatmap displays linear relationships between numerical variables, sorted by correlation with `SepsisLabel`.")

        # Calculate correlation on a sample for performance
        with st.spinner("Calculating correlations..."):
            # Use only the most relevant columns for correlation to improve performance
            # Select top 20 columns by variance
            top_cols = df_sample[NUMERICAL_COLS].var().nlargest(20).index.tolist()
            if 'SepsisLabel' not in top_cols:
                top_cols.append('SepsisLabel')
                
            corr_matrix = df_sample[top_cols].corr()
            corr_to_sepsis = corr_matrix.sort_values(by='SepsisLabel', ascending=False)
        
        # Matplotlib/Seaborn generate heatmap
        fig, ax = plt.subplots(figsize=(14, 12)) 
        sns.heatmap(
            corr_to_sepsis, 
            annot=True, 
            fmt=".2f", 
            cmap='vlag', 
            ax=ax, 
            linewidths=.5, 
            linecolor='black', 
            annot_kws={"size": 8}
        ) 
        
        plt.title('Numerical Variable Correlation Matrix (Sorted by SepsisLabel Correlation)', fontsize=14)
        plt.tight_layout() 
        st.pyplot(fig)


        # 2. Inter-group Difference Comparison (Diagnostic Clues)
        st.subheader("2. Key Metric Group Differences (Diagnostic Clues)")
        st.markdown("Box plots are used to visually compare the statistical distribution of core metrics between Sepsis-positive and Sepsis-negative groups.")

        # Allow user to select features for comparison
        key_features = st.multiselect(
            "Select key metrics for comparison (Max 3):", 
            options=[col for col in NUMERICAL_COLS if col != 'SepsisLabel'], 
            default=['HR', 'Temp', 'MAP'] if all(c in NUMERICAL_COLS for c in ['HR', 'Temp', 'MAP']) else ([c for c in NUMERICAL_COLS if c != 'SepsisLabel'][:3])
        )

        cols = st.columns(len(key_features) if len(key_features) > 0 else 1)
        
        for i, feature in enumerate(key_features):
            with cols[i % len(cols)]:
                # Use the sampled dataframe for box plots
                fig_box = px.box(
                    df_sample, 
                    x='SepsisLabel', 
                    y=feature, 
                    color='SepsisLabel', 
                    title=f'{feature} Distribution Comparison',
                    color_discrete_map={'0':'green', '1':'red'},
                    points='outliers'  # Only show outliers as points for better performance
                )
                fig_box.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_box, use_container_width=True)
        
        # 3. Pair Plot Analysis
        st.subheader("3. Pair Plot Analysis")
        st.markdown("Analyze relationships between selected variables with pair plots.")
        
        # Allow user to select features for pair plot
        pair_features = st.multiselect(
            "Select features for pair plot analysis (2-4 recommended):", 
            options=[col for col in NUMERICAL_COLS if col != 'SepsisLabel'],
            default=['HR', 'Temp'] if all(c in NUMERICAL_COLS for c in ['HR', 'Temp']) else ([c for c in NUMERICAL_COLS if c != 'SepsisLabel'][:2])
        )
        
        if len(pair_features) >= 2:
            # Create pair plot with selected features plus SepsisLabel
            # Limit to 1000 samples for pair plot to improve performance
            with st.spinner("Generating pair plot..."):
                plot_data = df_sample[pair_features + ['SepsisLabel']].sample(min(1000, len(df_sample)))
                
                # Use a more efficient approach for pair plots
                fig, ax = plt.subplots(figsize=(12, 10))
                pair_plot = sns.pairplot(
                    plot_data, 
                    hue='SepsisLabel', 
                    palette={0: 'green', 1: 'red'},
                    plot_kws={'alpha': 0.6, 's': 15},  # Smaller points, more transparent
                    diag_kind='kde',
                    corner=True  # Only show lower triangle for fewer plots
                )
                plt.tight_layout()
                st.pyplot(pair_plot.fig)
        else:
            st.warning("Please select at least 2 features for the pair plot.")
        
        # 4. Conclusion Section
        st.markdown("---")
        st.subheader("📝 Diagnostic Analysis Conclusion")
        st.success("Based on the analyses above, the following preliminary conclusions can be drawn:")
        st.markdown(f"""
        1.  **Strongly Correlated Variable**: According to the heatmap, the variable most correlated with the SepsisLabel is **{corr_to_sepsis.index[1]}** (Correlation: `{corr_to_sepsis.iloc[1]['SepsisLabel']:.2f}`).
        2.  **Key Metric Differences**:
            * **Heart Rate (HR)**: The median/variance of HR in the Sepsis-positive group appears to be significantly **higher** than in the non-Sepsis group, consistent with clinical tachycardia.
            * **Temperature (Temp)**: The Temp distribution in the Sepsis-positive group might be **more dispersed**, suggesting cases of both hyperthermia (fever) and hypothermia (shock).
            * **Mean Arterial Pressure (MAP)**: The median MAP in the Sepsis group may be **lower**, indicating a risk of shock.
        3.  **Clinical Implications**: The pair plot analysis reveals important interactions between vital signs that could help in early detection of sepsis risk.
        """)
    else:
        st.warning("The 'SepsisLabel' column is missing from the dataset. Diagnostic analysis cannot be performed.")