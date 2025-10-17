import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler

def get_minimal_sample(df, sample_size=3000, random_seed=69):
    """Get a small stratified sample for quick analysis"""
    if 'SepsisLabel' not in df.columns:
        return df.sample(n=min(sample_size, len(df)), random_state=random_seed)
    
    df_clean = df.dropna(subset=['SepsisLabel'])
    sepsis = df_clean[df_clean['SepsisLabel'] == 1]
    non_sepsis = df_clean[df_clean['SepsisLabel'] == 0]
    
    # Calculate sizes maintaining ratio but with minimum representation
    ratio = len(sepsis) / len(df_clean)
    sepsis_size = max(500, min(int(sample_size * ratio), len(sepsis)))
    non_sepsis_size = min(sample_size - sepsis_size, len(non_sepsis))
    
    # Sample and combine
    sepsis_sample = sepsis.sample(n=sepsis_size, random_state=random_seed)
    non_sepsis_sample = non_sepsis.sample(n=non_sepsis_size, random_state=random_seed)
    
    return pd.concat([sepsis_sample, non_sepsis_sample]).sample(frac=1, random_state=random_seed)

@st.cache_data(ttl=3600)
def normalize_feature(data):
    """Normalize data using StandardScaler"""
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

@st.cache_data(ttl=3600)
def calculate_correlation_stats(df, feature1, feature2, normalize=True):
    """Calculate correlation statistics between two features"""
    # Create copy of the data for normalization
    data = df.copy()
    
    if normalize:
        # Normalize features
        data[feature1] = normalize_feature(data[feature1].values)
        data[feature2] = normalize_feature(data[feature2].values)
    
    # Overall correlation
    overall_corr = data[[feature1, feature2]].corr().iloc[0, 1]
    
    # Correlation by sepsis status
    sepsis_corr = data[data['SepsisLabel'] == 1][[feature1, feature2]].corr().iloc[0, 1]
    non_sepsis_corr = data[data['SepsisLabel'] == 0][[feature1, feature2]].corr().iloc[0, 1]
    
    return {
        'overall': overall_corr,
        'sepsis': sepsis_corr,
        'non_sepsis': non_sepsis_corr
    }

@st.cache_data(ttl=3600)
def create_scatter_with_density(df, feature1, feature2, normalize=True):
    """Create scatter plot with density contours by sepsis status"""
    # Create copy of the data for normalization
    data = df.copy()
    
    if normalize:
        # Normalize features
        data[feature1] = normalize_feature(data[feature1].values)
        data[feature2] = normalize_feature(data[feature2].values)
    
    fig = go.Figure()

    # Add scatter plots for each group
    for label, name, color in [(0, 'No Sepsis', 'blue'), (1, 'Sepsis', 'red')]:
        group_data = data[data['SepsisLabel'] == label]
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=group_data[feature1],
            y=group_data[feature2],
            mode='markers',
            name=name,
            marker=dict(color=color, size=5, opacity=0.5)
        ))

    # Update axis titles to indicate normalization
    x_title = f"{feature1} {'(Normalized)' if normalize else ''}"
    y_title = f"{feature2} {'(Normalized)' if normalize else ''}"

    fig.update_layout(
        title=f'Relationship between {feature1} and {feature2}',
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=500,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def get_clinical_interpretation(corr_stats, feature1, feature2):
    """Generate clinical interpretation based on correlation statistics"""
    overall_corr = abs(corr_stats['overall'])
    sepsis_corr = abs(corr_stats['sepsis'])
    non_sepsis_corr = abs(corr_stats['non_sepsis'])
    
    # Determine correlation strength descriptions
    def get_strength(corr):
        if corr >= 0.7: return "strong"
        elif corr >= 0.4: return "moderate"
        elif corr >= 0.2: return "weak"
        else: return "very weak"
    
    # Determine relationship description based on correlation strength
    def get_relationship_description(corr):
        if abs(corr) > 0.3:
            if corr > 0:
                return "tend to change together"
            else:
                return "tend to change in opposite directions"
        else:
            return "show little consistent relationship and may vary independently"
    
    # Determine if correlation patterns differ between groups
    corr_diff = abs(sepsis_corr - non_sepsis_corr)
    pattern_differs = corr_diff > 0.1
    
    interpretation = f"""
    ### Key Findings in the Relationship Analysis
    
    #### 1. Overall Relationship Pattern
    - The relationship between {feature1} and {feature2} shows a {get_strength(overall_corr)} correlation (r = {corr_stats['overall']:.2f})
    - This suggests that these variables {get_relationship_description(overall_corr)}
    
    #### 2. Sepsis vs Non-Sepsis Patterns
    - In Sepsis cases: {get_strength(sepsis_corr)} correlation (r = {corr_stats['sepsis']:.2f})
    - In Non-Sepsis cases: {get_strength(non_sepsis_corr)} correlation (r = {corr_stats['non_sepsis']:.2f})
    - {'The relationship pattern differs between sepsis and non-sepsis cases' if pattern_differs else 'The relationship pattern is similar in both groups'}
    
    ### Clinical Implications
    
    1. **Diagnostic Value:**
    - {'These variables show different patterns in sepsis vs non-sepsis cases, suggesting potential diagnostic value' if pattern_differs else 'The relationship between these variables remains consistent regardless of sepsis status'}
    - {'The strong correlation in sepsis cases suggests these measurements could be used together for monitoring' if sepsis_corr > 0.6 else 'These variables should be monitored independently'}
    
    2. **Monitoring Recommendations:**
    - {'Consider tracking these variables together as they show significant correlation' if overall_corr > 0.4 else 'Monitor these variables independently as they show limited correlation'}
    - {'Pay special attention to deviations from the expected relationship pattern' if pattern_differs else 'Focus on individual variable ranges rather than their relationship'}
    
    3. **Risk Assessment:**
    - {'Changes in one variable likely indicate changes in the other' if overall_corr > 0.5 else 'Changes in these variables appear largely independent'}
    - {'The relationship between these variables could be useful for early detection' if pattern_differs else 'The relationship between these variables may not be a strong indicator for sepsis'}
    """
    return interpretation

def diagnostic_analytics():
    st.markdown("""
    # 🔬 Diagnostic Analytics: Understanding Variable Relationships
    
    ## Analysis Goals
    This diagnostic analysis aims to:
    1. Identify meaningful relationships between vital signs and clinical measurements
    2. Understand how these relationships differ between sepsis and non-sepsis cases
    3. Discover patterns that could aid in early sepsis detection
    4. Provide clinical insights for monitoring and risk assessment
    
    ## Methodology
    We analyze pairs of variables to:
    - Measure their correlation and statistical relationships
    - Compare patterns between sepsis and non-sepsis cases
    - Identify potential early warning indicators
    - Generate clinical recommendations based on the findings
    """)
    
    # Initialize session state for random seed if not exists
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 69

    # Load and sample data
    if 'data' in st.session_state and st.session_state.data['df'] is not None:
        full_df = st.session_state.data['df']
    else:
        try:
            full_df = pd.read_csv("./data/cleaned_dataset.csv")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Add refresh sample button in a small column
    col1, col2 = st.columns([4, 1])
    with col1:
        st.info(f"Using a sample of 3,000 records from {len(full_df):,} total records for quick analysis while maintaining the sepsis/non-sepsis ratio.")
    with col2:
        if st.button("🔄 New Sample"):
            st.session_state.random_seed = np.random.randint(0, 10000)
            st.rerun()
    
    # Get a small sample for quick analysis using the current random seed
    df = get_minimal_sample(full_df, random_seed=st.session_state.random_seed)
    st.caption(f"Current sample seed: {st.session_state.random_seed}")
    
    # Use only the 17 features used in model training
    FEATURE_COLUMNS = [
        'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
        'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender'
    ]
    
    # Get numerical columns for general analysis
    NUMERICAL_COLS = df.select_dtypes(include=np.number).columns.tolist()
    NUMERICAL_COLS = [col for col in NUMERICAL_COLS if col not in ['Patient_ID', 'SepsisLabel']]
    
    # Feature selection and normalization option
    st.markdown("### Analysis Configuration")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        feature1 = st.selectbox(
            "First Variable:",
            options=NUMERICAL_COLS,
            index=NUMERICAL_COLS.index('HR') if 'HR' in NUMERICAL_COLS else 0
        )
    
    with col2:
        remaining_cols = [col for col in NUMERICAL_COLS if col != feature1]
        feature2 = st.selectbox(
            "Second Variable:",
            options=remaining_cols,
            index=remaining_cols.index('Temp') if 'Temp' in remaining_cols else 0
        )
    
    with col3:
        normalize = st.checkbox("Normalize Variables", value=True,
            help="Standardize variables to have mean=0 and std=1 for better comparison of variables with different scales")
    
    if normalize:
        st.info("Variables are normalized using standard scaling (mean=0, std=1) to enable comparison of different measurement scales.")
    
    # Calculate correlation statistics
    corr_stats = calculate_correlation_stats(df, feature1, feature2, normalize=normalize)
    
    # Create visualization
    st.markdown("### Variable Relationship Visualization")
    fig = create_scatter_with_density(df, feature1, feature2, normalize=normalize)
    st.plotly_chart(fig, use_container_width=True)
    
    if normalize:
        st.markdown("""
        📊 **Note on Normalized Values:**
        - Values are standardized (z-scores) where 0 represents the mean
        - Distance from 0 represents standard deviations from the mean
        - This allows direct comparison between variables with different scales
        - Original patterns and relationships are preserved
        """)
    
    # Display correlation statistics
    st.markdown("### Correlation Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Correlation", f"{corr_stats['overall']:.3f}")
    with col2:
        st.metric("Correlation in Sepsis Cases", f"{corr_stats['sepsis']:.3f}")
    with col3:
        st.metric("Correlation in Non-Sepsis Cases", f"{corr_stats['non_sepsis']:.3f}")
    
    # Display clinical interpretation
    st.markdown(get_clinical_interpretation(corr_stats, feature1, feature2))
    
    # 3D PCA Clustering Visualization
    st.markdown("---")
    st.markdown("### 3D Principal Component Analysis - Patient Clustering")
    st.markdown("""
    This visualization shows how patients cluster in 3-dimensional space based on the 17 features used in model training. 
    The three axes represent the first three principal components (PC1, PC2, PC3), which capture the most important patterns in the data.
    Points are colored by sepsis status to reveal potential clustering patterns.
    """)
    st.info(f"**Features used:** {', '.join(FEATURE_COLUMNS)}")
    
    with st.spinner("Performing PCA analysis and generating 3D visualization..."):
        try:
            # Prepare data for PCA - use only the 17 training features
            available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
            pca_data = df[available_features].dropna()
            pca_labels = df.loc[pca_data.index, 'SepsisLabel'].astype(int)
            
            if len(pca_data) > 100:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # Sample if dataset is too large for performance
                if len(pca_data) > 5000:
                    sample_indices = pca_data.sample(n=5000, random_state=42).index
                    pca_data = pca_data.loc[sample_indices]
                    pca_labels = pca_labels.loc[sample_indices]
                    st.info(f"Showing a sample of 5,000 patients for performance. Total patients: {len(df):,}")
                
                # Standardize the data
                scaler = StandardScaler()
                pca_data_scaled = scaler.fit_transform(pca_data)
                
                # Apply PCA
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(pca_data_scaled)
                
                # Create 3D scatter plot
                fig_3d = go.Figure()
                
                # Add sepsis cases
                sepsis_mask = pca_labels == 1
                if sepsis_mask.any():
                    fig_3d.add_trace(go.Scatter3d(
                        x=pca_result[sepsis_mask, 0],
                        y=pca_result[sepsis_mask, 1],
                        z=pca_result[sepsis_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='#e74c3c',  # Red
                            opacity=0.7,
                            line=dict(width=0.5, color='darkred')
                        ),
                        name='Sepsis',
                        hovertemplate='<b>Sepsis Patient</b><br>' +
                                    'PC1: %{x:.2f}<br>' +
                                    'PC2: %{y:.2f}<br>' +
                                    'PC3: %{z:.2f}<extra></extra>'
                    ))
                
                # Add non-sepsis cases
                non_sepsis_mask = pca_labels == 0
                if non_sepsis_mask.any():
                    fig_3d.add_trace(go.Scatter3d(
                        x=pca_result[non_sepsis_mask, 0],
                        y=pca_result[non_sepsis_mask, 1],
                        z=pca_result[non_sepsis_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='#3498db',  # Blue
                            opacity=0.6,
                            line=dict(width=0.5, color='darkblue')
                        ),
                        name='No Sepsis',
                        hovertemplate='<b>No Sepsis Patient</b><br>' +
                                    'PC1: %{x:.2f}<br>' +
                                    'PC2: %{y:.2f}<br>' +
                                    'PC3: %{z:.2f}<extra></extra>'
                    ))
                
                # Update layout
                fig_3d.update_layout(
                    title={
                        'text': "3D Patient Clustering: Sepsis vs No Sepsis (17 Training Features)",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#2E86AB'}
                    },
                    scene=dict(
                        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                        zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)',
                        bgcolor='white',
                        xaxis=dict(
                            backgroundcolor='white',
                            gridcolor='lightgray',
                            showbackground=True,
                            zerolinecolor='gray'
                        ),
                        yaxis=dict(
                            backgroundcolor='white',
                            gridcolor='lightgray',
                            showbackground=True,
                            zerolinecolor='gray'
                        ),
                        zaxis=dict(
                            backgroundcolor='white',
                            gridcolor='lightgray',
                            showbackground=True,
                            zerolinecolor='gray'
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.3)
                        )
                    ),
                    width=900,
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=60),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='gray',
                        borderwidth=1
                    ),
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                
                # Use SVG renderer instead of WebGL for better browser compatibility
                st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'svg'}})
                
                # Display PCA metrics
                st.markdown("#### Principal Component Variance Explained")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PC1 Variance", f"{pca.explained_variance_ratio_[0]:.1%}",
                             help="Percentage of total variance captured by the first principal component")
                with col2:
                    st.metric("PC2 Variance", f"{pca.explained_variance_ratio_[1]:.1%}",
                             help="Percentage of total variance captured by the second principal component")
                with col3:
                    st.metric("PC3 Variance", f"{pca.explained_variance_ratio_[2]:.1%}",
                             help="Percentage of total variance captured by the third principal component")
                with col4:
                    total_var = pca.explained_variance_ratio_[:3].sum()
                    st.metric("Total Variance", f"{total_var:.1%}",
                             help="Combined variance explained by all three principal components")
                
                # Show top contributing features for each PC
                st.markdown("#### Principal Component Interpretation")
                st.markdown("The features that contribute most to each principal component:")
                
                # Get feature contributions using the 17 training features
                feature_contributions = pd.DataFrame(
                    pca.components_.T,
                    columns=['PC1', 'PC2', 'PC3'],
                    index=available_features
                )
                
                # Display in columns
                pc_cols = st.columns(3)
                for i, (pc_col, pc) in enumerate(zip(pc_cols, ['PC1', 'PC2', 'PC3']), 1):
                    with pc_col:
                        st.markdown(f"**PC{i} Top 5 Features:**")
                        top_features = feature_contributions[pc].abs().nlargest(5)
                        for rank, (feature, contribution) in enumerate(top_features.items(), 1):
                            direction = "↑" if feature_contributions.loc[feature, pc] > 0 else "↓"
                            st.markdown(f"{rank}. {direction} **{feature}** ({abs(contribution):.3f})")
                
                # Interpretation guidance
                st.markdown("""
                #### How to Interpret This Visualization
                
                - **Clustering**: If sepsis and non-sepsis patients form distinct clusters, it suggests they have different patterns in their vital signs
                - **Overlap**: Significant overlap indicates similar physiological patterns between groups
                - **Outliers**: Points far from the main clusters may represent patients with unusual vital sign patterns
                - **Principal Components**: Each PC represents a combination of multiple features that captures data variation
                """)
            
            else:
                st.warning("⚠️ Insufficient data for PCA analysis. At least 100 data points are required.")
        
        except Exception as e:
            st.error(f"Error generating 3D PCA visualization: {str(e)}")
            st.info("This may occur if there are too few numerical features or insufficient data variance.")
    
    # Final notes
        st.markdown("---")
    st.markdown("""
    ### Notes on Interpretation
    
    - Correlation strength does not imply causation
    - Individual patient variations may differ from population patterns
    - These findings should be considered alongside other clinical indicators
    - Regular monitoring of both variables is recommended regardless of correlation strength
    - PCA clustering provides a dimensionality-reduced view of complex multidimensional data
    """)