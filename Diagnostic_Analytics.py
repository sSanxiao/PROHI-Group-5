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
    
    # Determine if correlation patterns differ between groups
    corr_diff = abs(sepsis_corr - non_sepsis_corr)
    pattern_differs = corr_diff > 0.1
    
    interpretation = f"""
    ### Key Findings in the Relationship Analysis
    
    #### 1. Overall Relationship Pattern
    - The relationship between {feature1} and {feature2} shows a {get_strength(overall_corr)} correlation (r = {corr_stats['overall']:.2f})
    - This suggests that these variables {'tend to change together' if overall_corr > 0 else 'tend to change in opposite directions'}
    
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
    
    # Get numerical columns
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
    
    # Final notes
    st.markdown("""
    ### Notes on Interpretation
    
    - Correlation strength does not imply causation
    - Individual patient variations may differ from population patterns
    - These findings should be considered alongside other clinical indicators
    - Regular monitoring of both variables is recommended regardless of correlation strength
    """)