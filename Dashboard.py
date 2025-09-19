import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
XGBOOST_AVAILABLE = True
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CONFIG = {
    'THRESHOLD_SEPSIS': 0.6,      # Above this = Sepsis
    'THRESHOLD_NO_SEPSIS': 0.3,   # Below this = No Sepsis  
    'MODEL_PATH': 'models/xgboost_model.pkl',
    'LOGO_PATH': './assets/project-logo.jpg',
    'PARLIAMENT_SIZE': 15,
    'CONSENSUS_STRONG': 80,
    'CONSENSUS_MODERATE': 60,
    'TOP_FEATURES_COUNT': 10
}

FEATURE_COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime'
]

FEATURE_CATEGORIES = {
    "Vital Signs": ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2'],
    "Blood Gas": ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2'],
    "Organ Function": ['AST', 'BUN', 'Alkalinephos', 'Creatinine'],
    "Metabolic": ['Calcium', 'Chloride', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium'],
    "Hematology": ['Bilirubin_direct', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'],
    "Demographics": ['Age', 'Gender', 'HospAdmTime']
}

NORMAL_RANGES = {
    'HR': (60, 100), 'O2Sat': (95, 100), 'Temp': (36.1, 37.2), 'SBP': (90, 140), 
    'MAP': (70, 100), 'DBP': (60, 90), 'Resp': (12, 20), 'pH': (7.35, 7.45),
    'Glucose': (70, 140), 'Lactate': (0.5, 2.2), 'WBC': (4.0, 11.0), 'Hct': (38, 52),
    'Hgb': (12, 18), 'Platelets': (150, 450), 'Age': (18, 100),
    'BaseExcess': (-2, 2), 'HCO3': (22, 26), 'FiO2': (21, 100), 'PaCO2': (35, 45),
    'SaO2': (95, 100), 'AST': (10, 40), 'BUN': (7, 20), 'Alkalinephos': (44, 147),
    'Calcium': (8.5, 10.5), 'Chloride': (96, 106), 'Creatinine': (0.6, 1.3),
    'Bilirubin_direct': (0, 0.3), 'Magnesium': (1.7, 2.2), 'Phosphate': (2.5, 4.5),
    'Potassium': (3.5, 5.0), 'Bilirubin_total': (0.2, 1.2), 'TroponinI': (0, 0.04),
    'PTT': (25, 35), 'Fibrinogen': (200, 400), 'EtCO2': (35, 45), 'HospAdmTime': (0, 168)
}

st.set_page_config(
    page_title="PROHI Sepsis Prediction - Parliament of Doctors",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E86AB;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #A23B72;
    margin: 1rem 0;
}
.doctor-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem;
    text-align: center;
    border-left: 5px solid #2E86AB;
}
.sepsis-positive {
    background-color: #ffebee;
    border-left-color: #f44336;
}
.sepsis-negative {
    background-color: #e8f5e8;
    border-left-color: #4caf50;
}
.sepsis-unsure {
    background-color: #fff3e0;
    border-left-color: #ff9800;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load(CONFIG['MODEL_PATH'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_input_constraints(feature):
    """Get very permissive min/max constraints for input fields"""
    if feature in NORMAL_RANGES:
        normal_min, normal_max = NORMAL_RANGES[feature]
        # Very wide ranges to avoid any constraint errors
        min_val = 0.0 if normal_min >= 0 else normal_min * 10.0  # Allow very low values
        max_val = normal_max * 10.0  # Allow very high abnormal values
        return min_val, max_val
    else:
        # Very wide default range for features without defined ranges
        return 0.0, 10000.0

def generate_random_patient_data():
    """Generate realistic random medical values within very wide UI constraints"""
    np.random.seed()  # Ensure randomness
    
    random_data = {}
    
    # Generate values with some probability of being abnormal
    for feature in FEATURE_COLUMNS:
        if feature == 'Gender':
            random_data[feature] = np.random.choice([0, 1])  # 0=Female, 1=Male
            
        elif feature in ['Unit1', 'Unit2']:
            # Unit1 and Unit2 are complementary (Unit1 + Unit2 = 1)
            if feature == 'Unit1':
                random_data[feature] = np.random.choice([0, 1])
            else:  # Unit2
                random_data[feature] = 1 - random_data['Unit1']
                
        elif feature in NORMAL_RANGES:
            normal_min, normal_max = NORMAL_RANGES[feature]
            
            # 70% chance of normal values, 30% chance of abnormal
            if np.random.random() < 0.7:
                # Normal range with some variation
                value = np.random.uniform(normal_min, normal_max)
            else:
                # Abnormal values - much wider range but still reasonable
                if np.random.random() < 0.5:
                    # Below normal - allow very low values
                    low_bound = max(0.1, normal_min * 0.1) if normal_min > 0 else normal_min * 2
                    value = np.random.uniform(low_bound, normal_min)
                else:
                    # Above normal - allow high values but not extreme
                    high_bound = normal_max * 3.0
                    value = np.random.uniform(normal_max, high_bound)
            
            # Round to appropriate decimal places
            if feature in ['HR', 'Age', 'Resp', 'FiO2', 'SBP', 'MAP', 'DBP', 'WBC', 'Platelets', 'Alkalinephos', 'Fibrinogen']:
                random_data[feature] = int(round(value))
            else:
                random_data[feature] = round(value, 2)
        else:
            # Fallback for any remaining features - wider range
            random_data[feature] = round(np.random.uniform(0.1, 500.0), 2)
    
    return random_data

def create_advisory_board_visualization(tree_predictions, tree_scores):
    n_trees = len(tree_predictions)
    threshold_sepsis = CONFIG['THRESHOLD_SEPSIS']
    threshold_no_sepsis = CONFIG['THRESHOLD_NO_SEPSIS']
    
    decisions = []
    colors = []
    
    # For Random Forest: tree_scores are probabilities for class 1 (Sepsis)
    # For XGBoost: tree_scores are raw scores that need interpretation
    for i, score in enumerate(tree_scores):
        if hasattr(score, '__len__') and len(score) > 1:  # Random Forest probability array
            sepsis_prob = score[1]  # Probability of class 1 (Sepsis)
        else:  # Single score (XGBoost or already extracted probability)
            sepsis_prob = float(score)
        
        if sepsis_prob >= threshold_sepsis:
            decisions.append("Sepsis")
            colors.append("#f44336")  # Red
        elif sepsis_prob <= threshold_no_sepsis:
            decisions.append("No Sepsis") 
            colors.append("#4caf50")  # Green
        else:
            decisions.append("Uncertain")
            colors.append("#ff9800")  # Orange
    
    sepsis_count = decisions.count("Sepsis")
    no_sepsis_count = decisions.count("No Sepsis")
    uncertain_count = decisions.count("Uncertain")
    
    angles = np.linspace(0, np.pi, n_trees)
    x = np.cos(angles)
    y = np.sin(angles)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=CONFIG['PARLIAMENT_SIZE'],
            color=colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=[f"AI Advisor {i+1}<br>Decision: {dec}<br>Sepsis Prob: {score[1] if hasattr(score, '__len__') and len(score) > 1 else score:.3f}" 
              for i, (dec, score) in enumerate(zip(decisions, tree_scores))],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text=f"AI Advisory Board ({n_trees} Trees)<br>" +
                 f"<span style='color:#f44336'>Sepsis: {sepsis_count}</span> | " +
                 f"<span style='color:#4caf50'>No Sepsis: {no_sepsis_count}</span> | " +
                 f"<span style='color:#ff9800'>Uncertain: {uncertain_count}</span>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=800,
        height=400,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig, sepsis_count, no_sepsis_count, uncertain_count

def get_individual_tree_predictions(model, X):
    tree_predictions = []
    tree_scores = []
    
    if XGBOOST_AVAILABLE and hasattr(model, 'get_booster'):  # XGBoost model
        dmatrix = xgb.DMatrix(X)
        
        # Get predictions from each tree (boosting rounds)
        for i in range(model.n_estimators):
            tree_score = model.get_booster().predict(dmatrix, iteration_range=(i, i+1))[0]
            # Convert XGBoost logit to probability
            tree_prob = 1 / (1 + np.exp(-tree_score))
            tree_scores.append(tree_prob)
            tree_predictions.append(1 if tree_prob > 0.5 else 0)
    
    else:  # Random Forest or other models
        for tree in model.estimators_:
            pred = tree.predict(X)[0]
            prob = tree.predict_proba(X)[0]
            tree_predictions.append(int(pred))  # Ensure integer: 0 or 1
            tree_scores.append(prob)  # Keep full probability array [prob_class0, prob_class1]
    
    return tree_predictions, tree_scores

def main():
    st.markdown('<h1 class="main-header">🩺 PROHI Sepsis Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI Advisory Board - XGBoost Ensemble for Sepsis Detection</h2>', unsafe_allow_html=True)
    
    st.sidebar.image(CONFIG['LOGO_PATH'], width=200)
    st.sidebar.markdown("## About")
    st.sidebar.info("""
    This dashboard uses an XGBoost model where each tree acts as an "AI advisor" 
    contributing to sepsis diagnosis. The advisory board visualization shows how 
    individual trees vote and their continuous scores combine for the final prediction.
    """)
    
    model = load_model()
    if model is None:
        st.error("Could not load the trained model. Please ensure the model file exists.")
        return
    
    model_type = "XGBoost" if XGBOOST_AVAILABLE and hasattr(model, 'get_booster') else "Random Forest"
    st.success(f"✅ {model_type} model loaded successfully! ({model.n_estimators} trees ready to advise)")
    
    st.markdown('<h3 class="sub-header">📋 Patient Information Input</h3>', unsafe_allow_html=True)
    
    # Add refresh button to generate random values
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        if st.button("🎲 Generate Random Patient", type="secondary", help="Generate random realistic medical values"):
            st.session_state.refresh_values = True
    
    with col_info:
        st.info("💡 Use the refresh button to generate random patient data for testing, or input values manually.")
    
    # Generate random data if refresh button was pressed
    if st.session_state.get('refresh_values', False):
        random_data = generate_random_patient_data()
        st.session_state.update(random_data)
        st.session_state.refresh_values = False
        st.success("🎲 Random patient data generated! Values updated across all tabs.")
        st.rerun()
    
    input_data = {}
    
    tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))
    
    for i, (category, features) in enumerate(FEATURE_CATEGORIES.items()):
        with tabs[i]:
            st.markdown(f"**{category}**")
            
            cols = st.columns(3)
            
            for j, feature in enumerate(features):
                col_idx = j % 3
                
                with cols[col_idx]:
                    # Use session state value if available, otherwise use default
                    session_key = f"input_{feature}"
                    
                    if feature == 'Gender':
                        default_gender = st.session_state.get(feature, 0)
                        value = st.selectbox(
                            f"{feature}", 
                            options=[0, 1], 
                            index=int(default_gender),
                            format_func=lambda x: "Female" if x == 0 else "Male",
                            help="0=Female, 1=Male",
                            key=session_key
                        )
                    elif feature in ['Unit1', 'Unit2']:
                        default_unit = st.session_state.get(feature, 0)
                        value = st.selectbox(
                            f"{feature}", 
                            options=[0, 1], 
                            index=int(default_unit),
                            format_func=lambda x: "No" if x == 0 else "Yes",
                            help=f"Is patient in {feature}?",
                            key=session_key
                        )
                    else:
                        # Get appropriate constraints for this feature
                        min_val, max_val = get_input_constraints(feature)
                        
                        if feature in NORMAL_RANGES:
                            normal_min, normal_max = NORMAL_RANGES[feature]
                            help_text = f"Normal range: {normal_min}-{normal_max} (Input range: {min_val:.1f}-{max_val:.1f})"
                            default_val = (normal_min + normal_max) / 2
                        else:
                            help_text = f"Enter the measured value (Range: {min_val:.1f}-{max_val:.1f})"
                            default_val = min_val
                        
                        # Use random value if available, otherwise use default
                        current_val = st.session_state.get(feature, default_val)
                        
                        # Ensure current value is within constraints
                        current_val = np.clip(float(current_val), min_val, max_val)
                        
                        # Determine appropriate step size
                        if feature in ['HR', 'Age', 'Resp', 'FiO2', 'SBP', 'MAP', 'DBP', 'WBC', 'Platelets', 'Alkalinephos', 'Fibrinogen']:
                            step = 1.0  # Integer values
                        elif max_val > 1000:
                            step = 1.0  # Large ranges use integer steps
                        else:
                            step = 0.1  # Decimal values
                        
                        value = st.number_input(
                            f"{feature}", 
                            min_value=float(min_val), 
                            max_value=float(max_val),
                            value=float(current_val),
                            step=step,
                            help=help_text,
                            key=session_key
                        )
                    
                    input_data[feature] = value
    
    if st.button("🔍 Get Doctor Opinions", type="primary", use_container_width=True):
        X_input = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        X_input = X_input.fillna(0)
        
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        
        tree_preds, tree_scores = get_individual_tree_predictions(model, X_input)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🏛️ AI Advisory Board Decision</h3>', unsafe_allow_html=True)
        
        # Correctly interpret probabilities: probability[1] is probability of class 1 (Sepsis)
        sepsis_prob = probability[1]  # This is the probability of Sepsis (class 1)
        no_sepsis_prob = probability[0]  # This is the probability of No Sepsis (class 0)
        
        if sepsis_prob >= CONFIG['THRESHOLD_SEPSIS']:
            st.error(f"🚨 **SEPSIS RISK DETECTED** (Confidence: {sepsis_prob:.1%})")
        elif sepsis_prob <= CONFIG['THRESHOLD_NO_SEPSIS']:
            st.success(f"✅ **LOW SEPSIS RISK** (Confidence: {no_sepsis_prob:.1%})")
        else:
            st.warning(f"⚠️ **UNCERTAIN PREDICTION** (Sepsis: {sepsis_prob:.1%}, No Sepsis: {no_sepsis_prob:.1%})")
        
        fig, sepsis_votes, no_sepsis_votes, uncertain_votes = create_advisory_board_visualization(
            tree_preds, tree_scores
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="doctor-card sepsis-positive">
                <h4>🔴 Sepsis</h4>
                <h2>{sepsis_votes}</h2>
                <p>advisors</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="doctor-card sepsis-negative">
                <h4>🟢 No Sepsis</h4>
                <h2>{no_sepsis_votes}</h2>
                <p>advisors</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="doctor-card sepsis-unsure">
                <h4>🟡 Uncertain</h4>
                <h2>{uncertain_votes}</h2>
                <p>advisors</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📊 Detailed Analysis</h3>', unsafe_allow_html=True)
        
        # Create probability scores for histogram
        prob_scores = []
        decisions = []
        
        for score in tree_scores:
            if hasattr(score, '__len__') and len(score) > 1:  # Random Forest
                sepsis_prob = score[1]
            else:  # XGBoost or single value
                sepsis_prob = float(score)
            
            prob_scores.append(sepsis_prob)
            
            if sepsis_prob >= CONFIG['THRESHOLD_SEPSIS']:
                decisions.append("Sepsis")
            elif sepsis_prob <= CONFIG['THRESHOLD_NO_SEPSIS']:
                decisions.append("No Sepsis")
            else:
                decisions.append("Uncertain")
        
        score_df = pd.DataFrame({
            'Advisor': [f"Tree {i+1}" for i in range(len(tree_scores))],
            'Sepsis_Probability': prob_scores,
            'Decision': decisions
        })
        
        fig_hist = px.histogram(
            score_df, 
            x='Sepsis_Probability', 
            color='Decision',
            nbins=20,
            title="Distribution of Tree Predictions (Sepsis Probability)",
            color_discrete_map={
                'Sepsis': '#f44336',
                'No Sepsis': '#4caf50',
                'Uncertain': '#ff9800'
            }
        )
        fig_hist.add_vline(x=CONFIG['THRESHOLD_SEPSIS'], line_dash="dash", line_color="red", 
                          annotation_text=f"Sepsis Threshold ({CONFIG['THRESHOLD_SEPSIS']})")
        fig_hist.add_vline(x=CONFIG['THRESHOLD_NO_SEPSIS'], line_dash="dash", line_color="green", 
                          annotation_text=f"No Sepsis Threshold ({CONFIG['THRESHOLD_NO_SEPSIS']})")
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Prediction-specific feature importance using SHAP-like approach or input values
        st.markdown("### 🎯 Key Features for This Prediction")
        
        # Create feature importance based on input values and their deviation from normal ranges
        feature_importance_scores = []
        for i, feature in enumerate(FEATURE_COLUMNS):
            input_value = input_data[feature]
            
            # Calculate importance based on deviation from normal range
            if feature in NORMAL_RANGES:
                min_val, max_val = NORMAL_RANGES[feature]
                normal_center = (min_val + max_val) / 2
                deviation = abs(input_value - normal_center) / (max_val - min_val)
                importance = min(deviation, 1.0)  # Cap at 1.0
            else:
                # For non-numeric features, use a simple scoring
                importance = 0.5 if input_value != 0 else 0.1
            
            feature_importance_scores.append(importance)
        
        importance_df = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Input_Value': [input_data[f] for f in FEATURE_COLUMNS],
            'Deviation_Score': feature_importance_scores
        }).sort_values('Deviation_Score', ascending=False).head(CONFIG['TOP_FEATURES_COUNT'])
        
        fig_importance = px.bar(
            importance_df, 
            x='Deviation_Score', 
            y='Feature',
            orientation='h',
            title=f"Top {CONFIG['TOP_FEATURES_COUNT']} Most Influential Features for This Patient",
            hover_data=['Input_Value']
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🩺 Clinical Interpretation</h3>', unsafe_allow_html=True)
        
        total_decisive = sepsis_votes + no_sepsis_votes
        consensus_pct = max(sepsis_votes, no_sepsis_votes) / len(tree_preds) * 100 if len(tree_preds) > 0 else 0
        
        if consensus_pct >= CONFIG['CONSENSUS_STRONG']:
            consensus_level = "Strong"
            consensus_color = "green" if no_sepsis_votes > sepsis_votes else "red"
        elif consensus_pct >= CONFIG['CONSENSUS_MODERATE']:
            consensus_level = "Moderate"
            consensus_color = "orange"
        else:
            consensus_level = "Weak"
            consensus_color = "gray"
        
        # Calculate average probability
        avg_sepsis_prob = np.mean([score[1] if hasattr(score, '__len__') and len(score) > 1 else score for score in tree_scores])
        
        st.markdown(f"""
        **Consensus Analysis:**
        - <span style="color:{consensus_color}">**{consensus_level} Consensus**</span> ({consensus_pct:.1f}% agreement)
        - Trees voting Sepsis: **{sepsis_votes}**
        - Trees voting No Sepsis: **{no_sepsis_votes}**
        - Trees uncertain: **{uncertain_votes}**
        - Average sepsis probability: **{avg_sepsis_prob:.1%}**
        - Final model prediction: **{sepsis_prob:.1%}** (Sepsis), **{no_sepsis_prob:.1%}** (No Sepsis)
        
        **Clinical Recommendation:**
        """, unsafe_allow_html=True)
        
        if sepsis_prob >= CONFIG['THRESHOLD_SEPSIS']:
            if consensus_pct >= 70:
                st.error("🚨 **HIGH PRIORITY**: Strong consensus for sepsis risk. Immediate clinical evaluation and intervention recommended.")
            else:
                st.warning("⚠️ **MODERATE PRIORITY**: Sepsis risk detected with some uncertainty. Close monitoring and clinical assessment advised.")
        elif sepsis_prob <= CONFIG['THRESHOLD_NO_SEPSIS']:
            if consensus_pct >= 70:
                st.success("✅ **LOW PRIORITY**: Strong consensus for low sepsis risk. Continue routine monitoring per protocol.")
            else:
                st.info("ℹ️ **ROUTINE MONITORING**: Low sepsis risk indicated but maintain standard care vigilance.")
        else:
            st.warning(f"🤔 **UNCERTAIN PREDICTION**: Model is uncertain (Sepsis: {sepsis_prob:.1%}). Consider additional clinical assessment, laboratory tests, and expert consultation. Monitor closely for clinical deterioration.")

if __name__ == "__main__":
    main()