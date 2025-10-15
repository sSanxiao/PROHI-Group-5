import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap
import warnings
warnings.filterwarnings('ignore')

def prescriptive_analytics():
    st.info("This page provides explanations of the sepsis prediction model using SHAP (SHapley Additive exPlanations) values, helping clinicians understand which factors contribute most to sepsis risk.")
    st.markdown("---")
    
    # Use cached data from the main dashboard if available
    if 'data' in st.session_state:
        df = st.session_state.data['df']
        model_data = st.session_state.data['model_data']
        
        # Determine if we need demo mode
        demo_mode = (df is None or model_data is None)
    else:
        # Fallback to checking files directly
        model_path = './models/random_forest.pkl'
        data_path = './data/cleaned_dataset.csv'
        
        # Check if model exists
        if not os.path.exists(model_path):
            st.warning("Model file not found. Using demo mode with simulated data.")
            demo_mode = True
        else:
            demo_mode = False
        
        # Check if data exists
        if not os.path.exists(data_path):
            st.warning("Dataset file not found. Using demo mode with simulated data.")
            demo_mode = True
    
    if demo_mode:
        # Create demo data and model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Define feature names
        features = [
            'HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'FiO2', 'AST', 'BUN',
            'Creatinine', 'Glucose', 'Lactate', 'WBC', 'Age', 'Gender'
        ]
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
        y = (X['HR'] > 0.5) & (X['Temp'] > 0.5) | (X['Lactate'] > 1.0)
        y = y.astype(int)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Create scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a sample for individual explanation
        sample_idx = 0
        sample = X.iloc[sample_idx:sample_idx+1]
        
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Package data for display
        data = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'X': X,
            'y': y,
            'sample': sample,
            'explainer': explainer,
            'shap_values': shap_values
        }
    else:
        try:
            # Use cached data if available
            if 'data' in st.session_state and st.session_state.data['model_data'] is not None:
                # Get model from cached data
                model_data = st.session_state.data['model_data']
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                features = model_data.get('features')
                
                # Get dataset from cached data
                df = st.session_state.data['df']
            else:
                # Fallback to loading directly from files
                model_data = joblib.load(model_path)
                model = model_data['model']
                scaler = model_data['scaler']
                features = model_data['features']
                
                # Load a sample of data
                df = pd.read_csv(data_path)
                
                # Drop unnecessary columns
                columns_drop = {
                    'Unnamed: 0', 'Unit1', 'Unit2'
                }
                df = df.drop(columns=[col for col in columns_drop if col in df.columns])
            
            # Split into features and target
            X = df.drop(['Patient_ID', 'SepsisLabel'], axis=1, errors='ignore')
            y = df['SepsisLabel'] if 'SepsisLabel' in df.columns else None
            
            # Sample data for SHAP analysis (limit to 100 samples for performance)
            sample_size = min(100, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            
            with st.spinner("Computing SHAP values... This may take a moment."):
                # Create a SHAP explainer
                explainer = shap.TreeExplainer(model)
                X_sample_scaled = scaler.transform(X_sample)
                shap_values = explainer.shap_values(X_sample_scaled)
            
            # Select a sample for individual explanation
            sepsis_samples = df[df['SepsisLabel'] == 1] if 'SepsisLabel' in df.columns else df.sample(1)
            sample_idx = sepsis_samples.index[0]
            sample = X.loc[sample_idx:sample_idx].reset_index(drop=True)
            
            # Package data for display
            data = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'X': X_sample,
                'y': y.loc[X_sample.index] if y is not None else None,
                'sample': sample,
                'explainer': explainer,
                'shap_values': shap_values
            }
        except Exception as e:
            st.error(f"Error loading model or data: {e}")
            st.warning("Switching to demo mode with simulated data.")
            
            # Create demo data and model (same as above)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            features = [
                'HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'FiO2', 'AST', 'BUN',
                'Creatinine', 'Glucose', 'Lactate', 'WBC', 'Age', 'Gender'
            ]
            
            np.random.seed(42)
            n_samples = 1000
            X = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
            y = (X['HR'] > 0.5) & (X['Temp'] > 0.5) | (X['Lactate'] > 1.0)
            y = y.astype(int)
            
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X, y)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            sample_idx = 0
            sample = X.iloc[sample_idx:sample_idx+1]
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            data = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'X': X,
                'y': y,
                'sample': sample,
                'explainer': explainer,
                'shap_values': shap_values
            }
    
    # Display SHAP analysis
    st.subheader("1. Global Feature Importance")
    st.markdown("""
    This plot shows which features are most important for predicting sepsis across all patients.
    Features are ranked by their average impact on model predictions.
    """)
    
    # Create and display the summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle the SHAP values carefully to avoid index errors
    try:
        # If shap_values is a list (for multi-class models), use the first element
        if isinstance(data['shap_values'], list) and len(data['shap_values']) > 0:
            shap_values_to_plot = data['shap_values'][1]  # Use class 1 (sepsis)
        else:
            shap_values_to_plot = data['shap_values']
        
        # Make sure the feature names match the shape of shap_values
        feature_names = list(data['X'].columns)
        if shap_values_to_plot.shape[1] != len(feature_names):
            # If shapes don't match, create generic feature names
            feature_names = [f"Feature {i}" for i in range(shap_values_to_plot.shape[1])]
        
        # Create a simple bar plot of mean absolute SHAP values
        shap_importance = np.abs(shap_values_to_plot).mean(0)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': shap_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot using matplotlib directly to avoid SHAP API issues
        plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance (based on SHAP values)')
    except Exception as e:
        st.warning(f"Could not create SHAP bar plot: {e}")
        plt.text(0.5, 0.5, "SHAP plot unavailable", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretation:**
    - Features at the top have the greatest impact on model predictions
    - The magnitude (length of bar) indicates how much a feature influences the prediction on average
    - This helps clinicians understand which vital signs and lab values are most predictive of sepsis
    """)
    
    st.subheader("2. Feature Impact Distribution")
    st.markdown("""
    This plot shows how each feature affects predictions across all patients.
    - Red points indicate high feature values
    - Blue points indicate low feature values
    - Position on x-axis shows whether the feature increases risk (right) or decreases risk (left)
    """)
    
    # Create and display the summary plot with all points
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Handle the SHAP values carefully to avoid index errors
    try:
        # If shap_values is a list (for multi-class models), use the first element
        if isinstance(data['shap_values'], list) and len(data['shap_values']) > 0:
            shap_values_to_plot = data['shap_values'][1]  # Use class 1 (sepsis)
        else:
            shap_values_to_plot = data['shap_values']
        
        # Make sure the feature names match the shape of shap_values
        feature_names = list(data['X'].columns)
        if shap_values_to_plot.shape[1] != len(feature_names):
            # If shapes don't match, create generic feature names
            feature_names = [f"Feature {i}" for i in range(shap_values_to_plot.shape[1])]
        
        # Create a simpler version of the SHAP summary plot
        # First, get the feature importance order
        shap_importance = np.abs(shap_values_to_plot).mean(0)
        feature_order = np.argsort(-shap_importance)
        
        # Get the top 10 features
        top_features = feature_order[:10]
        
        # For each feature, create a scatter plot of SHAP values
        for i, feature_idx in enumerate(top_features):
            feature_name = feature_names[feature_idx]
            feature_shap = shap_values_to_plot[:, feature_idx]
            feature_value = data['X'].iloc[:, feature_idx] if hasattr(data['X'], 'iloc') else data['X'][:, feature_idx]
            
            # Normalize feature values for coloring
            norm_values = (feature_value - feature_value.min()) / (feature_value.max() - feature_value.min() + 1e-10)
            
            plt.subplot(10, 1, i+1)
            plt.scatter(feature_shap, np.ones(len(feature_shap)) * i, c=norm_values, 
                      cmap='coolwarm', alpha=0.8, s=20)
            plt.yticks([i], [feature_name])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            
            if i == 9:  # Only add x-label to the bottom plot
                plt.xlabel('SHAP value (impact on model output)')
        
        plt.tight_layout()
    except Exception as e:
        st.warning(f"Could not create SHAP beeswarm plot: {e}")
        plt.text(0.5, 0.5, "SHAP plot unavailable", ha='center', va='center', fontsize=14)
    
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretation:**
    - Scattered points show how feature values affect individual predictions
    - For example, high lactate values (red points) typically push predictions toward sepsis (positive SHAP value)
    - This visualization helps understand the range and direction of each feature's impact
    """)
    
    # Individual patient analysis
    st.subheader("3. Individual Patient Analysis")
    
    # Let user select a sample patient or use a random one
    if st.button("Select Random Patient"):
        if demo_mode:
            sample_idx = np.random.randint(0, len(data['X']))
            sample = data['X'].iloc[sample_idx:sample_idx+1]
        else:
            sepsis_samples = df[df['SepsisLabel'] == 1] if 'SepsisLabel' in df.columns else df
            sample_idx = sepsis_samples.sample(1).index[0]
            sample = X.loc[sample_idx:sample_idx].reset_index(drop=True)
        
        data['sample'] = sample
    
    st.markdown("### Patient Feature Values")
    
    # Display the patient's feature values
    feature_values = pd.DataFrame({
        'Feature': data['sample'].columns,
        'Value': data['sample'].values[0]
    })
    st.dataframe(feature_values.style.background_gradient(cmap='coolwarm', subset=['Value']))
    
    # Calculate SHAP values for this sample
    sample_scaled = scaler.transform(data['sample'])
    sample_shap_values = explainer.shap_values(sample_scaled)
    
    if isinstance(sample_shap_values, list):
        # For multi-class models, use class 1 (sepsis)
        sample_shap_values = sample_shap_values[1] if len(sample_shap_values) > 1 else sample_shap_values[0]
    
    # Calculate prediction
    prediction = model.predict_proba(sample_scaled)[0]
    sepsis_prob = prediction[1] if len(prediction) > 1 else prediction[0]
    
    # Display prediction
    st.markdown(f"### Sepsis Risk: {sepsis_prob:.1%}")
    
    # Force plot for individual prediction
    st.markdown("### Feature Contribution Analysis")
    st.markdown("""
    This plot shows how each feature contributes to this patient's sepsis risk prediction.
    - Red features push the prediction toward sepsis
    - Blue features push the prediction away from sepsis
    - The width of each feature represents its importance for this specific patient
    """)
    
    # Create force plot
    fig, ax = plt.subplots(figsize=(12, 3))
    
    try:
        # Get the base value (expected value)
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1:
                base_value = explainer.expected_value[1]  # Use class 1 (sepsis)
            else:
                base_value = explainer.expected_value
        else:
            base_value = 0.0
        
        # Get the SHAP values for the sample
        if isinstance(sample_shap_values, list) and len(sample_shap_values) > 1:
            sample_shap_to_plot = sample_shap_values[1]
        else:
            sample_shap_to_plot = sample_shap_values
        
        # Create a simple horizontal bar chart showing feature contributions
        feature_names = list(data['sample'].columns)
        if len(feature_names) != len(sample_shap_to_plot):
            feature_names = [f"Feature {i}" for i in range(len(sample_shap_to_plot))]
        
        # Sort by absolute contribution
        indices = np.argsort(np.abs(sample_shap_to_plot))[-10:]  # Top 10 features
        
        # Create a horizontal bar chart
        plt.barh(
            [feature_names[i] for i in indices],
            [sample_shap_to_plot[i] for i in indices],
            color=['red' if x > 0 else 'blue' for x in [sample_shap_to_plot[i] for i in indices]]
        )
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title('Top Features Impacting This Patient\'s Prediction')
    except Exception as e:
        st.warning(f"Could not create SHAP force plot: {e}")
        plt.text(0.5, 0.5, "SHAP plot unavailable", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display waterfall plot for this patient
    st.markdown("### Detailed Feature Impact")
    st.markdown("""
    This waterfall plot shows step-by-step how each feature moves the prediction from the baseline.
    It helps understand the cumulative effect of features on this patient's sepsis risk.
    """)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    try:
        # Get the base value (expected value)
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1:
                base_value = explainer.expected_value[1]  # Use class 1 (sepsis)
            else:
                base_value = explainer.expected_value
        else:
            base_value = 0.0
        
        # Get the SHAP values for the sample
        if isinstance(sample_shap_values, list) and len(sample_shap_values) > 1:
            sample_shap_to_plot = sample_shap_values[1]
        else:
            sample_shap_to_plot = sample_shap_values
            
        # Flatten if needed
        if len(sample_shap_to_plot.shape) > 1:
            sample_shap_to_plot = sample_shap_to_plot.flatten()
        
        # Get feature names
        feature_names = list(data['sample'].columns)
        if len(feature_names) != len(sample_shap_to_plot):
            feature_names = [f"Feature {i}" for i in range(len(sample_shap_to_plot))]
        
        # Sort by absolute contribution
        sorted_idx = np.argsort(np.abs(sample_shap_to_plot))
        sorted_values = sample_shap_to_plot[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        # Take top 15 features
        top_idx = sorted_idx[-15:]
        top_values = sample_shap_to_plot[top_idx]
        top_names = [feature_names[i] for i in top_idx]
        
        # Create a cumulative waterfall chart manually
        cumulative = [base_value]
        for val in top_values:
            cumulative.append(cumulative[-1] + val)
        
        # Create positions for the bars
        pos = list(range(len(top_names) + 1))
        
        # Create the waterfall chart
        plt.bar(pos[0], cumulative[0], bottom=0, color='gray', width=0.5)
        
        # Add the feature contributions
        for i in range(len(top_names)):
            # If positive contribution, plot from previous cumulative to new cumulative
            if top_values[i] >= 0:
                plt.bar(pos[i+1], top_values[i], bottom=cumulative[i], color='red', width=0.5)
            # If negative contribution, plot from new cumulative to previous cumulative
            else:
                plt.bar(pos[i+1], top_values[i], bottom=cumulative[i+1], color='blue', width=0.5)
        
        # Add feature names as y-tick labels
        plt.yticks(pos, ['Base Value'] + top_names)
        plt.xlabel('Contribution to Prediction')
        plt.title('Feature Contributions to Prediction')
        plt.grid(axis='x', alpha=0.3)
        
    except Exception as e:
        st.warning(f"Could not create SHAP waterfall plot: {e}")
        plt.text(0.5, 0.5, "SHAP waterfall plot unavailable", ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Clinical recommendations based on SHAP analysis
    st.subheader("4. Clinical Recommendations")
    
    # Get top positive and negative features
    try:
        # Handle different shapes of SHAP values
        if len(sample_shap_values.shape) > 1:
            # If 2D, take the first row
            shap_values_1d = sample_shap_values[0]
        else:
            # If already 1D, use as is
            shap_values_1d = sample_shap_values
            
        # Ensure we have 1D arrays for DataFrame
        if len(shap_values_1d.shape) > 1:
            # If still 2D (e.g., for multi-class models), take the positive class (index 1)
            if shap_values_1d.shape[0] > 1:
                shap_values_1d = shap_values_1d[1]
            else:
                # Flatten if it's a 2D array with one row
                shap_values_1d = shap_values_1d.flatten()
                
        # Make sure the length matches the number of features
        feature_names = list(data['sample'].columns)
        sample_values = data['sample'].values[0]
        
        # If lengths don't match, use generic feature names and truncate/pad arrays
        if len(feature_names) != len(shap_values_1d):
            min_len = min(len(feature_names), len(shap_values_1d))
            feature_names = feature_names[:min_len]
            shap_values_1d = shap_values_1d[:min_len]
            if len(sample_values) > min_len:
                sample_values = sample_values[:min_len]
                
        # Create DataFrame with 1D arrays
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values_1d,
            'Value': sample_values
        })
        
        # Sort by absolute SHAP value
        feature_importance['Abs SHAP'] = feature_importance['SHAP Value'].abs()
        feature_importance = feature_importance.sort_values('Abs SHAP', ascending=False)
    except Exception as e:
        st.warning(f"Error creating feature importance table: {e}")
        # Create a simple dummy DataFrame as fallback
        feature_importance = pd.DataFrame({
            'Feature': ['HR', 'Temp', 'MAP'],
            'SHAP Value': [0.1, 0.05, -0.03],
            'Value': [80, 37.2, 90],
            'Abs SHAP': [0.1, 0.05, 0.03]
        })
    
    # Get top risk-increasing features
    risk_increasing = feature_importance[feature_importance['SHAP Value'] > 0].head(3)
    
    # Get top risk-decreasing features
    risk_decreasing = feature_importance[feature_importance['SHAP Value'] < 0].head(3)
    
    # Display recommendations
    st.markdown("### Key Risk Factors")
    if not risk_increasing.empty:
        st.markdown("These factors are **increasing** this patient's sepsis risk:")
        for _, row in risk_increasing.iterrows():
            st.markdown(f"- **{row['Feature']}**: Value of {row['Value']:.2f} contributes {row['SHAP Value']:.4f} to sepsis risk")
    else:
        st.markdown("No significant risk-increasing factors identified.")
    
    st.markdown("### Protective Factors")
    if not risk_decreasing.empty:
        st.markdown("These factors are **decreasing** this patient's sepsis risk:")
        for _, row in risk_decreasing.iterrows():
            st.markdown(f"- **{row['Feature']}**: Value of {row['Value']:.2f} contributes {row['SHAP Value']:.4f} to sepsis risk")
    else:
        st.markdown("No significant risk-decreasing factors identified.")
    
    # Provide clinical guidance
    st.markdown("### Clinical Guidance")
    
    if sepsis_prob >= 0.7:
        st.error("""
        **High Sepsis Risk**
        
        Recommended actions:
        1. Immediate clinical assessment
        2. Consider blood cultures and lactate measurement
        3. Monitor key risk factors identified above
        4. Early antibiotic administration if sepsis is clinically suspected
        """)
    elif sepsis_prob >= 0.3:
        st.warning("""
        **Moderate Sepsis Risk**
        
        Recommended actions:
        1. Increased monitoring frequency
        2. Review risk factors identified above
        3. Consider additional diagnostic tests
        4. Re-evaluate within 2-4 hours
        """)
    else:
        st.success("""
        **Low Sepsis Risk**
        
        Recommended actions:
        1. Continue routine monitoring
        2. Be aware of protective factors that may be masking risk
        3. Re-evaluate if clinical condition changes
        """)
