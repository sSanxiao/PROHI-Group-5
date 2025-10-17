import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap
import warnings
warnings.filterwarnings('ignore')

# Feature columns used in Random Forest model
FEATURE_COLUMNS = [
    'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
    'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender'
]

def load_rf_model_for_shap():
    """Load Random Forest model for SHAP analysis"""
    try:
        model_path = './models/random_forest.pkl'
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            return data['model'], data.get('scaler'), data.get('features', FEATURE_COLUMNS)
        else:
            st.error("Random Forest model file not found.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")
        return None, None, None

def create_shap_explainer(model, X_background):
    """Create SHAP explainer with background data"""
    try:
        # For TreeExplainer, we can use the model directly without background data
        # The background data is used for feature attribution, not for the explainer itself
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {e}")
        return None

def plot_shap_force(explainer, shap_values, feature_names, feature_values, index=0):
    """Create a custom force plot visualization"""
    fig, ax = plt.subplots(figsize=(5, 2))
    
    try:
        # Get base value
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        base_value = float(base_value)  # Convert to scalar
        
        # Get SHAP values for the instance
        instance_shap = shap_values[index] if len(shap_values.shape) > 1 else shap_values
        
        # Ensure instance_shap is 1D
        if len(instance_shap.shape) > 1:
            instance_shap = instance_shap.flatten()
        
        # Sort features by absolute SHAP value, excluding Hour
        abs_shap = np.abs(instance_shap)
        # Create mask to exclude Hour feature
        hour_mask = np.array([name != 'Hour' for name in feature_names])
        # Apply mask to get indices of non-Hour features
        non_hour_indices = np.where(hour_mask)[0]
        # Sort only non-Hour features
        sorted_idx = non_hour_indices[np.argsort(abs_shap[non_hour_indices])[::-1][:10]]  # Top 10 non-Hour features
        
        # Create horizontal bar chart - convert to scalars
        colors = ['#ff0051' if float(instance_shap[int(i)]) > 0 else '#008bfb' for i in sorted_idx]
        bars = ax.barh(
            [feature_names[int(i)] for i in sorted_idx],
            [float(instance_shap[int(i)]) for i in sorted_idx],
            color=colors,
            alpha=0.8
        )
        
        # Add feature values as text on the bars themselves
        for i, (idx, bar) in enumerate(zip(sorted_idx, bars)):
            idx = int(idx)  # Convert to scalar
            val = float(feature_values[idx])  # Convert to scalar
            x_pos = bar.get_width()
            # Position text at the end of the bar, inside the bar
            text_x = x_pos * 0.5 if abs(x_pos) > 0.1 else x_pos * 0.8
            ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', 
                   va='center', ha='center',
                   fontsize=4, fontweight='bold', color='white')
        
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=6, fontweight='bold')
        ax.set_title(f'Top 10 Feature Contributions (Base Value: {base_value:.3f})', fontsize=7, fontweight='bold')
        ax.tick_params(axis='y', labelsize=5)  # Make y-axis labels smaller
        ax.tick_params(axis='x', labelsize=5) 
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        return fig
    except Exception as e:
        st.error(f"Error creating force plot: {e}")
        return None

def plot_shap_waterfall(shap_values, feature_names, feature_values, base_value, index=0):
    """Create a waterfall plot showing cumulative feature contributions"""
    fig, ax = plt.subplots(figsize=(3, 3))
    
    try:
        # Get SHAP values for the instance
        instance_shap = shap_values[index] if len(shap_values.shape) > 1 else shap_values
        
        # Ensure instance_shap is 1D
        if len(instance_shap.shape) > 1:
            instance_shap = instance_shap.flatten()
        
        # Convert base_value to scalar
        base_value = float(base_value)
        
        # Sort by absolute value and take top 15, excluding Hour
        abs_shap = np.abs(instance_shap)
        # Create mask to exclude Hour feature
        hour_mask = np.array([name != 'Hour' for name in feature_names])
        # Apply mask to get indices of non-Hour features
        non_hour_indices = np.where(hour_mask)[0]
        # Sort only non-Hour features
        sorted_idx = non_hour_indices[np.argsort(abs_shap[non_hour_indices])[::-1][:15]]  # Top 15 non-Hour features
        sorted_shap = [float(instance_shap[int(i)]) for i in sorted_idx]  # Convert to scalars
        sorted_names = [f"{feature_names[int(i)]} = {float(feature_values[int(i)]):.2f}" for i in sorted_idx]
        
        # Calculate cumulative values
        cumulative = [base_value]
        for val in sorted_shap:
            cumulative.append(cumulative[-1] + val)
        
        # Plot
        y_pos = np.arange(len(sorted_names) + 2)
        
        # Base value
        ax.barh(0, base_value, color='lightgray', height=0.6, label='Base Value')
        
        # Feature contributions
        for i, (shap_val, name) in enumerate(zip(sorted_shap, sorted_names), 1):
            color = '#ff0051' if shap_val > 0 else '#008bfb'
            ax.barh(i, shap_val, left=cumulative[i-1], color=color, height=0.6, alpha=0.8)
            
            # Add text showing the contribution directly on the bar
            mid_point = cumulative[i-1] + shap_val/2
            ax.text(mid_point, i, f'{shap_val:+.3f}', 
                   ha='center', va='center', fontsize=3, fontweight='bold', color='white')
        
        # Final prediction
        final_val = cumulative[-1]
        ax.barh(len(sorted_names) + 1, final_val, color='gold', height=0.6, label='Final Prediction')
        
        # Add final prediction value on the bar
        ax.text(final_val/2, len(sorted_names) + 1, f'{final_val:.3f}', 
               ha='center', va='center', fontsize=3, fontweight='bold', color='black')
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Base Value'] + sorted_names + ['Final Value'])
        ax.set_xlabel('Model Output Value', fontsize=6, fontweight='bold')
        ax.set_title('Waterfall Plot: Feature Contributions', fontsize=7, fontweight='bold')
        ax.tick_params(axis='y', labelsize=5)  # Make y-axis labels smaller
        ax.tick_params(axis='x', labelsize=5) 
        ax.legend(loc='best', fontsize=5)  # Make legend smaller
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
        return fig
    except Exception as e:
        st.error(f"Error creating waterfall plot: {e}")
        return None

def plot_shap_summary(shap_values, X, feature_names):
    """Create summary plot showing feature importance"""
    fig, ax = plt.subplots(figsize=(3, 3))
    
    try:
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features, excluding Hour
        # Create mask to exclude Hour feature
        hour_mask = np.array([name != 'Hour' for name in feature_names])
        # Apply mask to get indices of non-Hour features
        non_hour_indices = np.where(hour_mask)[0]
        # Sort only non-Hour features
        sorted_idx = non_hour_indices[np.argsort(mean_shap[non_hour_indices])[::-1][:15]]  # Top 15 non-Hour features
        
        # Plot
        ax.barh([feature_names[i] for i in sorted_idx], 
               mean_shap[sorted_idx],
               color='steelblue', alpha=0.8)
        
        ax.set_xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=6, fontweight='bold')
        ax.set_title('Global Feature Importance', fontsize=7, fontweight='bold')
        ax.tick_params(axis='y', labelsize=5)  # Make y-axis labels smaller
        ax.tick_params(axis='x', labelsize=5) 
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        return fig
    except Exception as e:
        st.error(f"Error creating summary plot: {e}")
        return None

def plot_shap_beeswarm(shap_values, X, feature_names):
    """Create beeswarm plot showing feature impact distribution"""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    try:
        # Calculate feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        # Sort features, excluding Hour
        # Create mask to exclude Hour feature
        hour_mask = np.array([name != 'Hour' for name in feature_names])
        # Apply mask to get indices of non-Hour features
        non_hour_indices = np.where(hour_mask)[0]
        # Sort only non-Hour features
        sorted_idx = non_hour_indices[np.argsort(mean_shap[non_hour_indices])[::-1][:15]]  # Top 15 non-Hour features
        
        # Create plot for each feature
        for i, feat_idx in enumerate(sorted_idx):
            feat_idx = int(feat_idx)  # Convert to scalar integer
            # Get SHAP values and feature values for this feature
            feat_shap = shap_values[:, feat_idx]
            feat_vals = X[:, feat_idx]
            
            # Normalize feature values for color mapping
            feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)
            
            # Scatter plot
            ax.scatter(feat_shap, np.ones(len(feat_shap)) * i, 
                      c=feat_vals_norm, cmap='coolwarm', 
                      alpha=0.6, s=30, edgecolors='none')
        
        # Labels
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[int(i)] for i in sorted_idx])
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=6, fontweight='bold')
        ax.set_title('Feature Impact Distribution\n(Red = High Feature Value, Blue = Low Feature Value)', 
                    fontsize=7, fontweight='bold')
        ax.tick_params(axis='y', labelsize=5)  # Make y-axis labels smaller
        ax.tick_params(axis='x', labelsize=5) 
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Feature Value', rotation=270, labelpad=10, fontsize=5)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
        return fig
    except Exception as e:
        st.error(f"Error creating beeswarm plot: {e}")
        return None

def prescriptive_analytics():
    st.markdown("""
    ## 💡 Prescriptive Analytics - Model Explainability
    
    This page provides detailed explanations of the **Random Forest model** predictions using SHAP (SHapley Additive exPlanations).
    SHAP values help understand which features contribute most to sepsis risk predictions and by how much.
    
    **Note:** SHAP analysis is only available for Single Timepoint Analysis (Random Forest model).
    """)
    
    st.info("💡 **Tip:** Input patient data in the Predictive Analytics tab first, then use it here for detailed analysis.")
    
    # Check if we're getting dimension mismatch issues
    if 'data' in st.session_state and st.session_state.data.get('df') is not None:
        df_check = st.session_state.data['df']
        num_features = len([col for col in df_check.columns if col not in ['Patient_ID', 'SepsisLabel']])
        if num_features != 17:
            st.warning(f"""
            ⚠️ **Data dimension issue detected!** The cached dataset has {num_features} features instead of the expected 17.
            
            **To fix this:** Click the "🔄 Clear Cache & Reload Data" button in the sidebar, then refresh this page.
            """)
    
    st.markdown("---")
    
    # Load Random Forest model
    with st.spinner("Loading Random Forest model..."):
        model, scaler, model_features = load_rf_model_for_shap()
    
    if model is None:
        st.error("⚠️ Random Forest model not available. Please ensure the model is trained and saved.")
        return
    
    # Use the model's features instead of FEATURE_COLUMNS
    # This ensures SHAP values match what the model expects
    if model_features is not None and len(model_features) > 0:
        working_features = model_features
        st.info(f"ℹ️ Model was trained with {len(working_features)} features: {', '.join(working_features)}")
    else:
        working_features = FEATURE_COLUMNS
        st.warning(f"⚠️ Model features not found in saved model. Using default {len(FEATURE_COLUMNS)} features.")
    
    # Data input options
    st.markdown("### 📥 Data Input Options")
    
    input_mode = st.radio(
        "Choose how to input patient data:",
        ["Use data from Predictive Analytics", "Select from dataset", "Manual input"],
        horizontal=False
    )
    
    X_data = None
    sample_indices = []
    
    if input_mode == "Use data from Predictive Analytics":
        # Check if there's data from Predictive Analytics in session state
        if 'predictive_input' in st.session_state and st.session_state.predictive_input is not None:
            st.success("✅ Using patient data from Predictive Analytics tab")
            
            # Display the input data in a more compact format
            input_df = pd.DataFrame([st.session_state.predictive_input])
            st.markdown("**Patient Data:**")
            
            # Display in columns for better layout
            cols_per_row = 4
            features_list = list(st.session_state.predictive_input.items())
            
            for i in range(0, len(features_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, (feat, val) in enumerate(features_list[i:i+cols_per_row]):
                    with cols[j]:
                        # Format value based on type
                        if feat == 'Gender':
                            display_val = "Female" if val == 0 else "Male"
                        else:
                            display_val = f"{val:.2f}" if isinstance(val, float) else str(val)
                        st.metric(feat, display_val)
            
            # Prepare for SHAP analysis - only use features the model was trained on
            available_features = [f for f in working_features if f in input_df.columns]
            X_data = input_df[available_features].values
            sample_indices = [0]
            
        else:
            st.warning("⚠️ No input data found from Predictive Analytics. Please input data in the Predictive Analytics tab first, or choose another input method.")
            return
    
    elif input_mode == "Select from dataset":
        # Load dataset
        try:
            if 'data' in st.session_state and st.session_state.data['df'] is not None:
                df = st.session_state.data['df']
            else:
                df = pd.read_csv('./data/cleaned_dataset.csv')
            
            # Keep only the 17 features we need plus ID and label
            columns_to_keep = [
                'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
                'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender',
                'Patient_ID', 'SepsisLabel'
            ]
            # Only keep columns that exist in the dataframe
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            df = df[columns_to_keep]
            
            st.markdown("**Select patients from the dataset:**")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_sepsis = st.selectbox(
                    "Filter by Sepsis Status:",
                    ["All patients", "Sepsis patients only", "Non-sepsis patients only"]
                )
            
            with col2:
                num_samples = st.slider("Number of samples to analyze:", 1, 10, 3)
            
            # Apply filter
            if filter_sepsis == "Sepsis patients only":
                df_filtered = df[df['SepsisLabel'] == 1]
            elif filter_sepsis == "Non-sepsis patients only":
                df_filtered = df[df['SepsisLabel'] == 0]
            else:
                df_filtered = df
            
            # Random sample button
            if st.button("🎲 Select Random Samples"):
                if len(df_filtered) >= num_samples:
                    sample_df = df_filtered.sample(n=num_samples, random_state=np.random.randint(1000))
                    st.session_state['selected_samples'] = sample_df
                else:
                    st.error(f"Not enough samples in filtered data. Available: {len(df_filtered)}")
            
            # Display selected samples
            if 'selected_samples' in st.session_state:
                sample_df = st.session_state['selected_samples']
                st.markdown(f"**Selected {len(sample_df)} samples:**")
                
                # Display in an expandable section if multiple samples
                available_features = [f for f in working_features if f in sample_df.columns]
                if len(sample_df) > 1:
                    with st.expander(f"View All {len(sample_df)} Patients", expanded=False):
                        st.dataframe(sample_df[available_features + ['SepsisLabel']], use_container_width=True)
                else:
                    st.dataframe(sample_df[available_features + ['SepsisLabel']], use_container_width=True)
                
                # Prepare for SHAP analysis - only use features the model was trained on
                X_data = sample_df[available_features].values
                sample_indices = list(range(len(X_data)))
            else:
                st.info("Click 'Select Random Samples' to choose patients for analysis.")
                return
                
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return
    
    else:  # Manual input
        st.markdown("**Enter patient data manually:**")
        
        # Create input form
        with st.form("manual_input_form"):
            input_data = {}
            
            # Organize inputs by category
            categories = {
                "Temporal & Duration": ['Hour', 'ICULOS'],
                "Vital Signs": ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp'],
                "Blood Gas": ['HCO3', 'pH', 'PaCO2'],
                "Lab Values": ['Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets'],
                "Demographics": ['Age', 'Gender']
            }
            
            for category, feats in categories.items():
                st.markdown(f"**{category}:**")
                cols = st.columns(3)
                for i, feat in enumerate(feats):
                    with cols[i % 3]:
                        if feat == 'Gender':
                            input_data[feat] = st.selectbox(feat, [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                        elif feat in ['Hour', 'Age', 'ICULOS']:
                            input_data[feat] = st.number_input(feat, min_value=0, max_value=200, value=0, step=1)
                        else:
                            # Default values based on normal ranges
                            defaults = {
                                'HR': 75, 'O2Sat': 97, 'Temp': 37, 'SBP': 120, 'MAP': 85, 'Resp': 16,
                                'HCO3': 24, 'pH': 7.4, 'PaCO2': 40, 'Creatinine': 1.0, 
                                'Bilirubin_direct': 0.2, 'WBC': 7.5, 'Platelets': 250
                            }
                            input_data[feat] = st.number_input(feat, value=float(defaults.get(feat, 0)), step=0.1)
            
            submitted = st.form_submit_button("Submit Data")
            
            if submitted:
                # Prepare for SHAP analysis - only use features the model was trained on
                input_df = pd.DataFrame([input_data])
                available_features = [f for f in working_features if f in input_df.columns]
                X_data = input_df[available_features].values
                sample_indices = [0]
                
                st.success("✅ Data submitted successfully!")
                
                # Display in a more compact format
                st.markdown("**Submitted Patient Data:**")
                cols_per_row = 4
                features_list = list(input_data.items())
                
                for i in range(0, len(features_list), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (feat, val) in enumerate(features_list[i:i+cols_per_row]):
                        with cols[j]:
                            # Format value based on type
                            if feat == 'Gender':
                                display_val = "Female" if val == 0 else "Male"
                            else:
                                display_val = f"{val:.2f}" if isinstance(val, float) else str(val)
                            st.metric(feat, display_val)
    
    # Perform SHAP analysis if we have data
    if X_data is not None and len(sample_indices) > 0:
        st.markdown("---")
        st.markdown("## 🔍 SHAP Analysis Results")
        
        with st.spinner("Computing SHAP values... This may take a moment."):
            try:
                # Load background data for SHAP
                if 'data' in st.session_state and st.session_state.data['df'] is not None:
                    df_bg = st.session_state.data['df']
                else:
                    df_bg = pd.read_csv('./data/cleaned_dataset.csv')
                
                # Keep only the 17 features we need
                columns_to_keep = [
                    'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'HCO3', 'pH', 'PaCO2', 
                    'Creatinine', 'Bilirubin_direct', 'WBC', 'Platelets', 'ICULOS', 'Age', 'Gender'
                ]
                # Only keep columns that exist in the dataframe
                available_cols = [col for col in columns_to_keep if col in df_bg.columns]
                df_bg = df_bg[available_cols]
                
                # Sample background data
                X_background = df_bg.sample(n=min(100, len(df_bg)), random_state=42).values
                
                # Scale the data
                if scaler is not None:
                    try:
                        # Try to use the loaded scaler
                        X_background_scaled = scaler.transform(X_background)
                        X_data_scaled = scaler.transform(X_data)
                    except Exception as e:
                        st.warning(f"Loaded scaler not fitted. Creating new scaler: {e}")
                        # Create and fit a new scaler
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_background_scaled = scaler.fit_transform(X_background)
                        X_data_scaled = scaler.transform(X_data)
                else:
                    # Create a new scaler if none exists
                    from sklearn.preprocessing import StandardScaler
                    st.info("Creating new scaler for data normalization.")
                    scaler = StandardScaler()
                    X_background_scaled = scaler.fit_transform(X_background)
                    X_data_scaled = scaler.transform(X_data)
                print(X_background_scaled[0])
                # Create SHAP explainer
                explainer = create_shap_explainer(model, X_background_scaled)
                
                if explainer is None:
                    st.error("Failed to create SHAP explainer.")
                    return
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_data_scaled)
                
                # Handle 3D SHAP values (samples, features, classes)
                if len(shap_values.shape) == 3:
                    # shap_values shape: (samples, features, classes)
                    # We want to show both classes, so we'll keep the 3D structure
                    st.info(f"ℹ️ SHAP values for both classes. Shape: {shap_values.shape} (samples, features, classes)")
                elif isinstance(shap_values, list) and len(shap_values) == 2:
                    # Convert list format to 3D array
                    shap_values = np.stack(shap_values, axis=-1)  # Stack along the last axis
                    st.info(f"ℹ️ Converted list to 3D array. Shape: {shap_values.shape}")
                else:
                    st.info(f"ℹ️ Using SHAP values. Shape: {shap_values.shape}")
                
                # Make predictions
                predictions = model.predict_proba(X_data_scaled)[:, 1]
                
            except Exception as e:
                st.error(f"Error computing SHAP values: {e}")
                return
        
        # Display results for each sample
        for idx, sample_idx in enumerate(sample_indices):
            st.markdown(f"### Patient {idx + 1}")
            
            # Display prediction
            pred_prob = predictions[sample_idx]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sepsis Risk", f"{pred_prob:.1%}")
            with col2:
                risk_level = "High" if pred_prob >= 0.44 else "Low" if pred_prob <= 0.34 else "Moderate"
                st.metric("Risk Level", risk_level)
            with col3:
                st.metric("Base Value", f"{explainer.expected_value[1]:.3f}" if isinstance(explainer.expected_value, (list, np.ndarray)) else f"{explainer.expected_value:.3f}")
            
            # Tabs for individual patient visualizations only
            tab1, tab2 = st.tabs([
                "📊 Feature Contributions", 
                "💧 Waterfall Plot"
            ])
            
            with tab1:
                st.markdown("**How to read:** Red bars increase sepsis risk, blue bars decrease risk. Bar length shows impact magnitude.")
                
                # For 3D SHAP values, show both classes
                if len(shap_values.shape) == 3:
                    # Show sepsis class (class 1)
                    shap_sepsis = shap_values[sample_idx, :, 1]  # Shape: (features,)
                    fig = plot_shap_force(explainer, shap_sepsis.reshape(1, -1), working_features, X_data[sample_idx], index=0)
                else:
                    fig = plot_shap_force(explainer, shap_values, working_features, X_data[sample_idx], index=0)
                if fig:
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
            
            with tab2:
                st.markdown("**How to read:** Shows cumulative feature effects. Starts from base value, each bar moves prediction up/down.")
                
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                
                # For 3D SHAP values, show sepsis class
                if len(shap_values.shape) == 3:
                    shap_sepsis = shap_values[sample_idx, :, 1]  # Shape: (features,)
                    fig = plot_shap_waterfall(shap_sepsis.reshape(1, -1), working_features, X_data[sample_idx], base_val, index=0)
                else:
                    fig = plot_shap_waterfall(shap_values, working_features, X_data[sample_idx], base_val, index=0)
                if fig:
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
            
            
            # Clinical insights
            st.markdown("#### 🏥 Clinical Insights")
            
            # Get SHAP values for both classes
            if len(shap_values.shape) == 3:
                # 3D case: (samples, features, classes)
                shap_no_sepsis = shap_values[sample_idx, :, 0]  # Class 0 (no sepsis)
                shap_sepsis = shap_values[sample_idx, :, 1]     # Class 1 (sepsis)
                
                st.markdown("#### 📊 SHAP Analysis for Both Classes")
                
                # Show both classes side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔵 No Sepsis (Class 0) - Risk-Decreasing Factors:**")
                    # Get top negative contributors for no-sepsis class, excluding Hour
                    top_neg_idx = np.argsort(shap_no_sepsis)[:10]  # Get more candidates
                    count = 0
                    for feat_idx in top_neg_idx:
                        feat_idx = int(feat_idx)
                        if 0 <= feat_idx < len(working_features):
                            feat_name = working_features[feat_idx]
                            if feat_name == 'Hour':  # Skip Hour feature
                                continue
                            shap_val = float(shap_no_sepsis[feat_idx])
                            if shap_val < 0:
                                count += 1
                                feat_val = float(X_data[sample_idx, feat_idx])
                                st.markdown(f"{count}. **{feat_name}** = {feat_val:.2f} (SHAP: {shap_val:.3f})")
                                if count >= 3:  # Show top 3 non-Hour features
                                    break
                
                with col2:
                    st.markdown("**🔴 Sepsis (Class 1) - Risk-Increasing Factors:**")
                    # Get top positive contributors for sepsis class, excluding Hour
                    top_pos_idx = np.argsort(shap_sepsis)[::-1][:10]  # Get more candidates
                    count = 0
                    for feat_idx in top_pos_idx:
                        feat_idx = int(feat_idx)
                        if 0 <= feat_idx < len(working_features):
                            feat_name = working_features[feat_idx]
                            if feat_name == 'Hour':  # Skip Hour feature
                                continue
                            shap_val = float(shap_sepsis[feat_idx])
                            if shap_val > 0:
                                count += 1
                                feat_val = float(X_data[sample_idx, feat_idx])
                                st.markdown(f"{count}. **{feat_name}** = {feat_val:.2f} (SHAP: +{shap_val:.3f})")
                                if count >= 3:  # Show top 3 non-Hour features
                                    break
                
                # Use sepsis class for the main analysis
                instance_shap = shap_sepsis
            else:
                # 2D case: (samples, features)
                instance_shap = shap_values[sample_idx] if len(shap_values.shape) > 1 else shap_values
                
                # Ensure instance_shap is 1D
                if len(instance_shap.shape) > 1:
                    instance_shap = instance_shap.flatten()
                
                # Ensure we have the right number of features
                if len(instance_shap) != len(working_features):
                    st.warning(f"⚠️ SHAP values dimension mismatch: got {len(instance_shap)} values, expected {len(working_features)} features. Skipping clinical insights.")
                    return
            
                # Additional detailed analysis for sepsis class
                st.markdown("#### 🔍 Detailed Sepsis Risk Analysis")
                top_pos_idx = np.argsort(instance_shap)[::-1][:10]  # Get more candidates
                top_neg_idx = np.argsort(instance_shap)[:10]  # Get more candidates
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 Top Risk-Increasing Factors:**")
                    risk_count = 0
                    for feat_idx in top_pos_idx:
                        feat_idx = int(feat_idx)  # Convert to scalar integer
                        if 0 <= feat_idx < len(working_features):  # Bounds check
                            feat_name = working_features[feat_idx]
                            if feat_name == 'Hour':  # Skip Hour feature
                                continue
                            shap_val = float(instance_shap[feat_idx])  # Convert to scalar
                            if shap_val > 0:
                                risk_count += 1
                                feat_val = float(X_data[sample_idx, feat_idx])  # Convert to scalar
                                st.markdown(f"{risk_count}. **{feat_name}** = {feat_val:.2f} (SHAP: +{shap_val:.3f})")
                                if risk_count >= 3:  # Show top 3 non-Hour features
                                    break
                    if risk_count == 0:
                        st.markdown("*No significant risk-increasing factors*")
                
                with col2:
                    st.markdown("**🔵 Top Risk-Decreasing Factors:**")
                    protect_count = 0
                    for feat_idx in top_neg_idx:
                        feat_idx = int(feat_idx)  # Convert to scalar integer
                        if 0 <= feat_idx < len(working_features):  # Bounds check
                            feat_name = working_features[feat_idx]
                            if feat_name == 'Hour':  # Skip Hour feature
                                continue
                            shap_val = float(instance_shap[feat_idx])  # Convert to scalar
                            if shap_val < 0:
                                protect_count += 1
                                feat_val = float(X_data[sample_idx, feat_idx])  # Convert to scalar
                                st.markdown(f"{protect_count}. **{feat_name}** = {feat_val:.2f} (SHAP: {shap_val:.3f})")
                                if protect_count >= 3:  # Show top 3 non-Hour features
                                    break
                    if protect_count == 0:
                        st.markdown("*No significant risk-decreasing factors*")
            
            if idx < len(sample_indices) - 1:
                st.markdown("---")
        
        # Global Analysis Section (only show if multiple patients)
        if len(sample_indices) > 1:
            st.markdown("---")
            st.markdown("### 🌐 Global Feature Analysis")
            st.markdown("These plots show patterns across all selected patients and are the same regardless of which patient you're viewing.")
            
            # Create tabs for global visualizations
            global_tab1, global_tab2 = st.tabs(["📈 Global Feature Importance", "🎯 Feature Impact Distribution"])
            
            with global_tab1:
                st.markdown("**Global Feature Importance:** Shows average feature impact across all patients. Longer bars = greater impact.")
                
                # For 3D SHAP values, use sepsis class for global analysis
                if len(shap_values.shape) == 3:
                    shap_for_summary = shap_values[:, :, 1]  # Use sepsis class
                else:
                    shap_for_summary = shap_values
                fig = plot_shap_summary(shap_for_summary, X_data_scaled, working_features)
                if fig:
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
            
            with global_tab2:
                st.markdown("**Feature Impact Distribution:** Each dot = one patient. Red = high value, blue = low value. X-axis = impact direction.")
                
                # For 3D SHAP values, use sepsis class for beeswarm plot
                if len(shap_values.shape) == 3:
                    shap_for_beeswarm = shap_values[:, :, 1]  # Use sepsis class
                else:
                    shap_for_beeswarm = shap_values
                fig = plot_shap_beeswarm(shap_for_beeswarm, X_data_scaled, working_features)
                if fig:
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
        
        # Final notes
        st.markdown("---")
        st.markdown("""
        ### 📚 Understanding SHAP Values
        
        **What are SHAP values?**
        - SHAP (SHapley Additive exPlanations) values explain the contribution of each feature to a prediction
        
        **Key concepts:**
        - **Base value**: The average model prediction across all training data
        - **SHAP value**: How much a feature moves the prediction from the base value
        - **Positive SHAP**: Feature increases sepsis risk
        - **Negative SHAP**: Feature decreases sepsis risk
        
        **Clinical application:**
        - Use SHAP values to understand **why** a patient is flagged as high-risk
        - Identify which vital signs or lab values need immediate attention
        - Monitor how changes in specific features might affect sepsis risk
        
        ⚠️ **Important**: SHAP explanations show model behavior, not causal relationships. Always combine with clinical judgment.
        """)

if __name__ == "__main__":
    prescriptive_analytics()
