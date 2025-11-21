
import streamlit as st
import pandas as pd
import numpy as np

def apply_feature_shock(df, shocks_dict, shock_type, alpha_value):
    df_shocked = df.copy()
    for feature, raw_shock_value in shocks_dict.items():
        if feature in df_shocked.columns:
            if shock_type == 'multiplicative':
                # x_stressed = x * (1 + alpha * delta_s)
                df_shocked[feature] = df_shocked[feature] * (1 + alpha_value * raw_shock_value)
            elif shock_type == 'additive':
                # x_stressed = x + alpha * delta_s
                df_shocked[feature] = df_shocked[feature] + (alpha_value * raw_shock_value)
    return df_shocked

scenario_shocks_config = {
    'Macro Downturn': {
        'shock_type': 'multiplicative',
        'shocks': {'income': -0.15, 'debt_to_income': 0.15, 'utilization': 0.20, 'credit_spread': 0.25}
    },
    'Market Stress': {
        'shock_type': 'multiplicative',
        'shocks': {'credit_spread': 0.50, 'volatility': 0.30, 'liquidity': -0.20}
    },
    'Housing Market Downturn': {
        'shock_type': 'multiplicative',
        'shocks': {'house_price': -0.20, 'LTV': 0.10, 'debt_to_income': 0.05}
    }
}

def run_page2():
    st.markdown("### 5.4. Defining the Stress Transformation Library")
    st.markdown(r"""
    Stress scenarios are translated into quantitative feature transformations $T_s(x)$. These transformations modify the input features according to predefined shock parameters. We will implement a function to apply multiplicative (scaling) or additive (shifting) shocks.

    The general form of a multiplicative shock is:

    $$x^{(s)} = x \odot (1 + \delta_s)$$

    Where $\odot$ denotes element-wise multiplication and $\delta_s$ is a vector of scenario-specific shocks. This means that each feature $x_i$ in the input $x$ is transformed into $x_i^{(s)} = x_i \times (1 + \delta_{s,i})$, where $\delta_{s,i}$ is the shock factor for feature $i$ in scenario $s$.

    For an additive shock, the form is:

    $$x^{(s)} = x + \delta_s$$

    Where $\delta_s$ is a vector of scenario-specific additive shifts. This means that each feature $x_i$ is transformed into $x_i + \delta_{s,i}$.

    **Business Value:**
    *   **Quantifiable Impact**: This library provides the mechanism to translate qualitative financial stress narratives (e.g., "recession") into concrete, quantifiable changes in the data that feed into the ML model.
    *   **Scenario Flexibility**: Supports both multiplicative (percentage-based) and additive (absolute-value-based) shocks, allowing for a wide range of realistic scenario constructions.
    *   **Standardization**: Ensures a consistent method for applying shocks across different features and scenarios, which is crucial for comparable and auditable stress test results.
    """)

    st.subheader("Pre-defined Scenario Configurations")
    st.markdown("Here are the pre-defined stress scenarios and their associated feature shocks:")
    st.json(scenario_shocks_config)


    if 'df_baseline' not in st.session_state or 'mock_model' not in st.session_state:
        st.warning("Please generate data and set up the model on 'Data Generation & Model Setup' page first.")
        return

    df_baseline = st.session_state['df_baseline']
    mock_model = st.session_state['mock_model']
    numerical_features = st.session_state['numerical_features']


    with st.sidebar:
        st.subheader("Scenario Definition & Application")
        selected_scenario_option = st.selectbox(
            "Select a Stress Scenario",
            options=list(scenario_shocks_config.keys()) + ["Custom Scenario"],
            key="selected_scenario_option",
            help="Choose a pre-defined scenario or define your own custom shocks."
        )

        if selected_scenario_option == "Custom Scenario":
            with st.expander("Define Custom Scenario"):
                custom_shock_type = st.radio(
                    "Select Shock Type",
                    options=['Multiplicative', 'Additive'],
                    key="custom_shock_type",
                    help="Choose whether shocks should multiply or add to feature values."
                )

                st.markdown("Add Feature-Shock Pairs:")
                if 'custom_shocks' not in st.session_state:
                    st.session_state['custom_shocks'] = []

                # Display existing custom shocks and allow removal
                for i, shock_pair in enumerate(st.session_state['custom_shocks']):
                    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
                    with col1:
                        st.text_input(f"Feature {i+1}", value=shock_pair['feature'], key=f"custom_feature_{i}")
                    with col2:
                        st.number_input(f"Shock Value {i+1}", value=shock_pair['value'], key=f"custom_value_{i}", format="%.4f")
                    with col3:
                        if st.button("Remove", key=f"remove_shock_{i}"):
                            st.session_state['custom_shocks'].pop(i)
                            st.experimental_rerun() # Rerun to update the list

                # Add new custom shock pair
                with st.form("add_custom_shock_form"):
                    col_f, col_v = st.columns(2)
                    with col_f:
                        new_feature = st.text_input("New Feature Name", key="new_feature_name_input")
                    with col_v:
                        new_shock_value = st.number_input("New Shock Value", value=0.0, key="new_shock_value_input", format="%.4f")
                    add_shock_button = st.form_submit_button("Add Custom Shock")
                    if add_shock_button and new_feature and new_feature not in [s['feature'] for s in st.session_state['custom_shocks']]:
                        st.session_state['custom_shocks'].append({'feature': new_feature, 'value': new_shock_value})
                        st.experimental_rerun()
                    elif add_shock_button and not new_feature:
                        st.warning("Feature name cannot be empty.")
                    elif add_shock_button and new_feature in [s['feature'] for s in st.session_state['custom_shocks']]:
                        st.warning(f"Feature '{new_feature}' already exists. Please modify the existing one or choose a different name.")

        alpha_value = st.slider(
            r"Shock Intensity ($\alpha$)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="alpha_value_slider",
            help=r"Adjust the intensity of the applied shocks. $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$ for multiplicative shocks or $x^{(s,\alpha)} = x + \alpha \cdot \delta_s$ for additive shocks."
        )
        st.session_state['alpha_value'] = alpha_value

        run_stress_test_button = st.button("Run Stress Test", help="Apply the selected scenario and calculate stressed predictions.")

    if run_stress_test_button:
        if selected_scenario_option == "Custom Scenario":
            if not st.session_state['custom_shocks']:
                st.error("Please define at least one custom feature-shock pair.")
                return
            current_shocks_dict = {item['feature']: item['value'] for item in st.session_state['custom_shocks']}
            current_shock_type = st.session_state['custom_shock_type'].lower()
        else:
            scenario_config = scenario_shocks_config[selected_scenario_option]
            current_shocks_dict = scenario_config['shocks']
            current_shock_type = scenario_config['shock_type'].lower()

        with st.spinner(f"Applying '{selected_scenario_option}' stress scenario with $\alpha$={alpha_value} and predicting..."):
            X_baseline_for_stress = df_baseline[numerical_features].copy() # Ensure we use the numerical features for stressing
            X_stressed = apply_feature_shock(X_baseline_for_stress, current_shocks_dict, current_shock_type, alpha_value)

            # Ensure 'exposure' is in df_baseline for EL calculation
            if 'exposure' not in df_baseline.columns:
                st.error("The 'exposure' feature is missing in the generated data, which is required for EL calculation.")
                return

            y_hat_stressed = mock_model.predict(X_stressed)

            df_stressed = df_baseline.copy()
            df_stressed['PD_stressed'] = y_hat_stressed
            df_stressed['Delta_PD'] = df_stressed['PD_stressed'] - df_stressed['PD_baseline']

            df_stressed['EL_baseline'] = df_stressed['PD_baseline'] * df_stressed['exposure']
            df_stressed['EL_stressed'] = df_stressed['PD_stressed'] * df_stressed['exposure']
            df_stressed['Delta_EL'] = df_stressed['EL_stressed'] - df_stressed['EL_baseline']

            st.session_state['df_stressed'] = df_stressed
            st.session_state['stressed_results_ready'] = True
            st.session_state['last_run_scenario'] = selected_scenario_option
            st.session_state['last_run_alpha'] = alpha_value
            st.success("Stress test completed successfully! Navigate to 'Impact Analysis & Visualizations' to see results.")
            st.markdown(f"**Scenario Applied:** `{selected_scenario_option}` with shock intensity $\alpha = {alpha_value}$")
            st.markdown("Here's a preview of the stressed data with new PD and EL metrics:")
            st.dataframe(df_stressed[['PD_baseline', 'PD_stressed', 'Delta_PD', 'EL_baseline', 'EL_stressed', 'Delta_EL']].head())

    else:
        if st.session_state.get('stressed_results_ready'):
            st.info(f"Last stress test run: '{st.session_state.get('last_run_scenario')}' with $\alpha$={st.session_state.get('last_run_alpha')}. Results are ready on 'Impact Analysis & Visualizations' page.")
            st.markdown("Here's a preview of the last stressed data:")
            st.dataframe(st.session_state['df_stressed'][['PD_baseline', 'PD_stressed', 'Delta_PD', 'EL_baseline', 'EL_stressed', 'Delta_EL']].head())
        else:
            st.info("No stress test has been run yet. Use the sidebar to define a scenario and run the stress test.")
