
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def run_page3():
    st.header("Impact Analysis & Visualizations")
    st.markdown("""
    This section provides a comprehensive analysis of the stress test results, visualizing the impact of scenarios on model predictions and key portfolio metrics. It includes comparisons of baseline and stressed metrics, distributions shifts, sensitivity trajectories, and segmented portfolio heatmaps.
    """)

    if 'df_stressed' not in st.session_state or not st.session_state.get('stressed_results_ready'):
        st.warning("Please run a stress test on the 'Scenario Definition & Stress Test' page first to see the results.")
        return

    df_stressed = st.session_state['df_stressed']
    mock_model = st.session_state['mock_model']
    numerical_features = st.session_state['numerical_features']
    current_shocks_config = st.session_state.get('last_run_shocks_config', {}) # Assuming this is stored in page2
    current_shock_type = st.session_state.get('last_run_shock_type', '') # Assuming this is stored in page2

    st.subheader("Baseline & Scenario Metrics Table")
    st.markdown("""
    This table summarizes key aggregate metrics for the portfolio under both baseline (unstressed) and stressed conditions, along with the absolute change due to the applied scenario.
    *   **Mean PD Baseline**: Average Probability of Default before stress.
    *   **Mean PD Stressed**: Average Probability of Default after applying the stress scenario.
    *   **Delta Mean PD**: Change in average PD (Stressed - Baseline).
    *   **Baseline EL**: Total Expected Loss before stress (sum of PD * Exposure).
    *   **Stressed EL**: Total Expected Loss after stress.
    *   **Delta EL**: Change in Total Expected Loss (Stressed - Baseline).
    """)

    # Calculate aggregate metrics
    mean_pd_baseline = df_stressed['PD_baseline'].mean()
    mean_pd_stressed = df_stressed['PD_stressed'].mean()
    delta_mean_pd = mean_pd_stressed - mean_pd_baseline

    baseline_el = df_stressed['EL_baseline'].sum()
    stressed_el = df_stressed['EL_stressed'].sum()
    delta_el = stressed_el - baseline_el

    metrics_data = {
        "Metric": ["Mean PD Baseline", "Mean PD Stressed", "Delta Mean PD", "Baseline EL", "Stressed EL", "Delta EL"],
        "Value": [mean_pd_baseline, mean_pd_stressed, delta_mean_pd, baseline_el, stressed_el, delta_el]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index("Metric"), use_container_width=True)


    st.subheader("Distribution Shifts Plots")
    st.markdown(r"""
    These plots illustrate the impact of the stress scenario on the distribution of Probability of Default (PD) and the distribution of the change in PD ($ \Delta \hat{y}^{(s)} $).
    *   The first plot compares the kernel density estimates of baseline PD and stressed PD.
    *   The second plot shows the distribution of the absolute change in PD for individual instances.
    """)

    # Plotly Histograms for PD_baseline, PD_stressed
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=df_stressed['PD_baseline'], name='Baseline PD', histnorm='probability density', opacity=0.7))
    fig_dist.add_trace(go.Histogram(x=df_stressed['PD_stressed'], name='Stressed PD', histnorm='probability density', opacity=0.7))
    fig_dist.update_layout(barmode='overlay', title=r'$	ext{Distribution of PDs (Baseline vs. Stressed)}$', xaxis_title='Probability of Default', yaxis_title='Density')
    st.plotly_chart(fig_dist, use_container_width=True)

    # Plotly Histogram for Delta_PD
    fig_delta_pd = px.histogram(df_stressed, x='Delta_PD', nbins=50, title=r'$	ext{Distribution of Change in PD (}\\Delta \hat{y}^{(s)}	ext{)}$', histnorm='probability density')
    fig_delta_pd.update_layout(xaxis_title='Change in PD', yaxis_title='Density')
    st.plotly_chart(fig_delta_pd, use_container_width=True)


    st.subheader("Sensitivity Trajectories Plot")
    st.markdown(r"""
    This line plot shows how the mean stressed Probability of Default changes as the shock intensity parameter ($\alpha$) varies from 0 (no shock) to 1 (full shock). This helps understand the model's sensitivity to increasing levels of stress for the selected scenario.
    *   A steeper curve indicates higher sensitivity to the shock.
    *   The formula for alpha path is $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$ for multiplicative shocks or $x^{(s,\alpha)} = x + \alpha \cdot \delta_s$ for additive shocks.
    """)

    # Re-run stress test for varying alpha to get sensitivity trajectory
    alpha_values = np.linspace(0.0, 1.0, 11) # From 0.0 to 1.0 in 0.1 increments
    mean_pd_stressed_trajectory = []

    from application_pages.page2 import apply_feature_shock, scenario_shocks_config # Import necessary functions from page2

    # Determine the actual shock type and shocks_dict from session state or default
    if st.session_state.get('selected_scenario_option') == "Custom Scenario":
        current_shocks_dict = {item['feature']: item['value'] for item in st.session_state['custom_shocks']}
        current_shock_type = st.session_state['custom_shock_type'].lower()
    else:
        scenario_config = scenario_shocks_config[st.session_state.get('last_run_scenario', 'Macro Downturn')]
        current_shocks_dict = scenario_config['shocks']
        current_shock_type = scenario_config['shock_type'].lower()

    for alpha in alpha_values:
        X_baseline_for_stress = st.session_state['df_baseline'][numerical_features].copy()
        X_stressed_alpha = apply_feature_shock(X_baseline_for_stress, current_shocks_dict, current_shock_type, alpha)
        y_hat_stressed_alpha = mock_model.predict(X_stressed_alpha)
        mean_pd_stressed_trajectory.append(np.mean(y_hat_stressed_alpha))

    fig_trajectory = px.line(
        x=alpha_values,
        y=mean_pd_stressed_trajectory,
        markers=True,
        title=r'$	ext{Mean Stressed PD Sensitivity to Shock Intensity (}\\alpha	ext{)}$',
        labels={'x': r'$\\alpha$ (Shock Intensity)', 'y': 'Mean Stressed PD'}
    )
    st.plotly_chart(fig_trajectory, use_container_width=True)


    st.subheader("Portfolio Heatmaps")
    st.markdown(r"""
    These heatmaps display aggregated impact metrics (e.g., mean $\Delta \text{PD}$, $\Delta \text{EL}$) across different segments of the portfolio (e.g., `rating_band`, `sector`, `region`). This helps identify which segments are most vulnerable to the applied stress scenario.
    *   **Mean $\Delta \text{PD}$ Heatmap**: Shows the average change in PD for each segment.
    *   **$\Delta \text{EL}$ Heatmap**: Shows the total change in Expected Loss for each segment.
    """)

    aggregation_segment = st.selectbox(
        "Select Aggregation Segment",
        options=['rating_band', 'sector', 'region'],
        key="aggregation_segment",
        help="Choose a categorical feature to aggregate stress impact metrics."
    )

    if aggregation_segment in df_stressed.columns:
        # Group by selected segment and calculate mean Delta_PD and sum Delta_EL
        grouped_impact = df_stressed.groupby(aggregation_segment).agg(
            Mean_Delta_PD=('Delta_PD', 'mean'),
            Delta_EL=('Delta_EL', 'sum')
        ).reset_index()

        # Heatmap for Mean Delta_PD
        fig_heatmap_pd = px.bar(
            grouped_impact,
            x=aggregation_segment,
            y='Mean_Delta_PD',
            color='Mean_Delta_PD',
            color_continuous_scale=px.colors.sequential.RdBu,
            title=f'Mean $\Delta$PD by {aggregation_segment}',
            labels={'Mean_Delta_PD': 'Mean Change in PD'}
        )
        st.plotly_chart(fig_heatmap_pd, use_container_width=True)

        # Heatmap for Delta_EL
        fig_heatmap_el = px.bar(
            grouped_impact,
            x=aggregation_segment,
            y='Delta_EL',
            color='Delta_EL',
            color_continuous_scale=px.colors.sequential.RdBu,
            title=f'$\Delta$EL by {aggregation_segment}',
            labels={'Delta_EL': 'Change in Expected Loss'}
        )
        st.plotly_chart(fig_heatmap_el, use_container_width=True)

    else:
        st.warning(f"Aggregation segment '{aggregation_segment}' not found in data.")


    st.subheader("Threshold Flags Table")
    st.markdown("""
    This table highlights individual instances where the absolute change in Probability of Default ($\Delta \text{PD}$) exceeds a predefined threshold. Such instances indicate significant model sensitivity or potential vulnerabilities at an individual level under the stress scenario.
    """)

    delta_pd_threshold = st.number_input(
        "Set \"Delta PD\" Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        key="delta_pd_threshold_input",
        help="Instances with an absolute change in PD greater than this threshold will be flagged."
    )

    flagged_instances = df_stressed[df_stressed['Delta_PD'].abs() > delta_pd_threshold].copy()

    if not flagged_instances.empty:
        st.info(f"Found {len(flagged_instances)} instances where absolute \"Delta PD\" > {delta_pd_threshold}.")
        st.dataframe(flagged_instances[['income', 'debt_to_income', 'PD_baseline', 'PD_stressed', 'Delta_PD', 'exposure', 'Delta_EL', 'rating_band', 'sector', 'region']].sort_values(by='Delta_PD', ascending=False))
    else:
        st.success(f"No instances found where absolute \"Delta PD\" > {delta_pd_threshold}.")
