
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown(r"""
# Financial Model Stress Analyzer

This Streamlit application will enable financial professionals to conduct scenario-based robustness tests on Machine Learning (ML) models. It allows users to define stress scenarios, apply them as quantitative input shocks to model features, and observe their impact on model predictions and aggregated portfolio metrics. The target audience includes quantitative analysts, model developers, risk managers, and validators.

### Learning Goals

Upon completion of this application, participants will be able to:
*   Understand and apply scenario-based stress testing directly to ML models via feature-level transformations $T_s(x)$.
*   Translate abstract financial scenarios (e.g., recession, market stress) into concrete quantitative shocks ($\delta_s$) applied to input features.
*   Measure and interpret ML model sensitivity to these structured shocks using risk-friendly metrics such as Probability of Default (PD), Value-at-Risk (VaR), and Expected Loss (EL).
*   Identify model behaviors that are fragile or insufficiently conservative under plausible adverse financial environments.
*   Integrate scenario-based robustness tests into a broader model validation and governance workflow, making ML model robustness tangible and auditable.

## Methodology Overview

This notebook employs a scenario-based stress testing methodology for Machine Learning (ML) models in finance. The core idea is to subject an ML model to structured, economically meaningful shocks applied directly to its input features, thereby simulating adverse financial environments. The model's response to these shocks is then measured using various risk metrics.

Here's a plain-English outline of the approach:

1.  **Synthetic Data Generation**: Create a realistic, yet controlled, dataset of financial features and a target variable (e.g., Probability of Default).
2.  **Mock ML Model**: Implement a placeholder ML model that provides predictions based on a fixed function of input features, bypassing the need for actual model training.
3.  **Baseline Prediction**: Obtain predictions from the mock model on the original, unstressed dataset to establish a reference point ($\hat{y} = f_\theta(x)$).
4.  **Stress Transformation Library**: Develop functions to apply quantitative shocks to input features, either multiplicatively ($x^{(s)} = x \odot (1 + \delta_s)$) or additively ($x^{(s)} = x + \delta_s$).
5.  **Scenario Parameterization**: Define concrete financial stress scenarios (e.g., "Macro Downturn") and map them to specific quantitative shocks ($\delta_s$) on input features.
6.  **Scenario Application & Stressed Predictions**: Apply defined shocks to baseline features to create stressed datasets ($x^{(s)}$), and then generate stressed predictions ($\hat{y}^{(s)} = f_\theta(x^{(s)})$) using the mock ML model.
7.  **Impact Metrics Calculation**: Quantify the difference between stressed and baseline predictions using metrics like individual prediction change ($\Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y}$), change in mean PD ($\Delta \overline{PD}$), and change in Expected Loss ($\Delta EL$).
8.  **Portfolio Aggregation**: Aggregate impact metrics by various portfolio segments (e.g., rating band, sector) to identify differential impacts.
9.  **Sensitivity Analysis (Alpha Path)**: Examine model response to gradually increasing shock intensity ($\alpha$) where $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$.
10. **Visualization**: Generate plots and tables to effectively communicate the results of the stress tests, including distribution shifts, sensitivity trajectories, and segmented impact heatmaps.

### Key Formulae

*   **Multiplicative Shock**: This transformation scales a feature $x$ by a factor determined by the shock $\delta_s$. This is particularly useful for modeling percentage changes in financial variables.
    $$x^{(s)} = x \odot (1 + \delta_s)$$
    Where $x^{(s)}$ is the stressed feature, $x$ is the baseline feature, and $\delta_s$ is the scenario-specific shock factor (e.g., 0.10 for a 10% increase, -0.15 for a 15% decrease).

*   **Additive Shock**: This transformation shifts a feature $x$ by a constant value $\delta_s$. This is suitable for modeling absolute changes.
    $$x^{(s)} = x + \delta_s$$
    Where $\delta_s$ is the scenario-specific additive shock value.

*   **Individual Prediction Change**: This metric quantifies the impact of stress on a single prediction by taking the difference between the stressed prediction and the baseline prediction.
    $$\Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y}$$
    Where $\hat{y}^{(s)}$ is the stressed prediction and $\hat{y}$ is the baseline prediction.

*   **Expected Loss (EL)**: A critical portfolio-level risk metric. For simplicity in this notebook, we define EL as the sum of `PD * exposure` across the portfolio. The change in EL ($\Delta EL^{(s)}$) is then the difference between stressed EL and baseline EL.
    $$EL = \sum (PD \times Exposure)$$
    $$\Delta EL^{(s)} = EL^{(s)} - EL_{baseline}$$
    Where $PD$ is the Probability of Default (our model output), and $Exposure$ is the financial exposure associated with each entity. This formula provides a direct financial impact measure, crucial for capital allocation and risk provisioning.

These formulae and the overall methodology provide a robust framework for assessing ML model stability and identifying potential vulnerabilities under various financial stress conditions, directly supporting prudent risk management and model validation.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Generation & Model Setup", "Scenario Definition & Stress Test", "Impact Analysis & Visualizations"])
if page == "Data Generation & Model Setup":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Scenario Definition & Stress Test":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Impact Analysis & Visualizations":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
