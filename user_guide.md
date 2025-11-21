id: 6920dd16680d55964bf2ffea_user_guide
summary: Scenario-Based Model Robustness Test - 3 User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Financial Model Stress Analyzer: A User Guide to ML Model Robustness

## 1. Understanding Scenario-Based Stress Testing for ML Models
Duration: 0:10:00

Welcome to the Financial Model Stress Analyzer! This application is designed to empower financial professionals, such as quantitative analysts, model developers, risk managers, and validators, to robustly test their Machine Learning (ML) models against various economic and market scenarios. In today's dynamic financial landscape, understanding how ML models behave under stress is paramount for prudent risk management and regulatory compliance.

<aside class="positive">
<b>Why is this important?</b> Financial ML models are often critical for decisions like credit scoring, fraud detection, and portfolio optimization. Without rigorous stress testing, these models could fail unexpectedly in adverse conditions, leading to significant financial losses or systemic risk. This application provides a hands-on way to explore these vulnerabilities.
</aside>

**Learning Goals:**

Upon completing this guide, you will be able to:
*   Understand and apply scenario-based stress testing directly to ML models via feature-level transformations $T_s(x)$.
*   Translate abstract financial scenarios (e.g., recession, market stress) into concrete quantitative shocks ($\delta_s$) applied to input features.
*   Measure and interpret ML model sensitivity to these structured shocks using risk-friendly metrics such as Probability of Default (PD), Value-at-Risk (VaR), and Expected Loss (EL).
*   Identify model behaviors that are fragile or insufficiently conservative under plausible adverse financial environments.
*   Integrate scenario-based robustness tests into a broader model validation and governance workflow, making ML model robustness tangible and auditable.

### Methodology Overview

This application employs a scenario-based stress testing methodology for Machine Learning (ML) models in finance. The core idea is to subject an ML model to structured, economically meaningful shocks applied directly to its input features, thereby simulating adverse financial environments. The model's response to these shocks is then measured using various risk metrics.

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

## 2. Generating Synthetic Financial Data
Duration: 0:05:00

In this step, we will generate a synthetic dataset that mimics a credit portfolio. This is crucial for demonstrating stress testing concepts without the complexities and sensitivities associated with real-world financial data. By generating controlled data, we can ensure reproducibility and isolate the effects of stress transformations on model predictions, thereby providing a clear and interpretable framework for model validation and risk assessment.

**Business Value:** Using synthetic data allows us to focus on the methodology of stress testing and model response analysis, rather than the challenges of real data acquisition and anonymization. The generated data supports the business goal of creating a robust framework for validating ML models under adverse conditions.

1.  **Navigate to the "Data Generation & Model Setup" Page:**
    In the sidebar, under "Navigation", select "Data Generation & Model Setup".

2.  **Configure and Generate Data:**
    On the left sidebar, you will find a slider labeled "Number of Samples".
    *   Adjust the slider to choose the number of synthetic data samples you wish to generate (e.g., 5000).
    *   Click the "Generate Synthetic Data" button.

<aside class="positive">
The application will now generate a DataFrame with various financial features like `income`, `debt_to_income`, `utilization`, `LTV`, `credit_spread`, `house_price`, `volatility`, `liquidity`, `exposure`, and categorical features like `rating_band`, `sector`, `region`. It also simulates a `PD_target` (Probability of Default) based on these features.
</aside>

3.  **Review the Generated Data:**
    Once the data is generated, you will see sections like "Generated Data Overview", "Data Information (`df.info()` summary)", and "Descriptive Statistics (`df.describe()`)".
    *   **"Generated Data Overview"**: Provides a preview of the first few rows, showing you the structure and content of your simulated credit portfolio.
    *   **"Data Information"**: Summarizes data types and non-null counts, confirming that our synthetic dataset is complete.
    *   **"Descriptive Statistics"**: Offers key statistics (mean, std, min, max, quartiles) for numerical features. This is useful for understanding the typical range and distribution of each feature in your simulated portfolio, setting expectations for "normal" conditions.

<aside class="positive">
<b>Interpretation:</b> Pay attention to features like `income` and `house_price` which often follow log-normal-like distributions, and ratios like `debt_to_income` and `utilization` which are typically bounded. The `PD_target` provides a baseline risk profile for the portfolio under normal conditions.
</aside>

## 3. Implementing the Mock ML Model and Baseline Predictions
Duration: 0:07:00

With our synthetic data ready, the next step is to set up a placeholder ML model and establish a baseline for its predictions.

1.  **Understanding the Mock ML Model:**
    Scroll down on the "Data Generation & Model Setup" page to "5.2. Implementing a Mock ML Model".
    *   This application uses a `MockMLModel` which is a simplified representation of a real ML model. It's not trained in the traditional sense; instead, it uses a predefined mathematical function and fixed coefficients to generate Probability of Default (PD) predictions.
    *   **Business Value**: This approach allows us to precisely control the model's behavior and isolate the impact of stress scenarios on its output, without the complexities of training a real model. It ensures reproducibility and makes the cause-and-effect of stress testing clear.
    *   The application displays the features the mock model uses (e.g., `income`, `debt_to_income`, `utilization`) and a preview of the input data (`X_baseline`) for these predictions.

2.  **Calculating Baseline Predictions:**
    Continue scrolling to "5.3. Calculating Baseline Predictions".
    *   The application automatically calculates the baseline predictions (`PD_baseline`) using the `MockMLModel` on the original, unstressed data. This is done automatically if not already calculated.
    *   **Business Value**: `PD_baseline` serves as our critical reference point. It represents the model's assessment of default risk under "business-as-usual" conditions. All subsequent analyses of stress impact will compare stressed predictions against this baseline.

3.  **Review Baseline Predictions:**
    The application will show a preview of the `df_baseline` DataFrame, now including the `PD_baseline` column. It also presents the mean and standard deviation of `PD_baseline`.
    *   **Mean `PD_baseline`**: Gives the average expected default rate for your entire portfolio under normal circumstances.
    *   **Standard Deviation `PD_baseline`**: Indicates how much individual PDs vary around the average, giving insight into the homogeneity of risk in the portfolio.

<aside class="positive">
<b>Key Takeaway:</b> Having a clear and stable `PD_baseline` is fundamental. It's the benchmark against which we will measure any shifts caused by adverse financial scenarios. A significant change from this baseline under stress will signal potential model fragility.
</aside>

## 4. Defining Stress Scenarios and Transformations
Duration: 0:08:00

Now that we have our data and a baseline model, we move to the core of stress testing: defining what a "stress scenario" actually means for our model's inputs.

1.  **Navigate to the "Scenario Definition & Stress Test" Page:**
    In the sidebar, under "Navigation", select "Scenario Definition & Stress Test".

2.  **Understanding Stress Transformations:**
    The page begins by explaining "5.4. Defining the Stress Transformation Library".
    *   Stress scenarios are translated into quantitative feature transformations, $T_s(x)$. These transformations modify the input features ($x$) based on predefined shock parameters ($\delta_s$).
    *   We use two main types:
        *   **Multiplicative Shock**: $x^{(s)} = x \odot (1 + \delta_s)$ - this scales features by a percentage (e.g., 10% increase in `credit_spread`).
        *   **Additive Shock**: $x^{(s)} = x + \delta_s$ - this shifts features by a fixed absolute value (e.g., +0.02 to `utilization`).
    *   **Business Value**: This mechanism allows us to convert abstract financial narratives (like "recession") into concrete, measurable changes in the data that feed our ML model, making the stress test quantifiable and auditable.

3.  **Explore Pre-defined Scenarios:**
    The application provides "Pre-defined Scenario Configurations".
    *   You'll see a JSON-like structure outlining scenarios like "Macro Downturn", "Market Stress", and "Housing Market Downturn". Each scenario specifies its `shock_type` (multiplicative or additive) and the `shocks` (feature-value pairs, e.g., `income: -0.15` for a 15% income reduction).
    *   These scenarios represent plausible adverse events a financial institution might face.

4.  **Select a Stress Scenario:**
    In the sidebar, locate the "Scenario Definition & Application" section.
    *   Use the "Select a Stress Scenario" dropdown. You can choose one of the pre-defined scenarios or select "Custom Scenario" to define your own.

5.  **Define a Custom Scenario (Optional):**
    If you choose "Custom Scenario":
    *   Expand the "Define Custom Scenario" section.
    *   Select the `shock_type` (Multiplicative or Additive) for your custom shocks.
    *   Use the "Add Feature-Shock Pairs" interface:
        *   Enter a `New Feature Name` (e.g., `income`, `LTV`).
        *   Enter a `New Shock Value` (e.g., `-0.10` for a 10% multiplicative decrease, or `0.05` for an additive shift).
        *   Click "Add Custom Shock". You can add multiple feature-shock pairs and remove existing ones.

<aside class="negative">
When defining a custom scenario, ensure the feature names you enter match the features in your generated dataset (e.g., `income`, `debt_to_income`). Mismatched names will not apply the shock.
</aside>

6.  **Adjust Shock Intensity ($\alpha$):**
    Below the scenario selection, find the "Shock Intensity ($\alpha$)" slider.
    *   This slider allows you to control the magnitude of the applied shocks. $\alpha=0$ means no shock is applied, while $\alpha=1$ applies the full shock defined in the scenario.
    *   The formula shown, $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$ (for multiplicative) or $x^{(s,\alpha)} = x + \alpha \cdot \delta_s$ (for additive), demonstrates how $\alpha$ scales the shock.
    *   **Business Value**: This parameter is crucial for sensitivity analysis, allowing you to observe how your model responds to gradually increasing levels of stress.

## 5. Running the Stress Test
Duration: 0:03:00

After defining your scenario and shock intensity, it's time to run the stress test and see its immediate impact on model predictions.

1.  **Execute the Stress Test:**
    In the sidebar, click the "Run Stress Test" button.
    *   The application will take your selected scenario (or custom scenario), apply the defined shocks to the baseline data with the chosen $\alpha$ intensity, and then generate new predictions using the `MockMLModel`.

2.  **What Happens During the Stress Test:**
    *   The input features (`X_baseline`) are transformed into `X_stressed` using the specified shock type and values, scaled by your chosen $\alpha$.
    *   The `MockMLModel` then calculates `PD_stressed` (stressed Probability of Default) based on `X_stressed`.
    *   The application calculates `Delta_PD`, which is the difference between `PD_stressed` and `PD_baseline` for each individual instance. This shows how much each individual prediction changed due to the stress.
    *   It also calculates `EL_baseline` (Expected Loss before stress) and `EL_stressed` (Expected Loss after stress) using the formula $EL = PD \times Exposure$. `Delta_EL` is then calculated.

3.  **Review Initial Stressed Results:**
    Once the stress test is complete, the application will display a message confirming completion and show a preview of the stressed data.
    *   You'll see columns like `PD_baseline`, `PD_stressed`, `Delta_PD`, `EL_baseline`, `EL_stressed`, and `Delta_EL` for the first few rows of your portfolio.
    *   This immediate preview gives you a first glimpse of how individual PDs and Expected Losses have shifted under the chosen stress scenario.

<aside class="positive">
<b>Actionable Insight:</b> Observe if `PD_stressed` values are generally higher or lower than `PD_baseline` depending on your scenario. A positive `Delta_PD` means an increased risk under stress, which is often the expected outcome in adverse scenarios.
</aside>

## 6. Analyzing Aggregate Impacts and Distribution Shifts
Duration: 0:08:00

Now that we have run a stress test, it's time to deeply analyze the results. We will start by looking at aggregated portfolio metrics and how the overall distribution of risk changes.

1.  **Navigate to the "Impact Analysis & Visualizations" Page:**
    In the sidebar, under "Navigation", select "Impact Analysis & Visualizations".

2.  **Review Baseline & Scenario Metrics Table:**
    The first section on this page is the "Baseline & Scenario Metrics Table".
    *   This table provides a concise summary of key portfolio-level metrics:
        *   **Mean PD Baseline**: The average Probability of Default across the entire portfolio before stress.
        *   **Mean PD Stressed**: The average Probability of Default after applying the stress scenario.
        *   **Delta Mean PD**: The absolute change in the average PD (Stressed - Baseline). This is a critical indicator of the overall impact on portfolio credit quality.
        *   **Baseline EL**: The total Expected Loss for the portfolio before stress.
        *   **Stressed EL**: The total Expected Loss for the portfolio after stress.
        *   **Delta EL**: The change in total Expected Loss (Stressed - Baseline). This metric provides a direct financial quantification of the scenario's impact, crucial for capital planning and risk provisioning.

<aside class="positive">
<b>Business Interpretation:</b> A positive `Delta Mean PD` and `Delta EL` indicate that the chosen scenario has worsened the portfolio's risk profile, leading to higher expected defaults and losses. The magnitude of these deltas tells you how severe the impact is.
</aside>

3.  **Examine Distribution Shifts Plots:**
    Below the metrics table, you'll find "Distribution Shifts Plots".
    *   **Distribution of PDs (Baseline vs. Stressed)**: This plot uses histograms (or kernel density estimates) to show the overall shape and spread of individual PDs before and after the stress. You can visually observe if the distribution has shifted to higher PDs, become wider, or changed shape.
    *   **Distribution of Change in PD ($\Delta \hat{y}^{(s)}$)**: This plot specifically shows the distribution of the individual changes in PD. A distribution skewed towards positive values indicates that most instances saw an increase in their PD, while a wider spread might suggest heterogeneous impacts.

<aside class="positive">
<b>Visual Insight:</b> These plots help you understand not just the average impact, but also the dispersion of the impact across the portfolio. For example, a stress might significantly increase PD for a few customers while leaving others relatively unaffected.
</aside>

## 7. Understanding Sensitivity Trajectories and Segmented Impacts
Duration: 0:08:00

This step focuses on understanding how the model's output changes with varying stress intensity and identifying which parts of your portfolio are most vulnerable.

1.  **Analyze Sensitivity Trajectories Plot:**
    Scroll down to "Sensitivity Trajectories Plot".
    *   This line plot shows how the **mean stressed Probability of Default** changes as the **shock intensity ($\alpha$)** gradually increases from 0 (no shock) to 1 (full shock).
    *   Recall that $\alpha$ scales your defined shocks: $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$ or $x^{(s,\alpha)} = x + \alpha \cdot \delta_s$.
    *   **Business Value**: This plot is crucial for sensitivity analysis. A steeper curve indicates that the model (and thus the portfolio's risk) is highly sensitive to even small increases in stress intensity for the selected scenario. A flatter curve suggests more resilience. It allows risk managers to assess the non-linear response of their models.

2.  **Explore Portfolio Heatmaps (Segmented Analysis):**
    Next, you'll find "Portfolio Heatmaps".
    *   These heatmaps help identify which specific segments of your portfolio are most affected by the stress scenario.
    *   **Select Aggregation Segment**: Use the dropdown to choose a categorical feature for aggregation, such as `rating_band`, `sector`, or `region`.
    *   **Mean $\Delta$PD by Segment**: This heatmap (displayed as a bar chart) shows the average change in PD for each category within the selected segment.
    *   **$\Delta$EL by Segment**: This heatmap shows the total change in Expected Loss for each category within the selected segment.
    *   **Business Value**: By segmenting the impact, you can pinpoint specific areas of your portfolio (e.g., "Technology" sector, "BBB" rating band, "South" region) that are particularly vulnerable to a given stress scenario. This is invaluable for targeted risk mitigation strategies, capital allocation, and portfolio rebalancing.

<aside class="positive">
<b>Actionable Insight:</b> If the "Financials" sector shows a significantly higher $\Delta$EL compared to other sectors under a "Macro Downturn" scenario, it signals a concentrated risk that may require deeper investigation or hedging strategies.
</aside>

## 8. Identifying Individual Vulnerabilities with Threshold Flags
Duration: 0:04:00

Finally, we zoom in to identify individual instances that exhibit significant changes under stress, which can highlight specific model vulnerabilities or high-risk accounts.

1.  **Configure Threshold Flags Table:**
    Scroll down to "Threshold Flags Table".
    *   This section allows you to flag individual instances (e.g., customer accounts, loans) where the model's prediction changed substantially due to the stress.
    *   **Set "Delta PD" Threshold**: Use the number input to define a threshold (e.g., 0.10). Any instance where the absolute change in PD (`Delta_PD`) is greater than this value will be flagged.
    *   **Business Value**: While aggregate metrics give a portfolio-level view, threshold flags highlight specific "outliers" or highly sensitive individual cases. This is crucial for granular model validation and for identifying individual accounts that might become problematic under stress.

2.  **Review Flagged Instances:**
    The application will display a table of all instances that exceed your defined `Delta_PD` threshold.
    *   This table typically includes key features (`income`, `debt_to_income`), baseline and stressed PDs (`PD_baseline`, `PD_stressed`), the change in PD (`Delta_PD`), exposure (`exposure`), change in EL (`Delta_EL`), and categorical segments (`rating_band`, `sector`, `region`).

<aside class="negative">
A large number of flagged instances or very high `Delta_PD` values for specific instances could indicate that your model is highly volatile for certain types of inputs under stress, or that the scenario disproportionately impacts specific segments. This warrants further investigation into both the model's behavior and the underlying portfolio characteristics.
</aside>

## 9. Conclusion and Next Steps
Duration: 0:02:00

Congratulations! You have successfully navigated the Financial Model Stress Analyzer application.

You have learned to:
*   Generate synthetic financial data to simulate a credit portfolio.
*   Understand and utilize a mock ML model to establish baseline predictions.
*   Define and apply various stress scenarios using both multiplicative and additive shocks, adjusting their intensity.
*   Analyze the aggregate impact of stress scenarios on portfolio-level metrics like Mean PD and Expected Loss.
*   Visualize distribution shifts and model sensitivity to increasing stress.
*   Identify vulnerable portfolio segments through heatmaps and pinpoint individual high-impact instances using threshold flags.

This comprehensive approach allows financial professionals to systematically evaluate the robustness of their ML models, gain deeper insights into their behavior under adverse conditions, and make more informed risk management decisions.

**Further Exploration:**
*   Experiment with different custom scenarios, combining various features and shock types.
*   Observe how changing the `Number of Samples` impacts the stability of the results (though this might be less directly related to stress testing concepts).
*   Consider how you might translate real-world economic forecasts into the quantitative shocks (`$\delta_s$`) used in this application.

By continuously challenging your ML models with plausible stress scenarios, you can build greater confidence in their reliability and integrate them more securely into critical financial workflows.
