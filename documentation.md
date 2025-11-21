id: 6920dd16680d55964bf2ffea_documentation
summary: Scenario-Based Model Robustness Test - 3 Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Financial Model Stress Analyzer: A Codelab for ML Model Robustness
## 1. Introduction: Understanding Financial Model Stress Analysis
Duration: 0:05

This codelab introduces you to **QuLab: Financial Model Stress Analyzer**, a Streamlit application designed to help financial professionals conduct robustness tests on Machine Learning (ML) models using scenario-based stress testing. In today's dynamic financial landscape, understanding how ML models behave under adverse conditions is paramount for risk management, regulatory compliance, and informed decision-making. This application provides a hands-on guide to defining stress scenarios, applying quantitative shocks to model features, and analyzing the impact on model predictions and key portfolio metrics.

<aside class="positive">
<b>Why is this important?</b> Financial models, especially those powered by ML, are increasingly used for critical tasks like credit scoring, fraud detection, and portfolio optimization. However, these models might perform unpredictably when faced with extreme market events or economic downturns—conditions they may not have encountered during training. Stress testing helps uncover these vulnerabilities proactively, ensuring models remain reliable and conservative under adverse scenarios.
</aside>

### Learning Goals

Upon completion of this application, participants will be able to:
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

### Application Architecture
The Streamlit application is structured into three main pages, accessible via the sidebar navigation. This modular design helps in logically separating the concerns of data preparation, scenario application, and results analysis.

<p align="center">
  <img src="https://i.imgur.com/W2dY355.png" alt="Application Architecture Diagram" width="700"/>
</p>

*   **app.py**: The main entry point, handling global configurations, navigation, and rendering the initial introduction. It imports and executes the functions responsible for each page.
*   **application_pages/page1.py**: Focuses on generating synthetic financial data, setting up a mock ML model, and calculating baseline predictions.
*   **application_pages/page2.py**: Manages scenario definition, applying stress transformations to the data, and generating stressed predictions and initial impact metrics.
*   **application_pages/page3.py**: Dedicated to comprehensive impact analysis through various visualizations and aggregated metrics.

Data and model objects are shared across pages using Streamlit's `st.session_state`, ensuring continuity as you navigate the application.

## 2. Setting Up the Environment and Generating Synthetic Data
Duration: 0:10

In this step, you will initialize the application by generating a synthetic financial dataset and performing initial data exploration. This forms the foundational portfolio that will be subjected to stress tests.

### Navigating to the Page
1.  On the left sidebar, ensure "Data Generation & Model Setup" is selected in the "Navigation" dropdown.

### Generating Synthetic Financial Data
The application starts by creating a synthetic dataset that mimics a credit portfolio. This approach is chosen to demonstrate stress testing concepts without the complexities and sensitivities associated with real-world financial data. By generating controlled data, we can ensure reproducibility and isolate the effects of stress transformations on model predictions.

1.  **Adjust Number of Samples**: In the sidebar, use the "Number of Samples" slider to select the desired size of your synthetic dataset. A larger number of samples (e.g., 5000-10000) will provide a more comprehensive dataset for analysis but might take slightly longer to generate.
2.  **Generate Data**: Click the **"Generate Synthetic Data"** button in the sidebar. You'll see a spinner indicating data generation is in progress. Once complete, a success message will appear.

<aside class="positive">
<b>Tip:</b> The `generate_financial_data` function uses `@st.cache_data`. This means if you don't change the "Number of Samples" input, the data will only be generated once, making subsequent runs faster.
</aside>

```python
# From application_pages/page1.py
@st.cache_data
def generate_financial_data(n_samples_input):
    data = {}
    # ... (feature generation logic) ...
    df = pd.DataFrame(data)
    # ... (PD_target calculation logic) ...
    return df
```

### Reviewing the Generated Data
After generation, the application will display an overview of your synthetic data:
*   **Generated Data Overview (`df.head()`)**: Shows the first few rows of the `df_baseline` DataFrame. This provides a glimpse into the features generated, such as `income`, `debt_to_income`, `utilization`, `LTV`, `credit_spread`, `house_price`, `volatility`, `liquidity`, `exposure`, `rating_band`, `sector`, `region`, and the target `PD_target`.
*   **Data Information (`df.info()`)**: Provides a summary of the DataFrame, including the number of entries, column names, non-null counts, and data types. This confirms the dataset's structure and completeness.
*   **Descriptive Statistics (`df.describe()`)**: Presents key statistical measures (mean, std, min, max, quartiles) for all numerical columns. This allows for a quick understanding of the distributions and ranges of your simulated financial features. For example, you can observe the average income, typical debt-to-income ratios, and the range of `PD_target` values.

<aside class="positive">
<b>Interpretation:</b> The initial inspections confirm that our synthetic dataset is well-structured and its statistical properties are consistent with what would be expected in a simulated financial credit portfolio, making it suitable for subsequent stress testing and model analysis.
</aside>

## 3. Implementing a Mock ML Model and Calculating Baseline Predictions
Duration: 0:07

This step focuses on establishing the core components for our stress testing framework: a mock ML model and its baseline predictions under normal conditions.

### Implementing a Mock ML Model
To simulate real-world model interaction without the overhead of actual training, we use a `MockMLModel`. This class acts as a placeholder for a pre-trained ML model, providing predictions based on a predefined mathematical function of its input features.

```python
# From application_pages/page1.py
@st.cache_resource
class MockMLModel:
    def __init__(self):
        self.coefficients = {
            'income': -0.05,
            'debt_to_income': 0.8,
            'utilization': 0.6,
            'LTV': 0.4,
            'credit_spread': 1.0,
            'house_price': -0.02,
            'volatility': 0.7,
            'liquidity': -0.3,
            'exposure': 0.000005, # Small impact on PD for exposure
        }
        self.intercept = -3.0
        self.expected_features = list(self.coefficients.keys())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = X[self.expected_features].copy()
        for col in ['income', 'house_price', 'exposure']:
            if col in X_processed.columns:
                X_processed[col] = np.log1p(X_processed[col])

        log_odds = self.intercept
        for feature, coeff in self.coefficients.items():
            if feature in X_processed.columns:
                log_odds += X_processed[feature] * coeff

        pd_predictions = 1 / (1 + np.exp(-log_odds))
        return np.clip(pd_predictions, 0.001, 0.999)
```

The application automatically instantiates `MockMLModel` and identifies the numerical features it uses (`numerical_features`). A preview of these features (`X_baseline.head()`) is displayed, confirming the input structure for the model.

<aside class="positive">
<b>Business Value:</b> Using a mock model allows us to isolate and precisely measure the impact of feature transformations without confounding effects from model retraining or dynamic learning. It ensures reproducibility and interpretability of stress test results.
</aside>

### Calculating Baseline Predictions
Before any stress is applied, it's crucial to establish a **baseline**. This involves obtaining predictions from our `MockMLModel` on the original, unstressed dataset. These baseline predictions ($\hat{y} = f_\theta(x)$) serve as the fundamental reference point for measuring the impact of all future stress scenarios.

The application calculates `PD_baseline` for each instance in your `df_baseline` and adds it as a new column.

```python
# From application_pages/page1.py
if 'PD_baseline' not in st.session_state:
    y_hat_baseline = mock_model.predict(X_baseline)
    st.session_state['df_baseline']['PD_baseline'] = y_hat_baseline
```

You'll see a preview of `df_baseline` with the new `PD_baseline` column, along with its mean and standard deviation.

<aside class="positive">
<b>Interpretation:</b> The `PD_baseline` reflects the model's assessment of default risk in a "business-as-usual" environment. Its mean and standard deviation provide insights into the inherent risk profile of the portfolio before any adverse events, setting the expectation against which stressed outcomes will be compared.
</aside>

## 4. Defining Stress Scenarios and Applying Feature Shocks
Duration: 0:15

Now that the data and mock model are ready, we will delve into the core of stress testing: defining scenarios and applying quantitative shocks.

### Navigating to the Page
1.  On the left sidebar, select "Scenario Definition & Stress Test" from the "Navigation" dropdown.

### Defining the Stress Transformation Library
Stress scenarios are translated into quantitative feature transformations $T_s(x)$. These transformations modify the input features according to predefined shock parameters. The application provides a function `apply_feature_shock` to handle both multiplicative (scaling) and additive (shifting) shocks.

*   **Multiplicative Shock**: Scales a feature $x$ by a factor $(1 + \delta_s)$. Useful for percentage changes (e.g., income drops by 15%).
    $$x^{(s)} = x \odot (1 + \delta_s)$$
*   **Additive Shock**: Shifts a feature $x$ by a constant value $\delta_s$. Suitable for absolute changes (e.g., interest rate increases by 2%).
    $$x^{(s)} = x + \delta_s$$

```python
# From application_pages/page2.py
def apply_feature_shock(df, shocks_dict, shock_type, alpha_value):
    df_shocked = df.copy()
    for feature, raw_shock_value in shocks_dict.items():
        if feature in df_shocked.columns:
            if shock_type == 'multiplicative':
                df_shocked[feature] = df_shocked[feature] * (1 + alpha_value * raw_shock_value)
            elif shock_type == 'additive':
                df_shocked[feature] = df_shocked[feature] + (alpha_value * raw_shock_value)
    return df_shocked
```

The application displays a `scenario_shocks_config` JSON, outlining several pre-defined stress scenarios (e.g., "Macro Downturn", "Market Stress", "Housing Market Downturn") with their associated shock types and feature-specific shock values.

<aside class="positive">
<b>Business Value:</b> This library provides the mechanism to translate qualitative financial stress narratives (e.g., "recession") into concrete, quantifiable changes in the data that feed into the ML model, ensuring consistent and auditable stress test results.
</aside>

### Applying Stress Scenarios
The sidebar on this page allows you to define and apply a stress scenario.

<p align="center">
  <img src="https://i.imgur.com/bQo9v6u.png" alt="Scenario Definition Flowchart" width="700"/>
</p>

1.  **Select a Stress Scenario**:
    *   Choose one of the pre-defined scenarios (e.g., "Macro Downturn", "Market Stress", "Housing Market Downturn") from the dropdown.
    *   **Or, Define a Custom Scenario**: If you select "Custom Scenario", an expander will appear.
        *   **Shock Type**: Select "Multiplicative" or "Additive".
        *   **Add Feature-Shock Pairs**: Use the form to add new features and their corresponding shock values. You can also remove existing custom shocks. Ensure the feature names exactly match those in your `df_baseline`.

2.  **Adjust Shock Intensity ($\alpha$)**: Use the "Shock Intensity ($\alpha$)" slider. This parameter scales the magnitude of the defined shocks.
    *   $\alpha = 0.0$: No shock is applied; stressed predictions will equal baseline predictions.
    *   $\alpha = 1.0$: The full shock defined in the scenario is applied.
    *   Values between 0.0 and 1.0 apply a partial shock, allowing for sensitivity analysis. The formula used is $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$ for multiplicative shocks or $x^{(s,\alpha)} = x + \alpha \cdot \delta_s$ for additive shocks.

3.  **Run Stress Test**: Click the **"Run Stress Test"** button. The application will:
    *   Apply the selected shocks to the numerical features of your baseline data.
    *   Generate `PD_stressed` using the `MockMLModel`.
    *   Calculate key impact metrics:
        *   `Delta_PD = PD_stressed - PD_baseline`
        *   `EL_baseline = PD_baseline * exposure`
        *   `EL_stressed = PD_stressed * exposure`
        *   `Delta_EL = EL_stressed - EL_baseline`
    *   Store the `df_stressed` DataFrame in the session state for further analysis.

After a successful run, a success message will confirm the scenario applied and its alpha value. A preview of the `df_stressed` DataFrame, including the newly calculated `PD_stressed`, `Delta_PD`, `EL_baseline`, `EL_stressed`, and `Delta_EL` columns, will be displayed.

<aside class="negative">
<b>Warning:</b> If you are defining a custom scenario, ensure that the feature names you enter match the columns in your generated dataset. Incorrect names will lead to those features not being shocked.
</aside>

## 5. Analyzing Stress Test Impacts and Visualizing Results
Duration: 0:18

This final step is dedicated to understanding the implications of the stress scenario through various visualizations and aggregated metrics.

### Navigating to the Page
1.  On the left sidebar, select "Impact Analysis & Visualizations" from the "Navigation" dropdown.

<aside class="negative">
<b>Prerequisite:</b> Ensure you have run a stress test on the "Scenario Definition & Stress Test" page first. Otherwise, a warning message will appear, and no results will be displayed.
</aside>

### Baseline & Scenario Metrics Table
This table provides an immediate, high-level summary of the portfolio's performance under both normal (baseline) and stressed conditions.
*   **Mean PD Baseline**: The average Probability of Default across the portfolio before stress.
*   **Mean PD Stressed**: The average Probability of Default after applying the stress scenario.
*   **Delta Mean PD**: The absolute change in average PD (Stressed - Baseline). A positive value indicates an increase in average default risk.
*   **Baseline EL**: Total Expected Loss for the portfolio before stress.
*   **Stressed EL**: Total Expected Loss after stress.
*   **Delta EL**: The absolute change in Total Expected Loss (Stressed - Baseline). This is a critical metric for financial impact assessment, directly indicating the additional loss expected under the stress scenario.

```python
# From application_pages/page3.py
mean_pd_baseline = df_stressed['PD_baseline'].mean()
mean_pd_stressed = df_stressed['PD_stressed'].mean()
delta_mean_pd = mean_pd_stressed - mean_pd_baseline

baseline_el = df_stressed['EL_baseline'].sum()
stressed_el = df_stressed['EL_stressed'].sum()
delta_el = stressed_el - baseline_el
```
Review this table to quickly grasp the overall impact of the chosen stress scenario on the portfolio's risk profile.

### Distribution Shifts Plots
These plots visually represent how the stress scenario alters the distribution of model predictions.

1.  **Distribution of PDs (Baseline vs. Stressed)**: A histogram (or KDE plot) comparing the Probability of Default distributions before and after stress. Look for shifts in the peak, changes in spread, or the emergence of a fatter tail in the stressed distribution, which would indicate a higher prevalence of default.
2.  **Distribution of Change in PD ($\Delta \hat{y}^{(s)}$)**: A histogram showing the distribution of the individual changes in PD. A distribution skewed towards positive values indicates that most instances experienced an increase in PD due to the stress.

### Sensitivity Trajectories Plot
This plot illustrates the model's sensitivity to varying intensities of the applied shock.

*   The horizontal axis represents the shock intensity ($\alpha$), ranging from 0.0 (no shock) to 1.0 (full shock).
*   The vertical axis shows the Mean Stressed PD.
*   **Interpretation**: A steeper upward slope indicates that the model's predicted PD increases significantly even with small increases in shock intensity, signaling high sensitivity. A flatter curve suggests the model is more robust to increasing stress levels for that particular scenario. This plot helps identify critical thresholds of shock intensity where the model's output might drastically change.

<aside class="positive">
<b>How it works:</b> To generate this plot, the application re-runs the stress test internally for multiple `alpha` values (0.0, 0.1, 0.2, ..., 1.0) and plots the mean PD for each `alpha`. This is an example of an "alpha path" analysis.
</aside>

### Portfolio Heatmaps
These heatmaps (represented as bar charts for clarity in Streamlit) help identify which segments of the portfolio are most vulnerable to the stress scenario.

1.  **Select Aggregation Segment**: Use the dropdown to choose a categorical feature for aggregation (e.g., `rating_band`, `sector`, `region`).
2.  **Mean $\Delta$PD Heatmap**: Displays the average change in PD for each sub-segment.
3.  **$\Delta$EL Heatmap**: Displays the total change in Expected Loss for each sub-segment.

<aside class="positive">
<b>Interpretation:</b> Segments with significantly higher positive bars in these plots are more impacted by the stress. For instance, if the "Financials" sector shows a much larger $\Delta$EL compared to "Utilities" under a "Market Stress" scenario, it indicates higher sectoral vulnerability, which is crucial for targeted risk mitigation strategies.
</aside>

### Threshold Flags Table
This table allows you to identify individual instances that exhibit significant changes in PD.

1.  **Set "Delta PD" Threshold**: Use the number input to define a threshold for the absolute change in PD.
2.  The table will then display all instances where the absolute `Delta_PD` exceeds this threshold. These are the individual loans or entities most impacted by the stress scenario.

<aside class="positive">
<b>Business Value:</b> Flagging individual instances helps risk managers zoom into specific cases that warrant further investigation. These could be potential "bad apples" or clusters of vulnerabilities that might not be obvious from aggregate metrics alone. It's a key tool for detailed model validation and portfolio oversight.
</aside>

You have now completed a full stress test cycle using the Financial Model Stress Analyzer! Experiment with different scenarios, custom shocks, and alpha values to explore various aspects of ML model robustness.
