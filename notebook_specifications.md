
# Technical Specification for Jupyter Notebook: Financial Model Stress Analyzer

## 1. Notebook Overview

This Jupyter Notebook, titled "Financial Model Stress Analyzer," is designed for quantitative analysts, model developers, risk managers, and validators. It provides a practical framework for conducting scenario-based robustness tests on Machine Learning (ML) models by applying structured, economically meaningful transformations to input features.

### Learning Goals

Upon completion of this notebook, participants will be able to:
*   Understand and apply scenario-based stress testing directly to ML models via feature-level transformations $T_s(x)$.
*   Translate abstract financial scenarios (e.g., recession, market stress) into concrete quantitative shocks ($\delta_s$) applied to input features.
*   Measure and interpret ML model sensitivity to these structured shocks using risk-friendly metrics such as Probability of Default (PD), Value-at-Risk (VaR), and Expected Loss (EL).
*   Identify model behaviors that are fragile or insufficiently conservative under plausible adverse financial environments.
*   Integrate scenario-based robustness tests into a broader model validation and governance workflow, making ML model robustness tangible and auditable.

## 2. Code Requirements

### List of Expected Libraries

The following Python libraries are expected to be used in the notebook:
*   `numpy` for numerical operations.
*   `pandas` for data manipulation and analysis.
*   `sklearn.preprocessing` for data scaling (if needed, but not explicitly required by current scope for feature generation itself, more for a complete ML pipeline context).
*   `sklearn.linear_model` (for conceptual mock model, not actual training).
*   `matplotlib.pyplot` for plotting and visualization.
*   `seaborn` for enhanced statistical data visualization.
*   `scipy.special` for sigmoid function.

### List of Algorithms or Functions to be Implemented

1.  **Synthetic Data Generation**:
    *   `generate_financial_data(n_samples: int) -> pandas.DataFrame`: Generates a synthetic tabular dataset of specified size with financial features and a target variable.
2.  **Mock ML Model**:
    *   `MockMLModel`: A class with a `predict(X: pandas.DataFrame) -> numpy.ndarray` method. This method simulates a pre-trained ML model's predictions (e.g., PD scores) based on input features without actual model training.
3.  **Stress Transformation Library**:
    *   `apply_feature_shock(df: pandas.DataFrame, shocks_dict: dict, shock_type: str = 'multiplicative') -> pandas.DataFrame`: Applies a dictionary of shocks (either multiplicative scaling or additive shifting) to specified features in a DataFrame.
4.  **Prediction Functions**:
    *   `get_predictions(model: MockMLModel, X: pandas.DataFrame) -> numpy.ndarray`: Wrapper to get predictions from the mock model.
5.  **Impact Metrics Calculation**:
    *   `calculate_prediction_change(y_stressed: numpy.ndarray, y_baseline: numpy.ndarray) -> numpy.ndarray`: Calculates the element-wise difference between stressed and baseline predictions.
    *   `calculate_mean_delta(y_stressed: numpy.ndarray, y_baseline: numpy.ndarray) -> float`: Calculates the change in the mean of predictions.
    *   `calculate_expected_loss(predictions: numpy.ndarray, exposures: numpy.ndarray) -> float`: Calculates the total Expected Loss for a portfolio given predictions (e.g., PDs) and exposures.
6.  **Portfolio Aggregation**:
    *   `aggregate_by_segment(df: pandas.DataFrame, segment_col: str, metric_col: str, agg_func: callable) -> pandas.DataFrame`: Aggregates a specified metric by a categorical segment column.
7.  **Scenario Management**:
    *   `run_single_scenario(model: MockMLModel, X_baseline: pandas.DataFrame, scenario_name: str, scenario_shocks: dict, shock_type: str) -> dict`: Orchestrates the application of a single scenario and returns baseline and stressed predictions.
    *   `run_multiple_scenarios(model: MockMLModel, X_baseline: pandas.DataFrame, scenarios_config: dict) -> dict`: Iterates through a dictionary of scenarios, applies each, and collects results.
8.  **Sensitivity Analysis (Alpha Path)**:
    *   `run_alpha_path_analysis(model: MockMLModel, X_baseline: pandas.DataFrame, base_shocks: dict, alpha_steps: list, shock_type: str) -> pandas.DataFrame`: Applies a scenario with varying intensity ($\alpha$) and collects mean predictions at each step.

### Visualization like charts, tables, plots that should be generated

1.  **Tables**:
    *   Summary table comparing baseline and stressed mean PD, mean EL, and their absolute changes for each scenario.
    *   Tables showing segments (e.g., `rating_band`, `sector`) where impact metrics exceed predefined thresholds.
2.  **Histograms/Density Plots**:
    *   Distributions of baseline predictions ($\hat{y}$), stressed predictions ($\hat{y}^{(s)}$), and the change in predictions ($\Delta \hat{y}^{(s)}$) for a selected scenario.
3.  **Line Plots**:
    *   Trajectories of portfolio-level metrics (e.g., mean PD) as the shock intensity ($\alpha$) varies from 0 to 1 for specific scenarios.
4.  **Heatmaps**:
    *   Aggregated impact metrics (e.g., $\Delta \text{mean PD}$) across different segments (e.g., `rating_band` vs. `sector`) for selected scenarios.

## 3. Notebook Sections (in detail)

### 3.1. Introduction to Financial Model Stress Analyzer

This section provides an overview of the notebook's purpose: to demonstrate scenario-based robustness testing for ML models in finance. We will explore how models react to structured changes in input features, mimicking real-world stress events.

### 3.2. Setting Up the Environment

This section ensures all necessary libraries are imported for data generation, manipulation, modeling, and visualization.

**Code Cell: Library Imports**
This code cell should import the following libraries:
*   `numpy` as `np`
*   `pandas` as `pd`
*   `matplotlib.pyplot` as `plt`
*   `seaborn` as `sns`
*   `scipy.special.expit` (for sigmoid function)

### 3.3. Generating Synthetic Financial Data

We will create a synthetic dataset that simulates a credit portfolio, allowing us to demonstrate stress testing without using sensitive real-world data. The dataset will include various financial characteristics and a target variable (e.g., Probability of Default).

**Code Cell: `generate_financial_data` Function Implementation**
This code cell should define a function named `generate_financial_data` that takes `n_samples` (integer) as input.
The function should generate a Pandas DataFrame with `n_samples` rows and the following columns:
*   **Numerical Features**:
    *   `income`: Generated using `np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples)`.
    *   `debt_to_income`: Generated using `np.random.normal(loc=0.3, scale=0.1, size=n_samples)`, clipped between 0.1 and 0.6.
    *   `utilization`: Generated using `np.random.beta(a=2, b=5, size=n_samples)`, scaled to be between 0 and 1.
    *   `house_price`: Generated using `np.random.lognormal(mean=12, sigma=0.5, size=n_samples)`.
    *   `LTV` (Loan-to-Value): Generated using `np.random.beta(a=5, b=2, size=n_samples)`, scaled to be between 0.2 and 0.9.
    *   `credit_spread`: Generated using `np.random.exponential(scale=0.01, size=n_samples) + 0.005`, clipped between 0.005 and 0.05.
    *   `volatility`: Generated using `np.random.exponential(scale=0.008, size=n_samples) + 0.01`, clipped between 0.01 and 0.05.
    *   `liquidity`: Generated using `np.random.beta(a=8, b=2, size=n_samples)`, scaled to be between 0.1 and 1.
*   **Categorical Features**:
    *   `rating_band`: Randomly sampled from `['A', 'B', 'C', 'D']` using `np.random.choice`.
    *   `sector`: Randomly sampled from `['Finance', 'Tech', 'Retail', 'Manufacturing']` using `np.random.choice`.
    *   `region`: Randomly sampled from `['North', 'South', 'East', 'West']` using `np.random.choice`.
*   **Auxiliary Column**:
    *   `exposure`: Generated using `np.random.lognormal(mean=10, sigma=0.5, size=n_samples)`.
*   **Target Variable (`PD_target`)**:
    *   Calculated as `expit(offset + w_income*log(income) + w_dti*debt_to_income + w_util*utilization + w_ltv*LTV + w_spread*credit_spread + noise)`.
    *   Define fixed weights (e.g., `w_income=-0.5`, `w_dti=2.0`, `w_util=1.5`, `w_ltv=1.0`, `w_spread=3.0`, `offset=-5.0`).
    *   `noise` should be `np.random.normal(0, 0.2, n_samples)`.
    *   Ensure `PD_target` is between 0 and 1.

**Code Cell: Data Generation and Initial Inspection**
This code cell should:
1.  Call `generate_financial_data` with `n_samples=5000` to create `df_baseline`.
2.  Display the first 5 rows of `df_baseline` using `.head()`.
3.  Display a summary of the DataFrame's info using `.info()`.
4.  Display descriptive statistics of numerical columns using `.describe()`.

### 3.4. Implementing a Mock ML Model

To simulate real-world model interaction, we will define a simple `MockMLModel` class. This model will not be trained but will provide predictions based on a predefined mathematical function of its input features, acting as a placeholder for a pre-trained ML model `f_\theta(x)`.

**Code Cell: `MockMLModel` Class Implementation**
This code cell should define a class named `MockMLModel`.
*   The `__init__` method should initialize fixed "coefficients" (weights) and an "intercept" that mimic a linear model followed by a sigmoid activation. These coefficients should be fixed numpy arrays (e.g., `np.array([-0.5, 2.0, 1.5, 1.0, 3.0])`) and `intercept` (e.g., -5.0).
*   The `predict` method should take a Pandas DataFrame `X` as input. It should select a specific subset of numerical features from `X` (e.g., `['income', 'debt_to_income', 'utilization', 'LTV', 'credit_spread']`). It should then compute a linear combination of these features using the predefined coefficients and intercept. Finally, it should apply the `scipy.special.expit` (sigmoid) function to transform the linear output into probabilities between 0 and 1.

**Code Cell: Instantiate Mock Model and Prepare Features**
This code cell should:
1.  Instantiate the `MockMLModel` as `mock_model`.
2.  Define `numerical_features` as a list of features the mock model expects (e.g., `['income', 'debt_to_income', 'utilization', 'LTV', 'credit_spread']`).
3.  Select these features from `df_baseline` to create `X_baseline`.

### 3.5. Calculating Baseline Predictions

Before applying any stress, we establish a baseline by obtaining predictions from our mock ML model on the original, unstressed dataset. This serves as the reference point for measuring scenario impacts. The baseline prediction is denoted as $\hat{y} = f_\theta(x)$.

**Code Cell: Calculate Baseline Predictions**
This code cell should:
1.  Call `mock_model.predict(X_baseline)` to obtain `y_hat_baseline`.
2.  Add `y_hat_baseline` as a new column named `PD_baseline` to `df_baseline`.

**Markdown Cell: Interpretation of Baseline Predictions**
Explain that `PD_baseline` represents the model's initial prediction (e.g., Probability of Default) for each entity under normal conditions. This is the starting point from which we will measure changes under stress. Display the mean and standard deviation of `PD_baseline`.

### 3.6. Defining the Stress Transformation Library

Stress scenarios are translated into quantitative feature transformations $T_s(x)$. These transformations modify the input features according to predefined shock parameters. We will implement a function to apply multiplicative (scaling) or additive (shifting) shocks. The general form of a multiplicative shock is $x^{(s)} = x \odot (1 + \delta_s)$, where $\odot$ denotes element-wise multiplication and $\delta_s$ is a vector of scenario-specific shocks.

**Code Cell: `apply_feature_shock` Function Implementation**
This code cell should define a function named `apply_feature_shock`.
*   It takes `df` (Pandas DataFrame), `shocks_dict` (dictionary where keys are feature names and values are shock factors/values), and `shock_type` (string, either 'multiplicative' or 'additive', default 'multiplicative') as arguments.
*   The function should create a copy of the input DataFrame `df_stressed`.
*   It should iterate through `shocks_dict`:
    *   If `shock_type` is 'multiplicative', apply `df_stressed[feature] *= shock_factor` for each feature-factor pair.
    *   If `shock_type` is 'additive', apply `df_stressed[feature] += shock_value` for each feature-value pair.
*   Return the `df_stressed` DataFrame.

### 3.7. Parameterizing Financial Stress Scenarios

This section focuses on defining concrete financial stress scenarios and mapping them to quantitative input shocks. Typical scenario types include 'macro downturn', 'market stress', or 'idiosyncratic borrower shocks', each affecting a specific set of features.

**Code Cell: Defining Scenario Parameters**
This code cell should define a dictionary named `scenario_shocks_config`.
Each key in `scenario_shocks_config` represents a scenario name (e.g., 'Macro Downturn', 'Market Stress'). The value for each scenario should be another dictionary specifying the feature-level shocks and their `shock_type`.

Example Scenarios:
*   **'Macro Downturn'**:
    *   `shock_type`: 'multiplicative'
    *   `shocks`: `{'income': 0.85, 'debt_to_income': 1.15, 'utilization': 1.2}` (e.g., 15% income drop, 15% DTI increase, 20% utilization increase).
*   **'Market Stress'**:
    *   `shock_type`: 'multiplicative'
    *   `shocks`: `{'credit_spread': 1.5, 'volatility': 1.3}` (e.g., 50% credit spread increase, 30% volatility increase).
*   **'Housing Market Downturn'**:
    *   `shock_type`: 'multiplicative'
    *   `shocks`: `{'house_price': 0.8, 'LTV': 1.1}` (e.g., 20% house price drop, 10% LTV increase).

**Markdown Cell: Explanation of Defined Scenarios**
Explain how the chosen financial scenarios translate into the quantitative shocks defined in `scenario_shocks_config`. Describe the expected impact of each scenario on the underlying financial conditions.

### 3.8. Applying a Single Stress Scenario and Measuring Model Response

We will now apply one of the defined stress scenarios to the baseline features to create a stressed dataset $x^{(s)}$. The mock ML model will then generate predictions on this stressed data, yielding stressed predictions $\hat{y}^{(s)} = f_\theta(x^{(s)})$.

**Code Cell: Apply 'Macro Downturn' Scenario**
This code cell should:
1.  Select the `X_baseline` features from `df_baseline`.
2.  Call `apply_feature_shock` with `X_baseline`, the 'Macro Downturn' shocks from `scenario_shocks_config`, and the corresponding `shock_type` to create `X_stressed_macro`.
3.  Call `mock_model.predict(X_stressed_macro)` to get `y_hat_stressed_macro`.
4.  Add `y_hat_stressed_macro` as a new column named `PD_stressed_macro` to a copy of `df_baseline` (which we will call `df_stressed_macro_results`).

**Markdown Cell: Interpretation of Stressed Predictions**
Describe how the model's predictions have changed under the 'Macro Downturn' scenario. Mention that we will quantify this change in the next section.

### 3.9. Calculating Individual Prediction Impact Metrics

The core of stress testing is to quantify the difference between stressed and baseline predictions. We calculate the individual prediction change as $\Delta \hat{y}^{(s)} = \hat{y}^{(s)} - \hat{y}$.

**Code Cell: Calculate Individual Prediction Changes for 'Macro Downturn'**
This code cell should:
1.  Calculate `delta_y_macro = df_stressed_macro_results['PD_stressed_macro'] - df_stressed_macro_results['PD_baseline']`.
2.  Add `delta_y_macro` as a new column named `Delta_PD_macro` to `df_stressed_macro_results`.
3.  Display descriptive statistics of `df_stressed_macro_results['Delta_PD_macro']` using `.describe()`.

**Markdown Cell: Interpretation of Individual Prediction Change**
Explain what `Delta_PD_macro` signifies (e.g., the individual change in Probability of Default for each customer). Discuss the range and mean of these changes, indicating the general direction and magnitude of the model's response.

### 3.10. Portfolio-Level Aggregation: Mean PD Change

Beyond individual changes, financial professionals require portfolio-level metrics. One such metric is the change in mean PD across the entire portfolio under a given stress scenario.

**Code Cell: Calculate Change in Mean PD for 'Macro Downturn'**
This code cell should:
1.  Calculate `mean_pd_baseline = df_stressed_macro_results['PD_baseline'].mean()`.
2.  Calculate `mean_pd_stressed_macro = df_stressed_macro_results['PD_stressed_macro'].mean()`.
3.  Calculate `delta_mean_pd_macro = mean_pd_stressed_macro - mean_pd_baseline`.
4.  Print `mean_pd_baseline`, `mean_pd_stressed_macro`, and `delta_mean_pd_macro` with descriptive labels.

**Markdown Cell: Interpretation of Portfolio-Level Mean PD Change**
Explain how `delta_mean_pd_macro` indicates the overall shift in the portfolio's expected default rate due to the 'Macro Downturn' scenario. Discuss its significance for risk management.

### 3.11. Portfolio-Level Aggregation: Expected Loss (EL) Change

Expected Loss (EL) is a critical risk metric for credit portfolios, typically calculated as $EL = \sum (\text{PD} \times \text{EAD} \times \text{LGD})$, where PD is Probability of Default, EAD is Exposure at Default, and LGD is Loss Given Default. For simplicity, we assume EAD and LGD are known (or effectively `exposure` in our synthetic data represents `PD * EAD * LGD` for each entity if PD is the output). Let's define EL as the sum of `PD * exposure`. We will calculate the change in EL, $\Delta \text{EL}^{(s)} = \text{EL}^{(s)} - \text{EL}$.

**Code Cell: `calculate_expected_loss` Function Implementation**
This code cell should define a function named `calculate_expected_loss` that takes `predictions` (numpy array of PDs) and `exposures` (numpy array of exposures) as input.
It should return the sum of the element-wise product of `predictions` and `exposures`.

**Code Cell: Calculate Expected Loss Changes for 'Macro Downturn'**
This code cell should:
1.  Call `calculate_expected_loss` with `df_stressed_macro_results['PD_baseline']` and `df_stressed_macro_results['exposure']` to get `EL_baseline`.
2.  Call `calculate_expected_loss` with `df_stressed_macro_results['PD_stressed_macro']` and `df_stressed_macro_results['exposure']` to get `EL_stressed_macro`.
3.  Calculate `delta_EL_macro = EL_stressed_macro - EL_baseline`.
4.  Print `EL_baseline`, `EL_stressed_macro`, and `delta_EL_macro` with descriptive labels.

**Markdown Cell: Interpretation of Expected Loss Changes**
Discuss the implications of `delta_EL_macro` for the financial institution's capital reserves and risk appetite. Explain how EL change provides a more direct financial impact metric than just PD change.

### 3.12. Running Multiple Stress Scenarios

To conduct a comprehensive robustness test, we need to apply a range of scenarios and consolidate their impacts. This section automates the process of running multiple predefined stress scenarios.

**Code Cell: `run_multiple_scenarios` Function Implementation**
This code cell should define a function named `run_multiple_scenarios`.
*   It takes `model` (`MockMLModel` instance), `X_baseline` (Pandas DataFrame of baseline features), and `scenarios_config` (the dictionary defined in section 3.7) as arguments.
*   It should initialize an empty dictionary `results_storage`.
*   For each `scenario_name` and its `config` in `scenarios_config`:
    *   Create a copy of `X_baseline`.
    *   Apply the shocks using `apply_feature_shock` to get `X_stressed`.
    *   Obtain stressed predictions `y_hat_stressed` using `model.predict(X_stressed)`.
    *   Store `y_hat_stressed` in `results_storage` under `scenario_name`.
*   Return `results_storage`.

**Code Cell: Execute Multiple Scenarios**
This code cell should:
1.  Call `run_multiple_scenarios` with `mock_model`, `X_baseline`, and `scenario_shocks_config`. Store the result in `all_scenario_predictions`.
2.  Print the keys of `all_scenario_predictions` to show which scenarios were run.

### 3.13. Visualization: Scenario vs. Baseline Metrics Table

Generating validation-ready outputs is crucial. This section presents a table comparing key aggregated metrics for each scenario against the baseline, providing a quick overview of model sensitivity.

**Code Cell: Generate and Display Scenario Metrics Table**
This code cell should:
11. Initialize an empty list `metrics_records`.
12. For `scenario_name`, `y_hat_stressed` in `all_scenario_predictions.items()`:
    *   Calculate `mean_pd_stressed = y_hat_stressed.mean()`.
    *   Calculate `delta_mean_pd = mean_pd_stressed - df_baseline['PD_baseline'].mean()`.
    *   Calculate `EL_stressed = calculate_expected_loss(y_hat_stressed, df_baseline['exposure'])`.
    *   Calculate `delta_EL = EL_stressed - EL_baseline` (using `EL_baseline` from 3.11).
    *   Append a dictionary `{'Scenario': scenario_name, 'Mean PD Stressed': mean_pd_stressed, 'Delta Mean PD': delta_mean_pd, 'EL Stressed': EL_stressed, 'Delta EL': delta_EL}` to `metrics_records`.
13. Create a Pandas DataFrame `scenario_metrics_df` from `metrics_records`.
14. Display `scenario_metrics_df`.

**Markdown Cell: Interpretation of Scenario Metrics Table**
Discuss the insights gained from the table, highlighting scenarios that result in the largest changes in mean PD and EL. This table serves as a summary for quick comparison of scenario impacts.

### 3.14. Visualization: Distribution Shifts (Histograms/Density Plots)

Visualizing the distributions of baseline, stressed, and change in predictions helps understand how the model's output shifts. Histograms or density plots reveal non-linear responses and potential concentration of risk.

**Code Cell: Plot Distribution Shifts for 'Macro Downturn'**
This code cell should:
1.  Create a figure with 3 subplots using `plt.figure` and `plt.subplot`.
2.  **Subplot 1**: Plot a histogram or density plot of `df_baseline['PD_baseline']` using `sns.histplot` or `sns.kdeplot`, labeled "Baseline PD Distribution".
3.  **Subplot 2**: Plot a histogram or density plot of `all_scenario_predictions['Macro Downturn']` using `sns.histplot` or `sns.kdeplot`, labeled "Stressed PD Distribution (Macro Downturn)".
4.  **Subplot 3**: Calculate `delta_pd_macro = all_scenario_predictions['Macro Downturn'] - df_baseline['PD_baseline']`. Plot a histogram or density plot of `delta_pd_macro` using `sns.histplot` or `sns.kdeplot`, labeled "Change in PD Distribution (Macro Downturn)".
5.  Set titles and labels for each subplot, and adjust layout using `plt.tight_layout()`.
6.  Display the plot using `plt.show()`.

**Markdown Cell: Interpretation of Distribution Shift Plots**
Analyze the plots, noting any shifts in the mean, changes in variance, or emergence of fat tails in the stressed distributions compared to the baseline. For $\Delta \hat{y}^{(s)}$, observe if changes are concentrated around zero or show significant positive/negative shifts.

### 3.15. Sensitivity Analysis: Alpha Path

The "alpha path" analysis explores the model's response to gradually increasing shock intensity. By varying a shock intensity parameter $\alpha \in [0,1]$, where $x^{(s,\alpha)} = x \odot (1 + \alpha \cdot \delta_s)$, we can observe potential non-linear model responses, thresholds, or instabilities.

**Code Cell: `run_alpha_path_analysis` Function Implementation**
This code cell should define a function named `run_alpha_path_analysis`.
*   It takes `model` (`MockMLModel` instance), `X_baseline` (Pandas DataFrame), `base_shocks` (dictionary of shocks for full intensity), `alpha_steps` (list of float values from 0 to 1), and `shock_type` (string) as arguments.
*   It should initialize an empty list `alpha_results`.
*   For each `alpha` in `alpha_steps`:
    *   Create `current_shocks_dict` by multiplying each shock factor/value in `base_shocks` by `alpha`.
    *   Apply these `current_shocks_dict` to `X_baseline` using `apply_feature_shock` to get `X_stressed_alpha`.
    *   Get predictions `y_hat_alpha` using `model.predict(X_stressed_alpha)`.
    *   Calculate `mean_pd_alpha = y_hat_alpha.mean()`.
    *   Append a dictionary `{'Alpha': alpha, 'Mean PD': mean_pd_alpha}` to `alpha_results`.
*   Return a Pandas DataFrame created from `alpha_results`.

**Code Cell: Execute Alpha Path Analysis for 'Macro Downturn'**
This code cell should:
1.  Define `alpha_steps = np.linspace(0, 1, 11)`.
2.  Extract the 'Macro Downturn' shocks from `scenario_shocks_config`.
3.  Call `run_alpha_path_analysis` with `mock_model`, `X_baseline`, the 'Macro Downturn' base shocks, `alpha_steps`, and the corresponding `shock_type`. Store the result in `alpha_path_df_macro`.
4.  Display the first few rows of `alpha_path_df_macro`.

### 3.16. Visualization: Sensitivity Trajectories (Line Plots)

Line plots of portfolio metrics along the alpha path clearly illustrate how the model's output changes as stress intensifies. This is particularly useful for identifying non-linear responses and stress thresholds.

**Code Cell: Plot Alpha Path Trajectory for 'Macro Downturn'**
This code cell should:
1.  Create a line plot using `sns.lineplot` with `alpha_path_df_macro`.
2.  Map 'Alpha' to the x-axis and 'Mean PD' to the y-axis.
3.  Add a title like "Mean PD Trajectory under Macro Downturn Stress (Alpha Path)".
4.  Add labels for x and y axes.
5.  Display the plot using `plt.show()`.

**Markdown Cell: Interpretation of Sensitivity Trajectory Plot**
Analyze the shape of the trajectory. Does it show a linear increase, or does the response accelerate or decelerate at certain $\alpha$ values? Discuss what this non-linearity might imply about model behavior under increasing stress.

### 3.17. Visualization: Portfolio Heatmaps by Segment

To understand differential impacts across different segments of the portfolio, heatmaps are effective. This allows risk managers to pinpoint specific segments that are most vulnerable to certain stress scenarios.

**Code Cell: Generate and Display Heatmap for Segmented Impact**
This code cell should:
1.  Create a copy of `df_baseline` named `df_results_with_delta`.
2.  For each `scenario_name`, `y_hat_stressed` in `all_scenario_predictions.items()`:
    *   Calculate `delta_pd = y_hat_stressed - df_results_with_delta['PD_baseline']`.
    *   Add this `delta_pd` as a new column to `df_results_with_delta` named `Delta_PD_{scenario_name.replace(' ', '_')}`.
3.  Aggregate `df_results_with_delta` by `rating_band` (or another categorical feature like `sector`) and calculate the mean of the `Delta_PD` for each scenario. Store this in `segmented_delta_pd`.
4.  Pivot `segmented_delta_pd` to prepare for a heatmap, with `rating_band` as index and scenarios as columns.
5.  Create a heatmap using `sns.heatmap` on the pivoted DataFrame, with annotations (`annot=True`), a suitable colormap (`cmap='coolwarm'`), and a title.
6.  Display the plot using `plt.show()`.

**Markdown Cell: Interpretation of Portfolio Heatmap**
Discuss which segments are most (or least) affected by which scenarios, as revealed by the heatmap. Highlight any segments showing particularly large changes, indicating potential concentrated risk.

### 3.18. Visualization: Threshold Flags

Visual indicators or tables highlighting instances where changes in predictions or metrics exceed predefined thresholds are vital for reporting and identifying material impacts.

**Code Cell: Identify and Display Threshold Exceedances**
This code cell should:
1.  Define a `threshold_delta_pd = 0.05` (e.g., a 5 percentage point increase in PD).
2.  Select a specific scenario (e.g., 'Macro Downturn').
3.  Calculate `delta_pd_macro = all_scenario_predictions['Macro Downturn'] - df_baseline['PD_baseline']`.
4.  Create a boolean mask where `delta_pd_macro > threshold_delta_pd`.
5.  Filter `df_baseline` using this mask to get `exceedances_df`.
6.  Display the count of instances exceeding the threshold.
7.  Display the first 10 rows of `exceedances_df` (including relevant features, baseline PD, stressed PD, and delta PD).

**Markdown Cell: Interpretation of Threshold Flags**
Explain how exceeding predefined thresholds indicates significant risk shifts. Discuss how these flags help in targeting further investigation or reporting, and how they contribute to validation-ready outputs.

### 3.19. Summary and Conclusion

This final section summarizes the key takeaways from the stress analysis. It reiterates how the notebook operationalizes model robustness testing for ML models by making scenario impacts concrete and measurable.

