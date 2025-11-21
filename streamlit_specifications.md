
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will enable financial professionals to conduct scenario-based robustness tests on Machine Learning (ML) models. It will allow users to define stress scenarios, apply them as quantitative input shocks to model features, and observe their impact on model predictions and aggregated portfolio metrics. The target audience includes quantitative analysts, model developers, risk managers, and validators.

### Learning Goals

Upon completion of this application, participants will be able to:
*   Understand and apply scenario-based stress testing directly to ML models via feature-level transformations $T_s(x)$.
*   Translate abstract financial scenarios (e.g., recession, market stress) into concrete quantitative shocks ($\delta_s$) applied to input features.
*   Measure and interpret ML model sensitivity to these structured shocks using risk-friendly metrics such as Probability of Default (PD), Value-at-Risk (VaR), and Expected Loss (EL).
*   Identify model behaviors that are fragile or insufficiently conservative under plausible adverse financial environments.
*   Integrate scenario-based robustness tests into a broader model validation and governance workflow, making ML model robustness tangible and auditable.

## 2. User Interface Requirements

The application will feature a clear, step-by-step workflow for stress testing ML models.

### Layout and Navigation Structure
*   **Main Content Area**: Will display data overview, scenario results, and visualizations.
*   **Sidebar**: Will host primary controls for data generation, scenario selection/definition, and stress test execution.
*   **Sequential Flow**: The application will guide users through the process: Data Generation $\rightarrow$ Mock Model Setup $\rightarrow$ Baseline Calculation $\rightarrow$ Scenario Definition $\rightarrow$ Stress Application $\rightarrow$ Impact Analysis $\rightarrow$ Visualization.
*   **Expanders**: Used to organize detailed input options or complex result sections (e.g., "Define Custom Scenario").

### Input Widgets and Controls

1.  **Data Generation Section (Sidebar)**
    *   **Number of Samples**: `st.slider` or `st.number_input` for `n_samples` (e.g., 1000 to 10000).
    *   **"Generate Synthetic Data" Button**: `st.button` to trigger `generate_financial_data`.
    *   **Display Data Info**: `st.dataframe` for `df.head()`, `st.write` for `df.info()` summary and `df.describe()`.

2.  **Scenario Definition & Application Section (Sidebar)**
    *   **Scenario Selection**: `st.selectbox` or `st.radio` for choosing a pre-defined scenario (e.g., 'Macro Downturn', 'Market Stress', 'Housing Market Downturn').
    *   **Custom Scenario Definition (within `st.expander`)**:
        *   **Shock Type**: `st.radio` ('Multiplicative', 'Additive').
        *   **Feature Shocks**: Multiple `st.text_input` (for feature name) and `st.number_input` (for shock value) fields, allowing users to add/remove feature-shock pairs.
    *   **Shock Intensity ($\alpha$) Slider**: `st.slider` for $\alpha$ ranging from 0 to 1 (e.g., 0.0, 0.1, ..., 1.0) for sensitivity analysis.
    *   **"Run Stress Test" Button**: `st.button` to apply the selected/defined scenario and calculate stressed predictions.

3.  **Visualization Controls (Sidebar or Main Content)**
    *   **Aggregation Segment Selector**: `st.selectbox` for categorical features like `rating_band`, `sector`, `region` for heatmap aggregation.

### Visualization Components (Main Content Area)

1.  **Baseline & Scenario Metrics Table**: `st.dataframe` or `st.table` comparing mean `PD_baseline`, mean `PD_stressed`, mean `Delta PD`, baseline `EL`, stressed `EL`, `Delta EL` for the selected scenario.
2.  **Distribution Shifts Plots**:
    *   **Histograms/Density Plots**: Using `matplotlib.pyplot` or `seaborn` (displayed with `st.pyplot`) for `PD_baseline`, `PD_stressed`, and `ΔPD`.
3.  **Sensitivity Trajectories Plot**:
    *   **Line Plot**: Using `matplotlib.pyplot` (displayed with `st.pyplot`) showing mean `PD_stressed` as `α` varies from 0 to 1 for the selected scenario.
4.  **Portfolio Heatmaps**:
    *   **Heatmap**: Using `seaborn.heatmap` (displayed with `st.pyplot`) showing aggregated impact metrics (e.g., mean `ΔPD`, `ΔEL`) across different segments (e.g., `rating_band`, `sector`, `region`).
5.  **Threshold Flags Table**: `st.dataframe` highlighting instances where changes in predictions or metrics exceed predefined thresholds (e.g., individual `ΔPD` > 0.1).

### Interactive Elements and Feedback Mechanisms
*   **Loading Spinners**: `st.spinner` or `st.progress` will be used to indicate ongoing computations (e.g., "Generating Data...", "Running Stress Test...").
*   **Informational Messages**: `st.info` or `st.success` messages to confirm actions or provide guidance.
*   **Dynamic Updates**: Visualizations and metrics will update dynamically upon selection of a new scenario, change in `α`, or regeneration of data.
*   **Error Handling**: `st.error` messages for invalid inputs or computational issues.

## 3. Additional Requirements

*   **Annotation and Tooltip Specifications**:
    *   All input widgets will have clear `help` text or `st.info` blocks explaining their purpose and expected input.
    *   All plots and tables will be accompanied by `st.markdown` descriptions, explaining what the visualization represents and how to interpret it, including all relevant LaTeX formulae.
    *   Tooltips for heatmap cells displaying exact values.
*   **State Preservation**:
    *   `st.session_state` will be used to store critical variables such as `df_baseline`, `mock_model`, `PD_baseline`, selected scenario, custom shock parameters, and the `alpha` value. This ensures that user selections and generated data persist across re-runs or widget interactions without losing state.
    *   The `np.random.seed(42)` will be applied consistently to ensure reproducible data generation within a session.

## 4. Notebook Content and Code Requirements

This section details how the Jupyter Notebook content, including markdown and code stubs, will be integrated into the Streamlit application.

### Application Header and Introduction
*   **Content**:
    ```markdown
    # Financial Model Stress Analyzer

    This Jupyter Notebook, titled "Financial Model Stress Analyzer," is designed for quantitative analysts, model developers, risk managers, and validators. It provides a practical framework for conducting scenario-based robustness tests on Machine Learning (ML) models by applying structured, economically meaningful transformations to input features.

    ### Learning Goals
    Upon completion of this notebook, participants will be able to:
    *   Understand and apply scenario-based stress testing directly to ML models via feature-level transformations $T_s(x)$.
    *   Translate abstract financial scenarios (e.g., recession, market stress) into concrete quantitative shocks ($\delta_s$) applied to input features.
    *   Measure and interpret ML model sensitivity to these structured shocks using risk-friendly metrics such as Probability of Default (PD), Value-at-Risk (VaR), and Expected Loss (EL).
    *   Identify model behaviors that are fragile or insufficiently conservative under plausible adverse financial environments.
    *   Integrate scenario-based robustness tests into a broader model validation and governance workflow, making ML model robustness tangible and auditable.
    ```
*   **Streamlit Integration**: Displayed using `st.title` and `st.markdown` at the top of the application.

### 2. Setting Up the Environment
*   **Content**:
    ```markdown
    ## 2. Setting Up the Environment

    This section ensures all necessary libraries are imported for data generation, manipulation, modeling, and visualization.
    ```
*   **Streamlit Integration**: `st.header` and `st.markdown`.
    *   **Code**: All `import` statements and `pd.set_option`, `np.random.seed` (or alternative for reproducibility) will be placed at the top of the `app.py` file.

### 3. Data/Inputs Overview & 5.1. Generating Synthetic Financial Data
*   **Content**:
    ```markdown
    ## 3. Data/Inputs Overview

    To effectively demonstrate stress testing without relying on sensitive real-world financial data, this notebook utilizes a synthetic dataset. This synthetic data simulates a credit portfolio, encompassing various financial characteristics and a target variable representing the Probability of Default (PD). The data generation process is designed to mimic realistic financial feature distributions and interdependencies, making it suitable for illustrating the concepts of scenario-based stress testing.

    **Assumptions:**
    *   The synthetic dataset is representative of a typical credit portfolio.
    *   Features like `income`, `debt_to_income`, `utilization`, `LTV`, `credit_spread`, `house_price`, `volatility`, `liquidity`, `exposure`, `rating_band`, `sector`, and `region` are generated with distributions that are plausible in a financial context.
    *   The `PD_target` is a simulated outcome based on a combination of these features, reflecting a simplified credit risk model.

    This approach allows us to focus on the methodology of stress testing and model response analysis, rather than the complexities of real data acquisition and anonymization. The generated data supports the business goal of creating a robust framework for validating ML models under adverse conditions.

    ## 5. Sectioned Implementation

    ### 5.1. Generating Synthetic Financial Data

    This section focuses on creating a synthetic dataset that mimics a credit portfolio. This is crucial for demonstrating stress testing concepts without the complexities and sensitivities associated with real-world financial data. By generating controlled data, we can ensure reproducibility and isolate the effects of stress transformations on model predictions, thereby providing a clear and interpretable framework for model validation and risk assessment.
    ```
*   **Streamlit Integration**: `st.header`, `st.subheader`, `st.markdown` for descriptive text.
    *   **Code for `generate_financial_data`**:
        ```python
        # @st.cache_data for performance, allows `n_samples` change
        def generate_financial_data(n_samples):
            # ... (full function code from notebook) ...
            return df
        ```
        This function will be called in response to user input for `n_samples` via `st.slider` in the sidebar.
    *   **Data Display**:
        ```python
        # df_baseline = generate_financial_data(n_samples_input)
        # st.subheader("Generated Data Overview")
        # st.dataframe(df_baseline.head())
        # st.write(df_baseline.info()) # Render as text
        # st.dataframe(df_baseline.describe())
        ```

### Interpretation of Data Generation and Initial Inspection
*   **Content**:
    ```markdown
    ### Interpretation of Data Generation and Initial Inspection

    The output above provides a first look at our synthetic financial dataset.
    *   **`df_baseline.head()`**: Displays the first five rows, giving us a concrete example of the generated features. We can see numerical features like `income`, `debt_to_income`, `utilization`, `house_price`, `LTV`, `credit_spread`, `volatility`, `liquidity`, and `exposure`, alongside categorical features such as `rating_band`, `sector`, and `region`, and our target variable `PD_target`. This snippet confirms the structure and content of our simulated credit portfolio.
    *   **`df_baseline.info()`**: This provides a summary of the DataFrame, including the number of entries, column names, non-null counts, and data types. We can observe that all columns have 5000 non-null entries, indicating a complete dataset without missing values. The data types (`float64` for numerical features and `object` for categorical features) are as expected for financial data analysis.
    *   **`df_baseline.describe()`**: This function gives us descriptive statistics for all numerical columns. Key observations include:
        *   **`income` and `house_price`**: Show large positive means and standard deviations, consistent with log-normal distributions for wealth-related features.
        *   **`debt_to_income`, `utilization`, `LTV`, `credit_spread`, `volatility`, `liquidity`**: These features exhibit means and ranges that are typical for financial ratios and market indicators, all within their defined clipped boundaries. For instance, `debt_to_income`'s mean around 0.35 suggests a reasonable average leverage, while `credit_spread` and `volatility` are in expected low percentage ranges.
        *   **`exposure`**: Shows a wide range, reflecting varied loan or credit exposures within a portfolio.
        *   **`PD_target`**: The mean PD around 0.05-0.10 (5-10%) indicates a reasonable base probability of default for a general portfolio, with a standard deviation that suggests variability in credit risk across customers.

    Overall, these initial inspections confirm that our synthetic dataset is well-structured and its statistical properties are consistent with what would be expected in a simulated financial credit portfolio, making it suitable for subsequent stress testing and model analysis.
    ```
*   **Streamlit Integration**: `st.subheader` and `st.markdown` to provide context after data generation.

### 5.2. Implementing a Mock ML Model
*   **Content**:
    ```markdown
    ### 5.2. Implementing a Mock ML Model

    To simulate real-world model interaction and provide a controlled environment for stress testing, we define a simple `MockMLModel` class. This model is not trained in the traditional sense; instead, it provides predictions based on a predefined mathematical function of its input features. It acts as a placeholder for a pre-trained Machine Learning model $f_\theta(x)$, allowing us to focus on the impact of feature-level shocks on model output without the overhead of actual model training.

    **Business Value:**
    *   **Isolation of Stress Impact**: By using a mock model with fixed parameters, we can isolate and precisely measure the impact of feature transformations without confounding effects from model retraining or dynamic learning.
    *   **Reproducibility**: The fixed nature of the mock model ensures that stress test results are perfectly reproducible.
    *   **Interpretability**: The simple, transparent logic of the mock model makes it easier to understand why certain feature shocks lead to specific changes in predictions.
    *   **Flexibility**: It allows for rapid prototyping and testing of various stress scenarios and impact measurement techniques before applying them to more complex, production-grade models.
    ```
*   **Streamlit Integration**: `st.subheader`, `st.markdown`.
    *   **Code for `MockMLModel`**:
        ```python
        # @st.cache_resource to instantiate once
        class MockMLModel:
            def __init__(self):
                # ... (full class code) ...
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                # ... (full method code) ...
        ```
        The model instance will be stored in `st.session_state` or `st.cache_resource`.
    *   **Instantiation & Feature Prep**:
        ```python
        # mock_model = MockMLModel() # Using cached instance
        # numerical_features = mock_model.expected_features
        # X_baseline = df_baseline[numerical_features]
        # st.write(f"MockMLModel instantiated. Features used: {numerical_features}")
        # st.dataframe(X_baseline.head())
        ```

### Interpretation of Mock ML Model Instantiation and Feature Preparation
*   **Content**:
    ```markdown
    ### Interpretation of Mock ML Model Instantiation and Feature Preparation

    This step initializes our `MockMLModel` and prepares the input features for making predictions.
    *   **`mock_model = MockMLModel()`**: We have successfully instantiated our mock machine learning model. This object now holds the predefined coefficients and intercept that simulate the behavior of a trained model. It's ready to generate predictions based on input features.
    *   **`numerical_features`**: This list explicitly defines the subset of features from our `df_baseline` that the `MockMLModel` expects and uses for its predictions. These are typically the most influential features in a real financial model, like `income`, `debt_to_income`, `utilization`, `LTV`, and `credit_spread`. Business-wise, this highlights the model's focus and potential areas for stress testing.
    *   **`X_baseline`**: This DataFrame represents the input features for our baseline predictions, extracted directly from the `df_baseline`. By showing its head, we confirm that the data is correctly structured and contains the necessary numerical features for the `mock_model` to operate. This `X_baseline` will serve as the unstressed input against which all future stressed scenarios will be compared.

    This setup is crucial because it establishes the "normal" operating conditions for our model before any stress is applied, providing the necessary foundation for comparing baseline and stressed predictions.
    ```
*   **Streamlit Integration**: `st.subheader` and `st.markdown`.

### 5.3. Calculating Baseline Predictions
*   **Content**:
    ```markdown
    ### 5.3. Calculating Baseline Predictions

    Before applying any stress, it is essential to establish a baseline by obtaining predictions from our mock ML model on the original, unstressed dataset. This serves as the reference point for measuring scenario impacts. The baseline prediction is denoted as $\hat{y} = f_\theta(x)$.

    **Business Value:**
    *   **Reference Point**: Baseline predictions provide a crucial benchmark. Without them, there would be no way to quantify the impact of stress scenarios, making it impossible to assess model sensitivity.
    *   **Performance Under Normal Conditions**: It allows us to understand the model's expected behavior and output distributions under typical, non-stressed market conditions. This is the "business-as-usual" view.
    *   **Foundation for Impact Metrics**: All subsequent impact metrics, such as the change in Probability of Default (PD) or Expected Loss (EL), are calculated relative to this baseline, ensuring that reported impacts are meaningful and contextualized.
    ```
*   **Streamlit Integration**: `st.subheader`, `st.markdown`.
    *   **Code**:
        ```python
        # y_hat_baseline = mock_model.predict(X_baseline)
        # df_baseline['PD_baseline'] = y_hat_baseline
        # st.session_state['df_baseline'] = df_baseline # Update session state
        # st.write("Baseline predictions (PD_baseline) calculated and added to df_baseline.")
        # st.dataframe(df_baseline.head())
        ```

### Interpretation of Baseline Predictions (and subsequent markdown)
*   **Content**:
    ```markdown
    ### Interpretation of Baseline Predictions

    The `PD_baseline` column, which has just been added to our `df_baseline` DataFrame, represents the mock ML model's initial prediction for the Probability of Default (PD) for each entity (e.g., customer, loan) under normal, unstressed financial conditions. This is the **starting point** from which all changes due to stress scenarios will be measured.

    From a business perspective, `PD_baseline` reflects the model's assessment of default risk in a "business-as-usual" environment. It is crucial for:
    *   **Establishing a reference**: Any increase or decrease in PD observed under stress will be relative to this baseline, providing a clear magnitude of impact.
    *   **Understanding current risk profile**: The distribution of `PD_baseline` gives insights into the inherent risk profile of the portfolio before any adverse events.

    Let's examine the summary statistics for `PD_baseline`:
    ```
*   **Streamlit Integration**: `st.subheader`, `st.markdown`.
    *   **Code for statistics**:
        ```python
        # st.write(f"Mean of PD_baseline: {df_baseline['PD_baseline'].mean():.4f}")
        # st.write(f"Standard Deviation of PD_baseline: {df_baseline['PD_baseline'].std():.4f}")
        ```
    *   **Following markdown**:
        ```markdown
        The mean of `PD_baseline` indicates the average expected default rate across the entire portfolio under normal conditions. The standard deviation, on the other hand, tells us about the dispersion or variability of individual PDs around this mean. A higher standard deviation would imply a more heterogeneous portfolio in terms of default risk.

        These values are fundamental for financial risk managers as they set the expectation against which stressed outcomes will be compared. For example, if the mean PD significantly increases under a stress scenario, it signals a material impact on the portfolio's credit quality.
        ```
        This will be displayed using `st.markdown`.

### 5.4. Defining the Stress Transformation Library
*   **Content**:
    ```markdown
    ### 5.4. Defining the Stress Transformation Library

    Stress scenarios are translated into quantitative feature transformations $T_s(x)$. These transformations modify the input features according to predefined shock parameters. We will implement a function to apply multiplicative (scaling) or additive (shifting) shocks.

    The general form of a multiplicative shock is:

    $$x^{(s)} = x \odot (1 + \delta_s)$$

    Where $\odot$ denotes element-wise multiplication and $\delta_s$ is a vector of scenario-specific shocks. This means that each feature $x_i$ in the input $x$ is transformed into $x_i^{(s)} = x_i \times (1 + \delta_{s,i})$, where $\delta_{s,i}$ is the shock factor for feature $i$ in scenario $s$.

    For an additive shock, the form is:

    $$x^{(s)} = x + \delta_s$$

    Where $\delta_s$ is a vector of scenario-specific additive shifts. This means that each feature $x_i$ is transformed into $x_i^{(s)} = x_i + \delta_{s,i}$.

    **Business Value:**
    *   **Quantifiable Impact**: This library provides the mechanism to translate qualitative financial stress narratives (e.g., "recession") into concrete, quantifiable changes in the data that feed into the ML model.
    *   **Scenario Flexibility**: Supports both multiplicative (percentage-based) and additive (absolute-value-based) shocks, allowing for a wide range of realistic scenario constructions.
    *   **Standardization**: Ensures a consistent method for applying shocks across different features and scenarios, which is crucial for comparable and auditable stress test results.
    ```
*   **Streamlit Integration**: `st.subheader`, `st.markdown` (using `r"$$...$$"` for display equations).
    *   **Code for `apply_feature_shock`**:
        ```python
        def apply_feature_shock(df, shocks_dict, shock_type):
            # ... (full function code) ...
            return df_shocked
        ```
        This function will be central to applying shocks based on user-selected scenarios and parameters.
    *   **Pre-defined `scenario_shocks_config`**:
        ```python
        scenario_shocks_config = {
            'Macro Downturn': {
                'shock_type': 'multiplicative',
                'shocks': {'income': 0.85, 'debt_to_income': 1.15, 'utilization': 1.2}
            },
            'Market Stress': {
                'shock_type': 'multiplicative',
                'shocks': {'credit_spread': 1.5, 'volatility': 1.3}
            },
            'Housing Market Downturn': {
                'shock_type': 'multiplicative',
                'shocks': {'house_price': 0.8, 'LTV': 1.1}
            }
        }
        ```
        This dictionary will populate the `st.selectbox` for pre-defined scenarios.
        Its content will be printed using `st.write` or `st.dataframe` in the Streamlit app.

### 4. Methodology Overview & Key Formulae (Main Content Area)
*   **Content**:
    ```markdown
    ## 4. Methodology Overview

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
    ```
*   **Streamlit Integration**: This comprehensive section will be displayed using `st.header` and `st.markdown` (with `r"$$...$$"` for display equations and `r"$...$"` for inline equations) in the main content area, providing the theoretical background before or alongside the implementation details.

### Core Stress Test & Visualization Logic
*   **Concepts from User Requirements & Pseudo-code**:
    *   Model Prediction Integration: `y_hat_baseline = mock_model.predict(X_baseline)` and `y_hat_stressed = mock_model.predict(X_stressed)`.
    *   Impact Metrics: `Δy_hat^(s) = y_hat^(s) - y_hat`, `EL = sum(PD * Exposure)`, `ΔEL^(s) = EL^(s) - EL_baseline`.
    *   Portfolio Aggregation: Logic to group `df_baseline` by `rating_band`, `sector`, `region` and calculate mean `ΔPD`, `ΔEL`.
    *   Sensitivity Analysis: Loop through `alpha` values (e.g., `x^(s,α) = x ⊙ (1 + α * δ_s)`) to generate a trajectory of `PD` for plots.
*   **Streamlit Integration**: This logic will be implemented directly within Streamlit callbacks or functions triggered by user actions (e.g., "Run Stress Test" button). Results will be stored in `st.session_state` and rendered as `st.dataframe`, `st.pyplot` plots (using `matplotlib` and `seaborn`), or `st.table`.

    *   **Example for stress application and prediction**:
        ```python
        # Triggered by "Run Stress Test" button
        if st.button("Run Stress Test") or 'stressed_results' not in st.session_state:
            with st.spinner("Applying stress scenario and predicting..."):
                current_scenario = st.session_state.get('selected_scenario', 'Macro Downturn')
                scenario_config = scenario_shocks_config[current_scenario] # Or user custom config
                alpha = st.session_state.get('alpha_value', 1.0) # From slider

                # Create X_stressed based on alpha path
                shocked_features = {
                    feat: val * (1 + alpha * (val - 1) if scenario_config['shock_type'] == 'multiplicative' else alpha * val)
                    for feat, val in scenario_config['shocks'].items()
                }

                X_stressed = apply_feature_shock(st.session_state['df_baseline'][numerical_features].copy(), shocked_features, scenario_config['shock_type'])
                y_hat_stressed = mock_model.predict(X_stressed)

                df_stressed = st.session_state['df_baseline'].copy()
                df_stressed['PD_stressed'] = y_hat_stressed
                df_stressed['Delta_PD'] = df_stressed['PD_stressed'] - df_stressed['PD_baseline']
                df_stressed['EL_baseline'] = df_stressed['PD_baseline'] * df_stressed['exposure']
                df_stressed['EL_stressed'] = df_stressed['PD_stressed'] * df_stressed['exposure']
                df_stressed['Delta_EL'] = df_stressed['EL_stressed'] - df_stressed['EL_baseline']

                st.session_state['df_stressed'] = df_stressed
                st.session_state['stressed_results_ready'] = True
        ```
    *   **Example for visualization**:
        ```python
        if st.session_state.get('stressed_results_ready'):
            df_stressed = st.session_state['df_stressed']
            # Plot distributions
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.histplot(df_stressed['PD_baseline'], kde=True, ax=axes[0], color='blue', label='Baseline PD')
            sns.histplot(df_stressed['PD_stressed'], kde=True, ax=axes[0], color='red', label='Stressed PD')
            axes[0].set_title(r'Distribution of PDs ($ \hat{y} $ vs. $ \hat{y}^{(s)} $)')
            axes[0].legend()

            sns.histplot(df_stressed['Delta_PD'], kde=True, ax=axes[1], color='green')
            axes[1].set_title(r'Distribution of Change in PD ($ \Delta \hat{y}^{(s)} $)')

            st.pyplot(fig)

            # ... other plots and tables ...
        ```

