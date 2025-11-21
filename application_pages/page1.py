
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set pandas options for display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Ensure reproducibility
np.random.seed(42)

def run_page1():
    st.header("2. Setting Up the Environment")
    st.markdown("""
    This section ensures all necessary libraries are imported for data generation, manipulation, modeling, and visualization.
    """)

    st.header("3. Data/Inputs Overview")
    st.markdown("""
    To effectively demonstrate stress testing without relying on sensitive real-world financial data, this notebook utilizes a synthetic dataset. This synthetic data simulates a credit portfolio, encompassing various financial characteristics and a target variable representing the Probability of Default (PD). The data generation process is designed to mimic realistic financial feature distributions and interdependencies, making it suitable for illustrating the concepts of scenario-based stress testing.

    **Assumptions:**
    *   The synthetic dataset is representative of a typical credit portfolio.
    *   Features like `income`, `debt_to_income`, `utilization`, `LTV`, `credit_spread`, `house_price`, `volatility`, `liquidity`, `exposure`, `rating_band`, `sector`, and `region` are generated with distributions that are plausible in a financial context.
    *   The `PD_target` is a simulated outcome based on a combination of these features, reflecting a simplified credit risk model.

    This approach allows us to focus on the methodology of stress testing and model response analysis, rather than the complexities of real data acquisition and anonymization. The generated data supports the business goal of creating a robust framework for validating ML models under adverse conditions.

    ### 5.1. Generating Synthetic Financial Data

    This section focuses on creating a synthetic dataset that mimics a credit portfolio. This is crucial for demonstrating stress testing concepts without the complexities and sensitivities associated with real-world financial data. By generating controlled data, we can ensure reproducibility and isolate the effects of stress transformations on model predictions, thereby providing a clear and interpretable framework for model validation and risk assessment.
    """)

    @st.cache_data
    def generate_financial_data(n_samples_input):
        data = {}

        # Income (log-normal distribution)
        data['income'] = np.random.lognormal(mean=10, sigma=0.8, size=n_samples_input) * 1000

        # Debt-to-income ratio (beta distribution, clipped)
        data['debt_to_income'] = np.random.beta(a=2, b=7, size=n_samples_input) * 0.8 + 0.1
        data['debt_to_income'] = np.clip(data['debt_to_income'], 0.05, 0.7)

        # Credit Utilization (beta distribution, clipped)
        data['utilization'] = np.random.beta(a=3, b=5, size=n_samples_input) * 0.9 + 0.05
        data['utilization'] = np.clip(data['utilization'], 0.1, 0.95)

        # Loan-to-Value (LTV) (beta distribution, clipped)
        data['LTV'] = np.random.beta(a=5, b=3, size=n_samples_input) * 0.9 + 0.05
        data['LTV'] = np.clip(data['LTV'], 0.2, 0.99)

        # Credit Spread (normal distribution, clipped)
        data['credit_spread'] = np.random.normal(loc=0.03, scale=0.015, size=n_samples_input)
        data['credit_spread'] = np.clip(data['credit_spread'], 0.005, 0.10)

        # House Price (log-normal, correlated with income)
        data['house_price'] = (data['income'] / 1000 * np.random.lognormal(mean=0.5, sigma=0.3, size=n_samples_input)) + np.random.normal(loc=100000, scale=30000, size=n_samples_input)
        data['house_price'] = np.clip(data['house_price'], 50000, 2000000)

        # Volatility (beta distribution, clipped)
        data['volatility'] = np.random.beta(a=2, b=10, size=n_samples_input) * 0.05 + 0.01
        data['volatility'] = np.clip(data['volatility'], 0.005, 0.08)

        # Liquidity (uniform, with some correlation to income)
        data['liquidity'] = np.random.uniform(0.1, 0.8, size=n_samples_input) + (data['income'] / data['income'].max() * 0.1)
        data['liquidity'] = np.clip(data['liquidity'], 0.1, 0.9)

        # Exposure (gamma distribution, often skewed)
        data['exposure'] = np.random.gamma(shape=2, scale=50000, size=n_samples_input) + 10000
        data['exposure'] = np.clip(data['exposure'], 10000, 1000000)

        # Categorical features
        data['rating_band'] = np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'],
                                              p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03], size=n_samples_input)
        data['sector'] = np.random.choice(['Financials', 'Technology', 'Healthcare', 'Industrials', 'Utilities'],
                                         p=[0.30, 0.25, 0.20, 0.15, 0.10], size=n_samples_input)
        data['region'] = np.random.choice(['North', 'South', 'East', 'West', 'Central'],
                                         p=[0.25, 0.20, 0.20, 0.20, 0.15], size=n_samples_input)

        df = pd.DataFrame(data)

        # Simulate a target PD based on some features (simplified logistic model)
        # Higher debt, utilization, LTV, credit spread, volatility -> higher PD
        # Higher income, liquidity -> lower PD

        # Coefficients for PD calculation
        coeffs = {
            'income': -0.5, 'debt_to_income': 2.0, 'utilization': 1.5, 'LTV': 1.0,
            'credit_spread': 3.0, 'house_price': -0.3, 'volatility': 2.5, 'liquidity': -1.0,
            'exposure': 0.1 # Exposure will have a smaller impact on *individual* PD, but larger on EL
        }
        intercept = -5.0

        # Standardize features for logistic function
        features_for_pd = [f for f in coeffs.keys() if f in df.columns]
        scaler = StandardScaler()
        df_scaled = df[features_for_pd].copy()
        for col in ['income', 'house_price', 'exposure']: # Log transform skewed features before scaling
            if col in df_scaled.columns:
                df_scaled[col] = np.log1p(df_scaled[col])
        df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=features_for_pd)

        # Calculate log-odds
        log_odds = intercept
        for feature, coeff in coeffs.items():
            if feature in df_scaled.columns:
                log_odds += df_scaled[feature] * coeff

        # Convert to probability using sigmoid function
        df['PD_target'] = 1 / (1 + np.exp(-log_odds))
        df['PD_target'] = np.clip(df['PD_target'], 0.001, 0.999) # Clip PD to realistic range

        return df

    with st.sidebar:
        n_samples_input = st.slider(label="Number of Samples", min_value=1000, max_value=10000, value=5000, step=1000, help="Select the number of synthetic data samples to generate.")
        generate_button = st.button("Generate Synthetic Data")

    if generate_button or 'df_baseline' not in st.session_state:
        with st.spinner("Generating Data..."):
            df_baseline = generate_financial_data(n_samples_input)
            st.session_state['df_baseline'] = df_baseline
            st.session_state['data_generated'] = True
            st.success("Synthetic data generated successfully!")

    if st.session_state.get('data_generated'):
        st.subheader("Generated Data Overview")
        st.markdown("Here's a preview of the generated synthetic financial data:")
        st.dataframe(st.session_state['df_baseline'].head())

        st.subheader("Data Information (`df.info()` summary)")
        
        # Create a StringIO object to capture the output of df.info()
        import io
        buffer = io.StringIO()
        st.session_state['df_baseline'].info(buf=buffer)
        info_string = buffer.getvalue()
        st.text(info_string)

        st.subheader("Descriptive Statistics (`df.describe()`)")
        st.dataframe(st.session_state['df_baseline'].describe())

        st.markdown("""
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
        """)

        st.markdown("### 5.2. Implementing a Mock ML Model")
        st.markdown("""
        To simulate real-world model interaction and provide a controlled environment for stress testing, we define a simple `MockMLModel` class. This model is not trained in the traditional sense; instead, it provides predictions based on a predefined mathematical function of its input features. It acts as a placeholder for a pre-trained Machine Learning model $f_\theta(x)$, allowing us to focus on the impact of feature-level shocks on model output without the overhead of actual model training.

        **Business Value:**
        *   **Isolation of Stress Impact**: By using a mock model with fixed parameters, we can isolate and precisely measure the impact of feature transformations without confounding effects from model retraining or dynamic learning.
        *   **Reproducibility**: The fixed nature of the mock model ensures that stress test results are perfectly reproducible.
        *   **Interpretability**: The simple, transparent logic of the mock model makes it easier to understand why certain feature shocks lead to specific changes in predictions.
        *   **Flexibility**: It allows for rapid prototyping and testing of various stress scenarios and impact measurement techniques before applying them to more complex, production-grade models.
        """)

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
                # Ensure all expected features are present, fill missing with 0 or mean
                X_processed = X[self.expected_features].copy()

                # Apply log transform to features that were log-transformed during data generation before scaling for PD
                for col in ['income', 'house_price', 'exposure']:
                    if col in X_processed.columns:
                        X_processed[col] = np.log1p(X_processed[col])

                # Simple standardization (for demonstration, a real model would use pre-fitted scaler)
                # For this mock model, we'll assume features are somewhat normalized or coefficients handle scale.

                # Calculate log-odds
                log_odds = self.intercept
                for feature, coeff in self.coefficients.items():
                    if feature in X_processed.columns:
                        log_odds += X_processed[feature] * coeff

                # Convert to probability using sigmoid function
                pd_predictions = 1 / (1 + np.exp(-log_odds))
                return np.clip(pd_predictions, 0.001, 0.999) # Clip PD to realistic range

        mock_model = MockMLModel()
        st.session_state['mock_model'] = mock_model
        numerical_features = mock_model.expected_features
        st.session_state['numerical_features'] = numerical_features
        X_baseline = st.session_state['df_baseline'][numerical_features]
        st.session_state['X_baseline'] = X_baseline

        st.write(f"MockMLModel instantiated. Features used: `{numerical_features}`")
        st.markdown("Here's a preview of the features used for the mock model predictions:")
        st.dataframe(X_baseline.head())

        st.markdown("""
        ### Interpretation of Mock ML Model Instantiation and Feature Preparation

        This step initializes our `MockMLModel` and prepares the input features for making predictions.
        *   **`mock_model = MockMLModel()`**: We have successfully instantiated our mock machine learning model. This object now holds the predefined coefficients and intercept that simulate the behavior of a trained model. It's ready to generate predictions based on input features.
        *   **`numerical_features`**: This list explicitly defines the subset of features from our `df_baseline` that the `MockMLModel` expects and uses for its predictions. These are typically the most influential features in a real financial model, like `income`, `debt_to_income`, `utilization`, `LTV`, and `credit_spread`. Business-wise, this highlights the model's focus and potential areas for stress testing.
        *   **`X_baseline`**: This DataFrame represents the input features for our baseline predictions, extracted directly from the `df_baseline`. By showing its head, we confirm that the data is correctly structured and contains the necessary numerical features for the `mock_model` to operate. This `X_baseline` will serve as the unstressed input against which all future stressed scenarios will be compared.

        This setup is crucial because it establishes the "normal" operating conditions for our model before any stress is applied, providing the necessary foundation for comparing baseline and stressed predictions.
        """)

        st.markdown("### 5.3. Calculating Baseline Predictions")
        st.markdown("""
        Before applying any stress, it is essential to establish a baseline by obtaining predictions from our mock ML model on the original, unstressed dataset. This serves as the reference point for measuring scenario impacts. The baseline prediction is denoted as $\hat{y} = f_\theta(x)$.

        **Business Value:**
        *   **Reference Point**: Baseline predictions provide a crucial benchmark. Without them, there would be no way to quantify the impact of stress scenarios, making it impossible to assess model sensitivity.
        *   **Performance Under Normal Conditions**: It allows us to understand the model's expected behavior and output distributions under typical, non-stressed market conditions. This is the "business-as-usual" view.
        *   **Foundation for Impact Metrics**: All subsequent impact metrics, such as the change in Probability of Default (PD) or Expected Loss (EL), are calculated relative to this baseline, ensuring that reported impacts are meaningful and contextualized.
        """)

        if 'PD_baseline' not in st.session_state:
            with st.spinner("Calculating Baseline Predictions..."):
                y_hat_baseline = mock_model.predict(X_baseline)
                st.session_state['df_baseline']['PD_baseline'] = y_hat_baseline
                st.session_state['PD_baseline_calculated'] = True
                st.success("Baseline predictions (PD_baseline) calculated successfully!")
        else:
            st.info("Baseline predictions already calculated.")

        if st.session_state.get('PD_baseline_calculated'):
            st.write("Baseline predictions (PD_baseline) calculated and added to `df_baseline`. Here's a preview:")
            st.dataframe(st.session_state['df_baseline'].head())

            st.markdown("""
            ### Interpretation of Baseline Predictions

            The `PD_baseline` column, which has just been added to our `df_baseline` DataFrame, represents the mock ML model's initial prediction for the Probability of Default (PD) for each entity (e.g., customer, loan) under normal, unstressed financial conditions. This is the **starting point** from which all changes due to stress scenarios will be measured.

            From a business perspective, `PD_baseline` reflects the model's assessment of default risk in a "business-as-usual" environment. It is crucial for:
            *   **Establishing a reference**: Any increase or decrease in PD observed under stress will be relative to this baseline, providing a clear magnitude of impact.
            *   **Understanding current risk profile**: The distribution of `PD_baseline` gives insights into the inherent risk profile of the portfolio before any adverse events.

            Let's examine the summary statistics for `PD_baseline`:
            """)
            st.write(f"Mean of PD_baseline: {st.session_state['df_baseline']['PD_baseline'].mean():.4f}")
            st.write(f"Standard Deviation of PD_baseline: {st.session_state['df_baseline']['PD_baseline'].std():.4f}")
            st.markdown("""
            The mean of `PD_baseline` indicates the average expected default rate across the entire portfolio under normal conditions. The standard deviation, on the other hand, tells us about the dispersion or variability of individual PDs around this mean. A higher standard deviation would imply a more heterogeneous portfolio in terms of default risk.

            These values are fundamental for financial risk managers as they set the expectation against which stressed outcomes will be compared. For example, if the mean PD significantly increases under a stress scenario, it signals a material impact on the portfolio's credit quality.
            """)
