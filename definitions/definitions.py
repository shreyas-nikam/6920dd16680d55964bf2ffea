import pandas as pd
import numpy as np
from scipy.special import expit

def generate_financial_data(n_samples):
    """
    Generates a synthetic tabular dataset simulating a credit portfolio with
    various financial features and a target variable (PD_target).

    Arguments:
        n_samples: The number of rows (samples) to generate in the DataFrame.

    Output:
        A pandas.DataFrame containing synthetic financial features, an exposure column,
        and a PD_target column.
    """
    # Input validation
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")
    if n_samples < 0:
        raise ValueError("n_samples cannot be negative.")

    # Handle the case of zero samples by returning an empty DataFrame with expected columns
    if n_samples == 0:
        EXPECTED_COLUMNS = [
            'income', 'debt_to_income', 'utilization', 'house_price', 'LTV',
            'credit_spread', 'volatility', 'liquidity',
            'rating_band', 'sector', 'region',
            'exposure', 'PD_target'
        ]
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Initialize a dictionary to store generated data
    data = {}

    # --- Generate Numerical Features ---

    # income: Log-normal distribution to simulate varying income levels (e.g., ~$50k-$500k)
    data['income'] = np.random.lognormal(mean=11.5, sigma=0.8, size=n_samples)

    # house_price: Correlated with income, usually a multiple of income
    data['house_price'] = data['income'] * np.random.uniform(1.5, 5.0, size=n_samples)

    # debt_to_income: Uniform distribution, clipped to [0.1, 0.6]
    data['debt_to_income'] = np.random.uniform(0.05, 0.7, size=n_samples)
    data['debt_to_income'] = np.clip(data['debt_to_income'], 0.1, 0.6)

    # utilization: Uniform distribution, clipped to [0, 1]
    data['utilization'] = np.random.uniform(-0.1, 1.1, size=n_samples)
    data['utilization'] = np.clip(data['utilization'], 0, 1)

    # LTV (Loan-to-Value): Uniform distribution, clipped to [0.2, 0.9]
    data['LTV'] = np.random.uniform(0.1, 1.0, size=n_samples)
    data['LTV'] = np.clip(data['LTV'], 0.2, 0.9)

    # credit_spread: Uniform distribution, clipped to [0.005, 0.05]
    data['credit_spread'] = np.random.uniform(0.003, 0.06, size=n_samples)
    data['credit_spread'] = np.clip(data['credit_spread'], 0.005, 0.05)

    # volatility: Uniform distribution, clipped to [0.01, 0.05]
    data['volatility'] = np.random.uniform(0.005, 0.06, size=n_samples)
    data['volatility'] = np.clip(data['volatility'], 0.01, 0.05)

    # liquidity: Uniform distribution, clipped to [0.1, 1]
    data['liquidity'] = np.random.uniform(0.05, 1.2, size=n_samples)
    data['liquidity'] = np.clip(data['liquidity'], 0.1, 1)

    # exposure: Positive uniform distribution for loan/credit exposure
    data['exposure'] = np.random.uniform(10_000, 1_000_000, size=n_samples)

    # --- Generate Categorical Features ---

    # rating_band: Common credit rating categories
    rating_bands = ['A', 'B', 'C', 'D']
    data['rating_band'] = np.random.choice(rating_bands, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

    # sector: Industry sectors
    sectors = ['Finance', 'Tech', 'Retail', 'Manufacturing']
    data['sector'] = np.random.choice(sectors, size=n_samples, p=[0.25, 0.25, 0.25, 0.25])

    # region: Geographical regions
    regions = ['North', 'South', 'East', 'West']
    data['region'] = np.random.choice(regions, size=n_samples, p=[0.25, 0.25, 0.25, 0.25])

    # Create initial DataFrame
    df = pd.DataFrame(data)

    # --- Calculate PD_target (Probability of Default) ---

    # Map rating_band to a numerical score for PD calculation
    # Higher score indicates worse rating and thus higher PD
    rating_score_map = {'A': -1.5, 'B': -0.5, 'C': 0.5, 'D': 1.5}
    df['rating_score_temp'] = df['rating_band'].map(rating_score_map)

    # Define a linear predictor (logits) for PD using relevant features
    # Lower income, higher DTI, higher utilization, higher LTV, and worse rating
    # are assumed to increase PD.
    base_logits = -4.0 # Base intercept for overall PD level

    pd_logits = base_logits \
                + df['debt_to_income'] * 6 \
                + df['utilization'] * 4 \
                + df['LTV'] * 5 \
                + (1 / (df['income'] / 100_000)) * 0.5 \
                + df['rating_score_temp'] * 1.5 \
                + np.random.normal(0, 0.7, size=n_samples) # Add some random noise

    # Apply the sigmoid (expit) function to convert logits to probabilities (PD)
    df['PD_target'] = expit(pd_logits)

    # Clip PD_target to ensure it's strictly within (0,1) for robustness,
    # as expit approaches but doesn't mathematically reach 0 or 1.
    df['PD_target'] = np.clip(df['PD_target'], 0.0001, 0.9999)

    # Drop the temporary column used for PD calculation
    df = df.drop(columns=['rating_score_temp'])

    return df

import numpy as np

class MockMLModel:
    def __init__(self):
        """Initializes the MockMLModel with predefined fixed "coefficients" (weights) and an "intercept".
        These values mimic a simple linear model's parameters.
        """
        self.coefficients = np.array([-0.5, 2.0, 1.5, 1.0, 3.0])
        self.intercept = -5.0

import pandas as pd
import numpy as np
from scipy.special import expit

class MockMLModel:
    def __init__(self):
        """
        Initializes the MockMLModel with predefined coefficients, intercept, and
        expected feature names, simulating a pre-trained model.
        """
        self.coefficients = np.array([-0.5, 2.0, 1.5, 1.0, 3.0])
        self.intercept = -5.0
        self.expected_features = ['income', 'debt_to_income', 'utilization', 'LTV', 'credit_spread']

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Simulates a pre-trained ML model's predictions (e.g., PD scores) based on input features
        without actual model training. It selects a subset of numerical features from the input
        DataFrame, computes a linear combination using internal coefficients and intercept,
        and applies a sigmoid function to output probabilities.

        Arguments:
            self: The instance of the class.
            X: A pandas.DataFrame containing the input features for which to generate predictions.

        Output:
            A numpy.ndarray of predicted probabilities (e.g., PD scores) between 0 and 1.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas.DataFrame.")

        # Check for missing required features
        missing_features = [f for f in self.expected_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Input DataFrame is missing required features: {', '.join(missing_features)}")

        # Select the relevant features in the expected order
        X_processed = X[self.expected_features]

        # Validate that all selected features are numerical
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                raise TypeError(f"Column '{col}' must contain numerical data, but found non-numeric types.")

        # Convert the processed DataFrame to a NumPy array for dot product calculation
        X_values = X_processed.to_numpy(dtype=float)

        # Compute the linear combination: dot product of features with coefficients plus intercept
        linear_output = np.dot(X_values, self.coefficients) + self.intercept

        # Apply the sigmoid function to convert linear output to probabilities
        probabilities = expit(linear_output)

        return probabilities

import pandas as pd

def apply_feature_shock(df, shocks_dict, shock_type):
    """Applies quantitative shocks to specified features in a DataFrame.

    Arguments:
        df: The input pandas.DataFrame.
        shocks_dict: Dictionary {feature_name: shock_factor/value}.
        shock_type: 'multiplicative' or 'additive'.

    Output:
        A new pandas.DataFrame with shocked features.
    """
    if shock_type not in ['multiplicative', 'additive']:
        raise ValueError("Invalid shock_type. Must be 'multiplicative' or 'additive'.")

    # Create a copy to avoid modifying the original DataFrame
    df_shocked = df.copy()

    for feature, shock_value in shocks_dict.items():
        if feature not in df_shocked.columns:
            # This will naturally raise a KeyError if a non-existent column is accessed,
            # which is the desired behavior for test_shock_non_existent_feature_raises_key_error.
            # We can explicitly raise it for clarity, but pandas access will do it too.
            raise KeyError(f"Feature '{feature}' specified in shocks_dict not found in DataFrame columns.")

        if shock_type == 'multiplicative':
            df_shocked[feature] = df_shocked[feature] * shock_value
        elif shock_type == 'additive':
            df_shocked[feature] = df_shocked[feature] + shock_value
            
    return df_shocked

import pandas as pd
import numpy as np

def get_predictions(model, X):
    """A wrapper function to obtain predictions from a given machine learning model.

    Arguments:
        model: An instance of a machine learning model that has a 'predict' method.
        X: A pandas.DataFrame containing the input features.
    Output:
        A numpy.ndarray of predictions generated by the model.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input 'X' must be a pandas.DataFrame.")

    if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
        raise AttributeError("The 'model' object must have a callable 'predict' method.")

    predictions = model.predict(X)

    # Ensure the output is a numpy.ndarray as per the docstring
    if not isinstance(predictions, np.ndarray):
        predictions = np.asarray(predictions)
        
    return predictions

import numpy as np

def calculate_prediction_change(y_stressed, y_baseline):
    """
    Calculates the element-wise difference between stressed predictions and baseline predictions. This metric quantifies the individual change in model output for each data point under a specific stress scenario.
    Arguments:
        y_stressed: A numpy.ndarray of predictions from the model under stress conditions.
        y_baseline: A numpy.ndarray of predictions from the model under baseline (normal) conditions.
    Output:
        A numpy.ndarray representing the element-wise difference (y_stressed - y_baseline).
    """
    if not isinstance(y_stressed, np.ndarray):
        raise TypeError("y_stressed must be a numpy.ndarray.")
    if not isinstance(y_baseline, np.ndarray):
        raise TypeError("y_baseline must be a numpy.ndarray.")

    if y_stressed.shape != y_baseline.shape:
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Got y_stressed shape {y_stressed.shape} and y_baseline shape {y_baseline.shape}."
        )

    return y_stressed - y_baseline

import numpy as np

def calculate_mean_delta(y_stressed, y_baseline):
    """Calculates the change in the mean of predictions between a stressed scenario and a baseline.

    Arguments:
        y_stressed: A numpy.ndarray of predictions from the model under stress conditions.
        y_baseline: A numpy.ndarray of predictions from the model under baseline (normal) conditions.
    Output:
        A float representing the difference between the mean of stressed predictions and the mean of baseline predictions.
    """
    mean_stressed = np.mean(y_stressed)
    mean_baseline = np.mean(y_baseline)
    return mean_stressed - mean_baseline

import numpy as np

def calculate_expected_loss(predictions, exposures):
    """Calculates the total Expected Loss (EL) for a portfolio."""
    # Ensure shapes are compatible; numpy will raise ValueError if not.
    # This directly handles test case 4 for mismatched shapes.
    element_wise_loss = predictions * exposures

    # Sum the element-wise products to get the total Expected Loss.
    total_expected_loss = np.sum(element_wise_loss)

    # Ensure the output is a float, as specified.
    # np.sum typically returns a float if the input array contains floats,
    # but an explicit cast ensures it, especially for empty arrays which sum to 0
    # and for arrays of integers which might sum to an integer type.
    return float(total_expected_loss)

import pandas as pd
import numpy as np

def aggregate_by_segment(df, segment_col, metric_col, agg_func):
    """
    Aggregates a specified metric within a DataFrame by a categorical segment column using a provided aggregation function.
    This helps in understanding the impact of scenarios on different portfolio segments.

    Arguments:
        df: The input pandas.DataFrame containing the data.
        segment_col: A string representing the name of the categorical column to group by (e.g., 'rating_band').
        metric_col: A string representing the name of the numerical column to aggregate (e.g., 'Delta_PD').
        agg_func: A callable function (e.g., np.mean, np.sum) to apply for aggregation.

    Output:
        A pandas.Series showing the aggregated metric for each segment.
    """
    # Ensure columns exist before proceeding to avoid unexpected errors downstream
    # (Though pandas groupby/agg will raise KeyError if columns don't exist, this
    # explicit check can be added for earlier error detection if desired,
    # but the current tests expect pandas' default KeyError behavior).
    
    # Group by the segment column and apply the aggregation function to the metric column
    return df.groupby(segment_col)[metric_col].agg(agg_func)

import pandas as pd
import numpy as np

# Assuming MockMLModel and apply_feature_shock are defined elsewhere in this module
# or accessible through appropriate imports if this code were part of a larger project.
# For the purpose of completing the stub, we assume they are available in the scope.

# MockMLModel and apply_feature_shock are likely defined similarly to these examples
# if they were to be included in the same file for demonstration purposes.
# As per the test structure, they are expected to be in the same module where
# run_single_scenario is defined.

class MockMLModel:
    """A simple mock machine learning model for testing purposes."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates dummy predictions. If 'feature1' and 'feature2' exist,
        it sums them. Otherwise, returns zeros. Handles empty DataFrames.
        """
        if X.empty:
            return np.array([])
        
        pred_features = [col for col in ['feature1', 'feature2'] if col in X.columns]
        if not pred_features:
            # If no relevant features, return an array of zeros matching the number of rows
            return np.zeros(len(X))
            
        # Sum the relevant features and convert to numpy array
        return X[pred_features].sum(axis=1).values

def apply_feature_shock(df: pd.DataFrame, shocks_dict: dict, shock_type: str = 'multiplicative') -> pd.DataFrame:
    """
    Applies specified shocks to features in a DataFrame.

    Args:
        df: The input pandas DataFrame.
        shocks_dict: A dictionary where keys are feature names and values are shock magnitudes.
        shock_type: The type of shock to apply ('multiplicative' or 'additive').

    Returns:
        A new pandas DataFrame with shocks applied.
    
    Raises:
        ValueError: If an unknown shock_type is provided.
    """
    df_stressed = df.copy()
    for feature, shock_value in shocks_dict.items():
        if feature in df_stressed.columns:
            if shock_type == 'multiplicative':
                df_stressed[feature] *= shock_value
            elif shock_type == 'additive':
                df_stressed[feature] += shock_value
            else:
                raise ValueError(f"Unknown shock_type: {shock_type}")
    return df_stressed


def run_single_scenario(model, X_baseline, scenario_name, scenario_shocks, shock_type):
    """
    Orchestrates the application of a single financial stress scenario to baseline features,
    generates stressed predictions, and returns both baseline and stressed predictions.
    This function encapsulates the steps for evaluating one specific scenario's impact.

    Arguments:
        model: An instance of a machine learning model (e.g., MockMLModel) for making predictions.
        X_baseline: A pandas.DataFrame of original, unstressed input features.
        scenario_name: A string identifying the current scenario (e.g., 'Macro Downturn').
        scenario_shocks: A dictionary specifying the feature-level shocks for this scenario.
        shock_type: A string indicating the type of shock ('multiplicative' or 'additive').
    Output:
        A dictionary containing the baseline predictions and the stressed predictions for the scenario.
    """
    # 1. Generate baseline predictions
    baseline_predictions = model.predict(X_baseline)

    # 2. Apply feature shocks to create the stressed DataFrame
    X_stressed = apply_feature_shock(X_baseline, scenario_shocks, shock_type=shock_type)

    # 3. Generate stressed predictions
    stressed_predictions = model.predict(X_stressed)

    # 4. Return results
    return {
        'scenario_name': scenario_name,
        'baseline_predictions': baseline_predictions,
        'stressed_predictions': stressed_predictions
    }

import pandas as pd
import numpy as np

def run_multiple_scenarios(model, X_baseline: pd.DataFrame, scenarios_config: dict) -> dict:
    """    Iterates through a dictionary of predefined financial stress scenarios, applies each scenario's shocks to the baseline features, and collects the stressed predictions from the model for every scenario. This automates comprehensive robustness testing across various scenarios.
Arguments:
    model: An instance of a machine learning model (e.g., MockMLModel) for making predictions.
    X_baseline: A pandas.DataFrame of original, unstressed input features.
    scenarios_config: A dictionary where keys are scenario names and values are dictionaries containing 'shock_type' and 'shocks' for each scenario.
Output:
    A dictionary where keys are scenario names and values are numpy.ndarrays of stressed predictions for that scenario.
    """

    scenario_predictions = {}

    for scenario_name, config in scenarios_config.items():
        # Extract shock type and shocks from the scenario configuration.
        # This will raise a KeyError if 'shock_type' or 'shocks' are missing,
        # handling malformed configuration as per Test Case 5.
        shock_type = config['shock_type']
        shocks = config['shocks']

        # Apply the feature shocks to the baseline data.
        # The 'apply_feature_shock' function is assumed to be available
        # in the same module/scope as per the test setup.
        X_stressed = apply_feature_shock(X_baseline, shocks, shock_type)

        # Make predictions using the model on the stressed features.
        # This will raise an AttributeError if the model lacks a 'predict' method,
        # handling invalid model objects as per Test Case 4.
        predictions = model.predict(X_stressed)

        scenario_predictions[scenario_name] = predictions

    return scenario_predictions

import pandas as pd
import numpy as np

# Assuming 'apply_feature_shock' is imported or defined in the same scope
# as indicated by the test setup patching it from the module 'definition_b5ea310f3a4740f7ab7128e455e5bb98'.

def run_alpha_path_analysis(model, X_baseline, base_shocks, alpha_steps, shock_type):
    """
    Conducts an 'alpha path' analysis by applying a scenario with varying intensity (alpha) to the baseline features.
    It collects aggregated metrics (e.g., mean PD) at each alpha step to observe potential non-linear model responses or sensitivities to increasing stress severity.

    Arguments:
        model: An instance of a machine learning model (e.g., MockMLModel) for making predictions.
        X_baseline: A pandas.DataFrame of original, unstressed input features.
        base_shocks: A dictionary defining the full intensity (alpha=1) shocks for the scenario.
        alpha_steps: A list of float values (typically from 0 to 1) representing the varying intensity levels.
        shock_type: A string indicating the type of shock ('multiplicative' or 'additive').

    Output:
        A pandas.DataFrame with 'Alpha' values and the corresponding aggregated metric (e.g., 'Mean PD') at each intensity level.
    """
    results = []

    for alpha in alpha_steps:
        # Calculate current shock values by scaling base_shocks with alpha
        current_shocks_dict = {feature: value * alpha for feature, value in base_shocks.items()}
        
        # Make a copy of the baseline features to apply shocks
        X_stressed = X_baseline.copy()
        
        # Apply the current shocks using the external apply_feature_shock function
        X_stressed = apply_feature_shock(X_stressed, current_shocks_dict, shock_type)
        
        # Get predictions from the model
        y_hat = model.predict(X_stressed)
        
        # Calculate the aggregated metric (Mean PD)
        mean_pd = np.mean(y_hat)
        
        results.append({'Alpha': alpha, 'Mean PD': mean_pd})

    # Convert results to a pandas DataFrame
    result_df = pd.DataFrame(results)
    
    # If alpha_steps was empty, result_df will be empty. Ensure columns are correctly set.
    if result_df.empty and not alpha_steps:
        return pd.DataFrame(columns=['Alpha', 'Mean PD'])

    return result_df