import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Keep the placeholder block
from definition_da5bd0de9fea4be597daa54697b9aac0 import run_single_scenario
from definition_da5bd0de9fea4be597daa54697b9aac0 import MockMLModel, apply_feature_shock # Assuming these are also in your_module

# --- Mocks for the tests (if not directly patching functions/classes) ---
# For run_single_scenario, we need to mock MockMLModel and apply_feature_shock
# The notebook spec shows MockMLModel is a class and apply_feature_shock is a function.
# We will use the actual MockMLModel and apply_feature_shock from the module if available,
# or create simple mocks if they are not meant to be imported directly for testing.
# Based on the prompt, it implies these helper functions/classes are part of the module.

# If MockMLModel and apply_feature_shock were internal or in another module,
# we would define simple test-specific mocks here:
# class SimpleMockMLModel:
#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         if X.empty: return np.array([])
#         # Simple prediction logic for testing: sum first two numeric columns if they exist
#         pred_features = [col for col in ['feature1', 'feature2'] if col in X.columns]
#         if not pred_features: return np.zeros(len(X))
#         return X[pred_features].sum(axis=1).values
#
# def simple_mock_apply_feature_shock(df: pd.DataFrame, shocks_dict: dict, shock_type: str = 'multiplicative') -> pd.DataFrame:
#     df_stressed = df.copy()
#     for feature, shock_value in shocks_dict.items():
#         if feature not in df_stressed.columns: continue # Or raise ValueError depending on spec
#         if shock_type == 'multiplicative': df_stressed[feature] *= shock_value
#         elif shock_type == 'additive': df_stressed[feature] += shock_value
#         else: raise ValueError(f"Unknown shock_type: {shock_type}")
#     return df_stressed


# --- Pytest fixtures for common test data ---

@pytest.fixture
def mock_model_instance():
    """Returns an instance of the MockMLModel for testing."""
    # Assume MockMLModel in definition_da5bd0de9fea4be597daa54697b9aac0 has a simple constructor
    # If it needs specific init args, they would be passed here.
    return MockMLModel()

@pytest.fixture
def baseline_dataframe():
    """Returns a sample pandas DataFrame for X_baseline."""
    return pd.DataFrame({
        'feature1': [10.0, 20.0, 30.0],
        'feature2': [1.0, 2.0, 3.0],
        'other_feature': [100, 200, 300] # A feature not used in prediction/shocking
    })

@pytest.fixture
def empty_dataframe():
    """Returns an empty pandas DataFrame for X_baseline."""
    return pd.DataFrame(columns=['feature1', 'feature2', 'other_feature'])


# --- Test Cases for run_single_scenario ---

# Patching apply_feature_shock from definition_da5bd0de9fea4be597daa54697b9aac0
@patch('definition_da5bd0de9fea4be597daa54697b9aac0.apply_feature_shock')
def test_basic_multiplicative_scenario(mock_apply_shock, mock_model_instance, baseline_dataframe):
    """
    Tests a standard scenario with multiplicative shocks.
    Verifies that apply_feature_shock is called correctly and
    baseline/stressed predictions are as expected.
    """
    scenario_name = "Test Multiplicative"
    scenario_shocks = {'feature1': 0.8, 'feature2': 1.5}
    shock_type = 'multiplicative'

    # Manually calculate expected stressed DataFrame using apply_feature_shock logic
    # For a robust test, it's better to use the *actual* apply_feature_shock (if not patching)
    # or a known mock. Here we'll use a local mock of apply_feature_shock to calculate expected.
    # We define a temporary local_apply_feature_shock to determine the expected stressed_df.
    # The `mock_apply_shock` fixture will *replace* the module's function during the test.
    def local_apply_shock(df, shocks, s_type):
        df_copy = df.copy()
        for f, val in shocks.items():
            if f in df_copy.columns:
                if s_type == 'multiplicative': df_copy[f] *= val
                elif s_type == 'additive': df_copy[f] += val
        return df_copy

    # Setup the mock's return value for apply_feature_shock
    expected_stressed_df = local_apply_shock(baseline_dataframe, scenario_shocks, shock_type)
    mock_apply_shock.return_value = expected_stressed_df

    # Calculate expected predictions using MockMLModel's logic (e.g., sum of feature1 and feature2)
    expected_baseline_preds = mock_model_instance.predict(baseline_dataframe)
    expected_stressed_preds = mock_model_instance.predict(expected_stressed_df)

    # Run the function under test
    results = run_single_scenario(mock_model_instance, baseline_dataframe, scenario_name, scenario_shocks, shock_type)

    # Assertions
    mock_apply_shock.assert_called_once_with(baseline_dataframe, scenario_shocks, shock_type=shock_type)
    np.testing.assert_array_almost_equal(results['baseline_predictions'], expected_baseline_preds)
    np.testing.assert_array_almost_equal(results['stressed_predictions'], expected_stressed_preds)
    assert 'scenario_name' in results and results['scenario_name'] == scenario_name


@patch('definition_da5bd0de9fea4be597daa54697b9aac0.apply_feature_shock')
def test_basic_additive_scenario(mock_apply_shock, mock_model_instance, baseline_dataframe):
    """
    Tests a standard scenario with additive shocks.
    Ensures correct shock application and prediction calculation.
    """
    scenario_name = "Test Additive"
    scenario_shocks = {'feature1': -5.0, 'feature2': 10.0}
    shock_type = 'additive'

    def local_apply_shock(df, shocks, s_type):
        df_copy = df.copy()
        for f, val in shocks.items():
            if f in df_copy.columns:
                if s_type == 'multiplicative': df_copy[f] *= val
                elif s_type == 'additive': df_copy[f] += val
        return df_copy

    expected_stressed_df = local_apply_shock(baseline_dataframe, scenario_shocks, shock_type)
    mock_apply_shock.return_value = expected_stressed_df

    expected_baseline_preds = mock_model_instance.predict(baseline_dataframe)
    expected_stressed_preds = mock_model_instance.predict(expected_stressed_df)

    results = run_single_scenario(mock_model_instance, baseline_dataframe, scenario_name, scenario_shocks, shock_type)

    mock_apply_shock.assert_called_once_with(baseline_dataframe, scenario_shocks, shock_type=shock_type)
    np.testing.assert_array_almost_equal(results['baseline_predictions'], expected_baseline_preds)
    np.testing.assert_array_almost_equal(results['stressed_predictions'], expected_stressed_preds)
    assert 'scenario_name' in results and results['scenario_name'] == scenario_name


@patch('definition_da5bd0de9fea4be597daa54697b9aac0.apply_feature_shock')
def test_empty_baseline_dataframe(mock_apply_shock, mock_model_instance, empty_dataframe):
    """
    Tests behavior when X_baseline is an empty DataFrame.
    Should return empty numpy arrays for predictions.
    """
    scenario_name = "Empty Data Scenario"
    scenario_shocks = {'feature1': 0.8}
    shock_type = 'multiplicative'

    # Mock apply_feature_shock to return an empty DataFrame if given one
    mock_apply_shock.return_value = empty_dataframe.copy()

    results = run_single_scenario(mock_model_instance, empty_dataframe, scenario_name, scenario_shocks, shock_type)

    mock_apply_shock.assert_called_once_with(empty_dataframe, scenario_shocks, shock_type=shock_type)
    assert 'baseline_predictions' in results and len(results['baseline_predictions']) == 0
    assert 'stressed_predictions' in results and len(results['stressed_predictions']) == 0
    assert 'scenario_name' in results and results['scenario_name'] == scenario_name


@patch('definition_da5bd0de9fea4be597daa54697b9aac0.apply_feature_shock')
def test_empty_scenario_shocks(mock_apply_shock, mock_model_instance, baseline_dataframe):
    """
    Tests scenario where scenario_shocks is empty.
    Stressed predictions should be identical to baseline predictions.
    """
    scenario_name = "No Shocks Scenario"
    scenario_shocks = {} # Empty dictionary of shocks
    shock_type = 'multiplicative' # Type doesn't matter if shocks are empty

    # When no shocks are provided, apply_feature_shock should return a copy of the original df
    mock_apply_shock.return_value = baseline_dataframe.copy()

    expected_baseline_preds = mock_model_instance.predict(baseline_dataframe)
    # Since no shocks are applied, stressed predictions should be the same as baseline
    expected_stressed_preds = expected_baseline_preds

    results = run_single_scenario(mock_model_instance, baseline_dataframe, scenario_name, scenario_shocks, shock_type)

    mock_apply_shock.assert_called_once_with(baseline_dataframe, scenario_shocks, shock_type=shock_type)
    np.testing.assert_array_almost_equal(results['baseline_predictions'], expected_baseline_preds)
    np.testing.assert_array_almost_equal(results['stressed_predictions'], expected_stressed_preds)
    assert 'scenario_name' in results and results['scenario_name'] == scenario_name


@patch('definition_da5bd0de9fea4be597daa54697b9aac0.apply_feature_shock')
def test_invalid_shock_type(mock_apply_shock, mock_model_instance, baseline_dataframe):
    """
    Tests error handling when an unsupported shock_type is provided.
    Expects ValueError to be raised, originating from apply_feature_shock.
    """
    scenario_name = "Invalid Shock Type Scenario"
    scenario_shocks = {'feature1': 0.5}
    shock_type = 'unsupported_type'

    # Configure the mock apply_feature_shock to raise a ValueError
    mock_apply_shock.side_effect = ValueError(f"Unknown shock_type: {shock_type}")

    # Expect ValueError when run_single_scenario calls the mocked apply_feature_shock
    with pytest.raises(ValueError, match=f"Unknown shock_type: {shock_type}"):
        run_single_scenario(mock_model_instance, baseline_dataframe, scenario_name, scenario_shocks, shock_type)

    mock_apply_shock.assert_called_once_with(baseline_dataframe, scenario_shocks, shock_type=shock_type)