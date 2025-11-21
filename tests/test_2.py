import pytest
import pandas as pd
import numpy as np
from scipy.special import expit # As per notebook specification

# Coefficients and intercept from notebook spec section 3.4 for MockMLModel
_mock_model_coeffs = np.array([-0.5, 2.0, 1.5, 1.0, 3.0])
_mock_model_intercept = -5.0
_mock_model_features = ['income', 'debt_to_income', 'utilization', 'LTV', 'credit_spread']

def _calculate_expected_predictions(df: pd.DataFrame) -> np.ndarray:
    """
    Helper function to calculate expected predictions based on MockMLModel's logic.
    Assumes `df` contains all `_mock_model_features` and they are numeric.
    """
    df_processed = df[_mock_model_features]
    linear_output = np.dot(df_processed, _mock_model_coeffs) + _mock_model_intercept
    return expit(linear_output)

# --- Start of definition_c28fd19bd6954fc29f3a84c7adbb21d2 block ---
# NOTE: DO NOT REPLACE or REMOVE THIS BLOCK.
# It is used for automated testing.
# Assume the MockMLModel class is available in this module.
# For example:
# class MockMLModel:
#     def __init__(self):
#         self.coefficients = np.array([-0.5, 2.0, 1.5, 1.0, 3.0])
#         self.intercept = -5.0
#         self.expected_features = ['income', 'debt_to_income', 'utilization', 'LTV', 'credit_spread']
#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         # ... implementation ...
#         pass
# --- End of definition_c28fd19bd6954fc29f3a84c7adbb21d2 block ---

# Import the MockMLModel from the placeholder module for testing
from definition_c28fd19bd6954fc29f3a84c7adbb21d2 import MockMLModel

@pytest.fixture
def mock_model():
    """Fixture to provide an instance of MockMLModel."""
    return MockMLModel()

@pytest.mark.parametrize(
    "test_input_X, expected_exception",
    [
        # Test Case 1: Standard functionality with multiple rows
        # Covers expected output type, shape, value range [0, 1], and correct calculation.
        (
            pd.DataFrame({
                'income': [1.0, 2.0, 3.0],
                'debt_to_income': [0.1, 0.2, 0.3],
                'utilization': [0.5, 0.6, 0.7],
                'LTV': [0.6, 0.7, 0.8],
                'credit_spread': [0.01, 0.02, 0.03],
                'unrelated_col': [10.0, 20.0, 30.0] # Extra column, should be ignored
            }),
            None # No exception expected
        ),
        # Test Case 2: Empty DataFrame input
        # Covers edge case where DataFrame has no columns, leading to missing features.
        (
            pd.DataFrame(),
            ValueError # Expected: ValueError because required features are missing.
        ),
        # Test Case 3: DataFrame with missing required features
        # Covers edge case where input DataFrame lacks some critical features.
        (
            pd.DataFrame({
                'income': [1.0], 'debt_to_income': [0.1], 'utilization': [0.5]
                # 'LTV' and 'credit_spread' are intentionally missing
            }),
            ValueError # Expected: ValueError because specific required features are missing.
        ),
        # Test Case 4: Non-DataFrame input type
        # Covers robustness against incorrect input types.
        (
            [1, 2, 3], # A list instead of a pandas.DataFrame
            TypeError # Expected: TypeError as input 'X' must be a DataFrame.
        ),
        # Test Case 5: DataFrame with non-numerical data in an expected numerical feature column
        # Covers data type validation within the selected features.
        (
            pd.DataFrame({
                'income': [1.0], 'debt_to_income': ['invalid_str'], # 'debt_to_income' is non-numeric
                'utilization': [0.5], 'LTV': [0.6], 'credit_spread': [0.01]
            }),
            TypeError # Expected: TypeError because a required feature column contains non-numerical data.
        ),
    ]
)
def test_predict_functionality(mock_model, test_input_X, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            mock_model.predict(test_input_X)
    else:
        actual_output = mock_model.predict(test_input_X)
        expected_output = _calculate_expected_predictions(test_input_X)

        # Assertions for correct functionality
        assert isinstance(actual_output, np.ndarray)
        assert actual_output.shape == expected_output.shape
        # Use np.allclose for floating-point comparisons
        assert np.allclose(actual_output, expected_output, atol=1e-6)
        # Assert that all predictions are probabilities between 0 and 1
        assert np.all((actual_output >= 0) & (actual_output <= 1))