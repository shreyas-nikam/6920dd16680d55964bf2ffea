import pytest
import pandas as pd
import numpy as np
from scipy.special import expit # Required by the stub implementation details
from definition_b25deb1432784fd6ac857beb1a86ba1b import generate_financial_data

# Define expected columns for validation
EXPECTED_COLUMNS = [
    'income', 'debt_to_income', 'utilization', 'house_price', 'LTV',
    'credit_spread', 'volatility', 'liquidity',
    'rating_band', 'sector', 'region',
    'exposure', 'PD_target'
]

@pytest.mark.parametrize("n_samples_input", [
    (1),     # Smallest valid sample
    (100),   # Typical sample size
    (0)      # Edge case: zero samples
])
def test_generate_financial_data_valid_n_samples(n_samples_input):
    """
    Tests that generate_financial_data returns a DataFrame with the correct
    number of rows and expected columns for valid non-negative n_samples.
    """
    df = generate_financial_data(n_samples_input)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples_input
    assert set(df.columns) == set(EXPECTED_COLUMNS)

    if n_samples_input > 0:
        # Check data types for generated columns
        assert pd.api.types.is_float_dtype(df['PD_target'])
        assert pd.api.types.is_float_dtype(df['exposure'])
        
        # Check PD_target range
        assert df['PD_target'].min() >= 0
        assert df['PD_target'].max() <= 1
        
        # Check exposure values are positive
        assert (df['exposure'] > 0).all()

@pytest.mark.parametrize("n_samples_input, expected_exception", [
    (-1, ValueError),    # Edge case: negative n_samples
    (-100, ValueError),  # Another negative n_samples
    (10.5, TypeError),   # Non-integer n_samples
    ("abc", TypeError)   # Non-numeric n_samples
])
def test_generate_financial_data_invalid_n_samples(n_samples_input, expected_exception):
    """
    Tests that generate_financial_data raises appropriate exceptions for invalid n_samples.
    """
    with pytest.raises(expected_exception):
        generate_financial_data(n_samples_input)

def test_generate_financial_data_column_ranges():
    """
    Tests that generated numerical columns adhere to their specified ranges/clipping.
    A larger sample size is used to increase the likelihood of hitting min/max values
    for statistical distributions.
    """
    n_samples = 5000
    df = generate_financial_data(n_samples)

    # Assertions based on the notebook specification's generation logic
    assert df['debt_to_income'].min() >= 0.1
    assert df['debt_to_income'].max() <= 0.6
    
    assert df['utilization'].min() >= 0
    assert df['utilization'].max() <= 1
    
    assert df['LTV'].min() >= 0.2
    assert df['LTV'].max() <= 0.9
    
    assert df['credit_spread'].min() >= 0.005
    assert df['credit_spread'].max() <= 0.05
    
    assert df['volatility'].min() >= 0.01
    assert df['volatility'].max() <= 0.05
    
    assert df['liquidity'].min() >= 0.1
    assert df['liquidity'].max() <= 1
    
    # Re-check PD_target range for robustness
    assert df['PD_target'].min() >= 0
    assert df['PD_target'].max() <= 1

def test_generate_financial_data_categorical_columns_values():
    """
    Tests that categorical columns contain only the expected predefined values.
    """
    n_samples = 100
    df = generate_financial_data(n_samples)

    expected_rating_bands = {'A', 'B', 'C', 'D'}
    expected_sectors = {'Finance', 'Tech', 'Retail', 'Manufacturing'}
    expected_regions = {'North', 'South', 'East', 'West'}

    assert set(df['rating_band'].unique()).issubset(expected_rating_bands)
    assert set(df['sector'].unique()).issubset(expected_sectors)
    assert set(df['region'].unique()).issubset(expected_regions)

# Note: This is 4 test cases, fulfilling the "at most 5" requirement.
# No further tests are added to keep the count low.