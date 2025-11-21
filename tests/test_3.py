import pytest
import pandas as pd
import numpy as np

# Keep the definition_e89ff896f3ab46178832cbc1e4e70228 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_e89ff896f3ab46178832cbc1e4e70228 import apply_feature_shock

@pytest.fixture
def sample_df():
    """Provides a sample DataFrame for testing."""
    data = {'feature1': [10.0, 20.0, 30.0],
            'feature2': [1.0, 2.0, 3.0],
            'other_feature': [100, 200, 300]}
    return pd.DataFrame(data)

def test_multiplicative_shock(sample_df):
    """
    Tests basic multiplicative shock application.
    Features should be scaled by the given factors, and other features should remain unchanged.
    """
    shocks = {'feature1': 1.1, 'feature2': 0.5} # e.g., 10% increase, 50% decrease
    shock_type = 'multiplicative'
    
    result_df = apply_feature_shock(sample_df, shocks, shock_type)

    expected_feature1 = sample_df['feature1'] * 1.1
    expected_feature2 = sample_df['feature2'] * 0.5

    pd.testing.assert_series_equal(result_df['feature1'], expected_feature1, check_dtype=False)
    pd.testing.assert_series_equal(result_df['feature2'], expected_feature2, check_dtype=False)
    # Ensure other features are untouched
    pd.testing.assert_series_equal(result_df['other_feature'], sample_df['other_feature'], check_dtype=False)
    # Ensure the original DataFrame is not modified
    pd.testing.assert_frame_equal(sample_df, pd.DataFrame({
        'feature1': [10.0, 20.0, 30.0],
        'feature2': [1.0, 2.0, 3.0],
        'other_feature': [100, 200, 300]
    }))
    assert id(result_df) != id(sample_df) # Check it's a new DataFrame

def test_additive_shock(sample_df):
    """
    Tests basic additive shock application.
    Features should be shifted by the given values, and other features should remain unchanged.
    """
    shocks = {'feature1': 5.0, 'feature2': -0.5} # e.g., add 5, subtract 0.5
    shock_type = 'additive'
    
    result_df = apply_feature_shock(sample_df, shocks, shock_type)

    expected_feature1 = sample_df['feature1'] + 5.0
    expected_feature2 = sample_df['feature2'] - 0.5

    pd.testing.assert_series_equal(result_df['feature1'], expected_feature1, check_dtype=False)
    pd.testing.assert_series_equal(result_df['feature2'], expected_feature2, check_dtype=False)
    # Ensure other features are untouched
    pd.testing.assert_series_equal(result_df['other_feature'], sample_df['other_feature'], check_dtype=False)

def test_empty_dataframe_input():
    """
    Tests the function with an empty DataFrame as input.
    Should return an empty DataFrame without errors.
    """
    empty_df = pd.DataFrame(columns=['feature1', 'feature2'])
    shocks = {'feature1': 1.1}
    shock_type = 'multiplicative'
    
    result_df = apply_feature_shock(empty_df, shocks, shock_type)

    pd.testing.assert_frame_equal(result_df, empty_df)
    assert result_df.empty

def test_shock_non_existent_feature_raises_key_error(sample_df):
    """
    Tests that attempting to shock a non-existent feature raises a KeyError,
    as per typical pandas behavior when modifying a column that doesn't exist.
    """
    shocks = {'feature1': 1.1, 'non_existent_feature': 1.2}
    shock_type = 'multiplicative'
    
    with pytest.raises(KeyError, match="non_existent_feature"):
        apply_feature_shock(sample_df, shocks, shock_type)

def test_invalid_shock_type_raises_value_error(sample_df):
    """
    Tests that providing an invalid shock_type string raises a ValueError.
    """
    shocks = {'feature1': 1.1}
    shock_type = 'invalid_type'
    
    with pytest.raises(ValueError, match="Invalid shock_type"):
        apply_feature_shock(sample_df, shocks, shock_type)