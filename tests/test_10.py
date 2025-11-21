import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# This block will be kept as is. DO NOT REPLACE or REMOVE.
from definition_8dd48f50b9af40d0a79732efc1729b9a import run_multiple_scenarios, apply_feature_shock 
# End of definition_8dd48f50b9af40d0a79732efc1729b9a block

@pytest.fixture
def mock_model():
    """A mock ML model with a predict method."""
    model = MagicMock()
    # Configure predict to return an array of ones with the same number of rows as input.
    # Returns an empty array if input DataFrame is empty.
    model.predict.side_effect = lambda X: np.ones(len(X)) * 0.5 if not X.empty else np.array([])
    return model

@pytest.fixture
def X_baseline():
    """A sample baseline pandas DataFrame."""
    return pd.DataFrame({
        'income': [1000, 2000, 3000],
        'debt_to_income': [0.1, 0.2, 0.3],
        'utilization': [0.4, 0.5, 0.6]
    })

# Test Case 1: Happy Path - Multiple scenarios with different shock types and valid data.
def test_run_multiple_scenarios_happy_path(mocker, mock_model, X_baseline):
    # Mock apply_feature_shock to simply return a copy of the input DataFrame.
    # This simplifies the test focus to run_multiple_scenarios' orchestration.
    mocker.patch(f"{run_multiple_scenarios.__module__}.apply_feature_shock", side_effect=lambda df, shocks, shock_type: df.copy())

    scenarios_config = {
        'Scenario Alpha': {
            'shock_type': 'multiplicative',
            'shocks': {'income': 0.9, 'utilization': 1.1}
        },
        'Scenario Beta': {
            'shock_type': 'additive',
            'shocks': {'debt_to_income': 0.05}
        }
    }

    results = run_multiple_scenarios(mock_model, X_baseline, scenarios_config)

    assert isinstance(results, dict)
    assert 'Scenario Alpha' in results
    assert 'Scenario Beta' in results
    assert len(results) == 2

    # Verify results for Scenario Alpha
    assert isinstance(results['Scenario Alpha'], np.ndarray)
    assert results['Scenario Alpha'].shape == (len(X_baseline),)
    assert np.all(results['Scenario Alpha'] == 0.5) # Based on mock_model's predict side_effect

    # Verify results for Scenario Beta
    assert isinstance(results['Scenario Beta'], np.ndarray)
    assert results['Scenario Beta'].shape == (len(X_baseline),)
    assert np.all(results['Scenario Beta'] == 0.5)

    assert mock_model.predict.call_count == 2
    assert apply_feature_shock.call_count == 2 # ensure apply_feature_shock was called for each scenario

# Test Case 2: Edge Case - Empty scenarios_config dictionary.
def test_run_multiple_scenarios_empty_config(mocker, mock_model, X_baseline):
    mocker.patch(f"{run_multiple_scenarios.__module__}.apply_feature_shock") 

    scenarios_config = {}
    results = run_multiple_scenarios(mock_model, X_baseline, scenarios_config)

    assert isinstance(results, dict)
    assert len(results) == 0
    assert mock_model.predict.call_count == 0
    assert apply_feature_shock.call_count == 0

# Test Case 3: Edge Case - Empty X_baseline DataFrame.
def test_run_multiple_scenarios_empty_X_baseline(mocker, mock_model):
    mocker.patch(f"{run_multiple_scenarios.__module__}.apply_feature_shock", side_effect=lambda df, shocks, shock_type: df.copy())

    empty_X_baseline = pd.DataFrame(columns=['income', 'debt_to_income'])
    scenarios_config = {
        'Scenario Gamma': {
            'shock_type': 'multiplicative',
            'shocks': {'income': 0.8}
        }
    }

    results = run_multiple_scenarios(mock_model, empty_X_baseline, scenarios_config)

    assert isinstance(results, dict)
    assert 'Scenario Gamma' in results
    assert len(results) == 1
    assert isinstance(results['Scenario Gamma'], np.ndarray)
    assert results['Scenario Gamma'].shape == (0,) # Expect an empty numpy array of predictions
    assert mock_model.predict.call_count == 1
    assert apply_feature_shock.call_count == 1

# Test Case 4: Error Case - Invalid model object (missing 'predict' method).
def test_run_multiple_scenarios_invalid_model(mocker, X_baseline):
    mocker.patch(f"{run_multiple_scenarios.__module__}.apply_feature_shock")

    class InvalidModel:
        pass # This model lacks a 'predict' method

    invalid_model_instance = InvalidModel()
    scenarios_config = {
        'Scenario Delta': {
            'shock_type': 'multiplicative',
            'shocks': {'income': 1.1}
        }
    }

    with pytest.raises(AttributeError, match="object has no attribute 'predict'"):
        run_multiple_scenarios(invalid_model_instance, X_baseline, scenarios_config)
    
    # apply_feature_shock would likely be called before the AttributeError is raised,
    # depending on the internal implementation of run_multiple_scenarios.
    # However, for this test, verifying the AttributeError is sufficient.

# Test Case 5: Error Case - Malformed scenarios_config (missing 'shock_type' or 'shocks' key).
def test_run_multiple_scenarios_malformed_config(mocker, mock_model, X_baseline):
    # Mock apply_feature_shock, but it should not be called if the config is malformed early.
    mocker.patch(f"{run_multiple_scenarios.__module__}.apply_feature_shock") 

    # Scenario missing 'shocks' key
    scenarios_config_missing_shocks = {
        'Scenario Epsilon': {
            'shock_type': 'multiplicative',
            # 'shocks': {'income': 0.9} # This key is intentionally missing
        }
    }
    with pytest.raises(KeyError):
        run_multiple_scenarios(mock_model, X_baseline, scenarios_config_missing_shocks)
    
    # Scenario missing 'shock_type' key
    scenarios_config_missing_type = {
        'Scenario Zeta': {
            # 'shock_type': 'multiplicative', # This key is intentionally missing
            'shocks': {'income': 0.9}
        }
    }
    with pytest.raises(KeyError):
        run_multiple_scenarios(mock_model, X_baseline, scenarios_config_missing_type)

    assert apply_feature_shock.call_count == 0 # No calls to apply_feature_shock if config is bad
    assert mock_model.predict.call_count == 0 # No calls to predict either