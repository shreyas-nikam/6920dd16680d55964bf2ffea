import pytest
import numpy as np
from definition_627421d6b7d94254b2fd873d17daa6ba import MockMLModel

# Define expected fixed values for coefficients and intercept as per specification
EXPECTED_COEFFICIENTS = np.array([-0.5, 2.0, 1.5, 1.0, 3.0])
EXPECTED_INTERCEPT = -5.0

def test_mock_ml_model_init_coefficients_existence_and_type():
    """
    Test that the 'coefficients' attribute exists and is a numpy array.
    """
    model = MockMLModel()
    assert hasattr(model, 'coefficients'), "MockMLModel should have a 'coefficients' attribute."
    assert isinstance(model.coefficients, np.ndarray), "Coefficients should be a numpy array."
    assert model.coefficients.dtype == EXPECTED_COEFFICIENTS.dtype, "Coefficients dtype should match expected."

def test_mock_ml_model_init_intercept_existence_and_type():
    """
    Test that the 'intercept' attribute exists and is a numeric type (int or float).
    """
    model = MockMLModel()
    assert hasattr(model, 'intercept'), "MockMLModel should have an 'intercept' attribute."
    assert isinstance(model.intercept, (int, float)), "Intercept should be an integer or a float."

def test_mock_ml_model_init_coefficients_value():
    """
    Test that the 'coefficients' attribute is initialized with the correct predefined values.
    """
    model = MockMLModel()
    np.testing.assert_array_equal(model.coefficients, EXPECTED_COEFFICIENTS,
                                  "Coefficients values do not match the expected predefined array.")

def test_mock_ml_model_init_intercept_value():
    """
    Test that the 'intercept' attribute is initialized with the correct predefined value.
    """
    model = MockMLModel()
    assert model.intercept == EXPECTED_INTERCEPT, "Intercept value does not match the expected predefined value."

def test_mock_ml_model_init_coefficients_shape():
    """
    Test that the 'coefficients' numpy array has the correct shape/number of elements.
    """
    model = MockMLModel()
    assert model.coefficients.shape == EXPECTED_COEFFICIENTS.shape, \
        f"Coefficients array shape mismatch. Expected {EXPECTED_COEFFICIENTS.shape}, got {model.coefficients.shape}."
    assert len(model.coefficients) == len(EXPECTED_COEFFICIENTS), \
        f"Coefficients array length mismatch. Expected {len(EXPECTED_COEFFICIENTS)}, got {len(model.coefficients)}."