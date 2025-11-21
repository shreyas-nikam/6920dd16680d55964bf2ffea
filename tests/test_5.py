import pytest
import numpy as np
from definition_bb9cf4dd8ae543e3a8a7997ca121b76b import calculate_prediction_change

@pytest.mark.parametrize("y_stressed, y_baseline, expected", [
    # Test case 1: Standard scenario with positive float values
    (np.array([0.7, 0.8, 0.6]), np.array([0.5, 0.7, 0.5]), np.array([0.2, 0.1, 0.1])),
    # Test case 2: Empty numpy arrays (edge case)
    (np.array([]), np.array([]), np.array([])),
    # Test case 3: Arrays with negative values and mixed floats (ensures correct subtraction for various numbers)
    (np.array([0.1, -0.5, 1.0]), np.array([0.3, 0.2, 0.8]), np.array([-0.2, -0.7, 0.2])),
    # Test case 4: Mismatched array shapes (error case)
    (np.array([1, 2]), np.array([1, 2, 3]), ValueError),
    # Test case 5: Non-numpy array input for y_stressed (error case)
    ([1, 2, 3], np.array([0, 1, 2]), TypeError),
])
def test_calculate_prediction_change(y_stressed, y_baseline, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        # If an exception is expected, use pytest.raises
        with pytest.raises(expected):
            calculate_prediction_change(y_stressed, y_baseline)
    else:
        # Otherwise, assert the result
        result = calculate_prediction_change(y_stressed, y_baseline)
        np.testing.assert_array_almost_equal(result, expected)