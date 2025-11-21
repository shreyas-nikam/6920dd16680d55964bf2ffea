import pytest
import numpy as np
from definition_3eaa7f4c2b0642b2b99df8e6d77b86ff import calculate_expected_loss

@pytest.mark.parametrize("predictions, exposures, expected", [
    # Test case 1: Standard positive probabilities and exposures
    (np.array([0.1, 0.2, 0.3]), np.array([100, 200, 300]), 140.0),
    # Test case 2: Empty arrays, expected loss should be 0
    (np.array([]), np.array([]), 0.0),
    # Test case 3: Arrays with zeros and mixed values
    (np.array([0.0, 0.5, 1.0]), np.array([100, 0, 200]), 200.0),
    # Test case 4: Mismatched array shapes, should raise a ValueError
    (np.array([0.1, 0.2]), np.array([100, 200, 300]), ValueError),
    # Test case 5: Single element arrays
    (np.array([0.05]), np.array([1000]), 50.0),
])
def test_calculate_expected_loss(predictions, exposures, expected):
    try:
        result = calculate_expected_loss(predictions, exposures)
        assert isinstance(result, float)
        assert result == pytest.approx(expected)
    except Exception as e:
        assert isinstance(e, expected)
