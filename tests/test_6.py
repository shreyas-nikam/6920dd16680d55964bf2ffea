import pytest
import numpy as np
from definition_6fce1e889a454af5b04c39451ae9defd import calculate_mean_delta

@pytest.mark.parametrize("y_stressed, y_baseline, expected", [
    # Test Case 1: Basic functionality - positive delta
    (np.array([10, 20, 30]), np.array([1, 2, 3]), 18.0), # Mean: 20.0 - 2.0 = 18.0

    # Test Case 2: Basic functionality - negative delta
    (np.array([1, 2, 3]), np.array([10, 20, 30]), -18.0), # Mean: 2.0 - 20.0 = -18.0

    # Test Case 3: Basic functionality - zero delta
    (np.array([5, 5, 5]), np.array([5, 5, 5]), 0.0), # Mean: 5.0 - 5.0 = 0.0

    # Test Case 4: Edge case - empty arrays, resulting in NaN for mean and delta
    # np.mean([]) is NaN. NaN - NaN is NaN.
    (np.array([]), np.array([]), np.nan),

    # Test Case 5: Edge case - non-numeric or incorrect type input for numpy.ndarray
    # The docstring specifies numpy.ndarray. Passing a non-array, non-numeric type (like a string)
    # to np.mean would typically result in a TypeError.
    (np.array([1, 2, 3]), "not_a_numpy_array", TypeError), 
])
def test_calculate_mean_delta(y_stressed, y_baseline, expected):
    if expected is np.nan:
        # For NaN comparisons, use np.isnan
        result = calculate_mean_delta(y_stressed, y_baseline)
        assert np.isnan(result)
    elif isinstance(expected, type) and issubclass(expected, Exception):
        # Expecting an exception
        with pytest.raises(expected):
            calculate_mean_delta(y_stressed, y_baseline)
    else:
        # For float comparisons, use pytest.approx due to potential floating-point inaccuracies
        actual_result = calculate_mean_delta(y_stressed, y_baseline)
        assert actual_result == pytest.approx(expected)