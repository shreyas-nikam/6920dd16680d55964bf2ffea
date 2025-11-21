import pytest
import pandas as pd
import numpy as np
from definition_8d48ea4e1d414d0ebcd41aaa10de6b0b import aggregate_by_segment

def create_test_df():
    data = {
        'segment': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A'],
        'value': [10, 20, 15, 30, 25, 12, 35, 18],
        'other_col': [1, 2, 3, 4, 5, 6, 7, 8]
    }
    return pd.DataFrame(data)

def test_aggregate_by_segment_mean_aggregation():
    df = create_test_df()
    segment_col = 'segment'
    metric_col = 'value'
    agg_func = np.mean
    
    expected_output = pd.Series({'A': (10+15+12+18)/4, 'B': (20+25)/2, 'C': (30+35)/2}, name='value')
    
    result = aggregate_by_segment(df, segment_col, metric_col, agg_func)
    
    pd.testing.assert_series_equal(result.sort_index(), expected_output.sort_index(), check_dtype=True)

def test_aggregate_by_segment_sum_aggregation():
    df = create_test_df()
    segment_col = 'segment'
    metric_col = 'value'
    agg_func = np.sum
    
    expected_output = pd.Series({'A': 10+15+12+18, 'B': 20+25, 'C': 30+35}, name='value')
    
    result = aggregate_by_segment(df, segment_col, metric_col, agg_func)
    
    pd.testing.assert_series_equal(result.sort_index(), expected_output.sort_index(), check_dtype=True)

def test_aggregate_by_segment_empty_dataframe():
    df = pd.DataFrame(columns=['segment', 'value'])
    segment_col = 'segment'
    metric_col = 'value'
    agg_func = np.mean
    
    expected_output = pd.Series(dtype=float, name='value') 
    
    result = aggregate_by_segment(df, segment_col, metric_col, agg_func)
    
    pd.testing.assert_series_equal(result, expected_output, check_dtype=True)

def test_aggregate_by_segment_single_segment_data():
    df = pd.DataFrame({
        'segment': ['X', 'X', 'X', 'X'],
        'value': [5, 10, 15, 20]
    })
    segment_col = 'segment'
    metric_col = 'value'
    agg_func = np.sum
    
    expected_output = pd.Series({'X': 5+10+15+20}, name='value')
    
    result = aggregate_by_segment(df, segment_col, metric_col, agg_func)
    
    pd.testing.assert_series_equal(result, expected_output, check_dtype=True)

@pytest.mark.parametrize("segment_col_name, metric_col_name", [
    ("non_existent_segment", "value"),
    ("segment", "non_existent_metric"),
])
def test_aggregate_by_segment_missing_columns(segment_col_name, metric_col_name):
    df = create_test_df()
    agg_func = np.mean
    
    with pytest.raises(KeyError):
        aggregate_by_segment(df, segment_col_name, metric_col_name, agg_func)