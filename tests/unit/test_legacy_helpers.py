import numpy as np
import pandas as pd
import pytest


def test_trim_both_ends_and_no_trim():
    from sklearn_ml_toolbox.data_generation.step22_adapt_dimensions import cut_unusable_parts_of_dataframe
    frame = pd.DataFrame({'x': range(6)})
    assert cut_unusable_parts_of_dataframe(frame, 1, 2).x.tolist() == [1, 2, 3]
    pd.testing.assert_frame_equal(cut_unusable_parts_of_dataframe(frame), frame)


def test_clean_nan_preserves_nondefault_index():
    from sklearn_ml_toolbox.data_generation.step22_adapt_dimensions import clean_nan
    frame = pd.DataFrame({'x': [1, np.nan, 3]}, index=['a', 'b', 'c'])
    assert clean_nan(frame).index.tolist() == ['a', 'c']


def test_legacy_precision_recall_graphs(tmp_path):
    from sklearn_ml_toolbox.utils.data_visualization_functions import plot_precision_recall_evaluation
    plot_precision_recall_evaluation(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([[.9,.1],[.2,.8],[.7,.3],[.1,.9]]), {0:'no',1:'yes'}, tmp_path)
    assert len(list(tmp_path.glob('*.png'))) >= 3


def test_event_forward_fill_and_reference_signal():
    from sklearn_ml_toolbox.utils.stock_market_utils import calculateLatestEvent
    from sklearn_ml_toolbox.prediction.step71_value_postprocessing import generate_reference_results
    np.testing.assert_array_equal(calculateLatestEvent(np.array([0, 2, 0, 1, 0])), [0, 2, 2, 1, 1])
    np.testing.assert_array_equal(generate_reference_results(pd.Series([-1, 0, 1])), [0, 0, 1])
