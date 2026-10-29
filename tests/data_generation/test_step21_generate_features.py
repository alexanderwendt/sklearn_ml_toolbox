import numpy as np
import pandas as pd
from samples.debug_omxs30.steps.data_generation.step21_generate_features import price_normalizer


def test_price_normalizer_rolling_window_preserves_index():
    frame = pd.DataFrame({'Close': [10., 20., 15., 25., 30., 20.]}, index=pd.date_range('2020-01-01', periods=6))
    result = price_normalizer(frame, debug_param=True)
    pd.testing.assert_index_equal(result.index, frame.index)
    np.testing.assert_allclose(result.NormKurs5, [np.nan, np.nan, np.nan, np.nan, 1., 1/3], equal_nan=True)


def test_flat_price_is_neutral():
    result = price_normalizer(pd.DataFrame({'Close': [5.] * 6}), debug_param=True)
    assert result.NormKurs5.iloc[-1] == 0.5


def test_legacy_rsi_features_have_distinct_period_names():
    from samples.debug_omxs30.steps.data_generation.step21_generate_features import get_rsi

    source = pd.DataFrame({'Close': np.arange(1., 61.)})
    result = get_rsi(source, debug_param=True)
    assert list(result.columns) == ['RSI_9', 'RSI_14']
    assert result.iloc[-1].tolist() == [100., 100.]


def test_legacy_calendar_features_remain_numeric_for_histograms():
    from samples.debug_omxs30.steps.data_generation.step21_generate_features import get_periodical_indicators

    source = pd.DataFrame({'Date': pd.bdate_range('2022-01-01', periods=90)})
    result = get_periodical_indicators(source)
    dummy_columns = [c for c in result if '__' in c]
    assert dummy_columns
    assert all(result[c].dtype == np.uint8 for c in dummy_columns)
    for prefix in ('day_week__', 'month_year__', 'week_month__'):
        assert result.filter(like=prefix).sum(axis=1).eq(1).all()
    assert np.isfinite(np.quantile(result[dummy_columns[0]], 0.75))
