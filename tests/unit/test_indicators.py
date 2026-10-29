import numpy as np
import pandas as pd


def test_indicators_causal_and_named():
    from sklearn_ml_toolbox.data_generation import indicators as ta
    close = pd.Series(np.arange(1., 61.))
    assert ta.sma(close, length=3).iloc[2] == 2
    assert ta.rsi(close, length=5).iloc[-1] == 100
    assert set(ta.macd(close).columns) == {'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'}
    prefix = ta.macd(close.iloc[:45])
    pd.testing.assert_frame_equal(prefix, ta.macd(close).iloc[:45])


def test_stock_features_do_not_use_future_rows():
    from sklearn_ml_toolbox.data_generation.stockmarket import prepare
    frame = pd.DataFrame({'Close': np.arange(1., 101.)})
    pd.testing.assert_frame_equal(prepare(frame.iloc[:80].copy()), prepare(frame.copy()).iloc[:80])
