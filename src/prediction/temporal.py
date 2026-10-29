"""Transparent long/flat close-to-close research backtest.

Signal at close t becomes the position for return t→t+1. Costs are per
unit turnover; no leverage, borrowing, slippage or intrabar execution model.
"""
import numpy as np
import pandas as pd


def backtest(frame: pd.DataFrame, cost: float = 0.001) -> pd.DataFrame:
    if not 0 <= cost < 1:
        raise ValueError("cost must be between 0 and 1")
    if not {'Close', 'signal'}.issubset(frame):
        raise ValueError("Backtest needs Close and signal columns")
    close = pd.to_numeric(frame.Close, errors='raise')
    if close.isna().any() or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Prices must be finite and positive")
    if not frame.signal.isin([0, 1]).all():
        raise ValueError("signal must be 0 (flat) or 1 (long)")
    result = frame.copy()
    result['position'] = frame.signal.shift(1, fill_value=0)
    turnover = result.position.diff().fillna(result.position).abs()
    result['strategy_return'] = result.position * close.pct_change().fillna(0) - turnover * cost
    result['equity'] = (1 + result.strategy_return).cumprod()
    result['buy_hold'] = close / close.iloc[0]
    return result
