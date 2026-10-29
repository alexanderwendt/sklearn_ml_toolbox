"""Causal pandas indicators, replacing the obsolete pandas-ta dependency.

EMA uses an initial SMA seed; RSI uses Wilder's exponential average.
Warmup values stay missing. Flat RSI is neutral (50).
"""
import numpy as np
import pandas as pd


def sma(close, length=10, offset=0):
    return close.rolling(length).mean().shift(offset)


def ema(close, length):
    values = close.copy().astype(float)
    valid = values.first_valid_index()
    if valid is None:
        return values
    start = values.index.get_loc(valid)
    end = start + length - 1
    if end >= len(values):
        return values * np.nan
    seed = values.iloc[start:end + 1].mean()
    values.iloc[:end] = np.nan
    values.iloc[end] = seed
    return values.ewm(span=length, adjust=False).mean()


def rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, min_periods=length).mean()
    total = gain + loss
    return (100 * gain / total).mask(total == 0, 50).rename(f"RSI_{length}")


def stoch(high, low, close, k=14, d=3, smooth_k=3):
    bottom = low.rolling(k).min()
    spread = high.rolling(k).max() - bottom
    raw = (100 * (close - bottom) / spread).mask(spread == 0, 50)
    fast = raw.rolling(smooth_k).mean()
    return pd.DataFrame({f"STOCHk_{k}_{d}_{smooth_k}": fast, f"STOCHd_{k}_{d}_{smooth_k}": fast.rolling(d).mean()})


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    suffix = f"{fast}_{slow}_{signal}"
    return pd.DataFrame({f"MACD_{suffix}": line, f"MACDh_{suffix}": line - signal_line, f"MACDs_{suffix}": signal_line})
