"""Causal OHLC feature preparation; targets remain dataset-defined."""
import pandas as pd


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in frame:
        raise ValueError("Stock-market preparation requires Close")
    for period in (5, 20, 60):
        frame[f"return_{period}"] = frame.Close.pct_change(period)
        frame[f"mean_ratio_{period}"] = frame.Close / frame.Close.rolling(period).mean() - 1
    return frame
