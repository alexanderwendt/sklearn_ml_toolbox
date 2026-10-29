"""Split before fitting any data-dependent transformation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import Config


def split_dataset(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.strategy == "temporal":
        if config.time_column not in frame:
            raise ValueError(f"Missing time column: {config.time_column}")
        dates = pd.to_datetime(frame[config.time_column], errors="raise")
        if dates.isna().any():
            raise ValueError("Missing timestamps")
        frame = frame.loc[dates.sort_values(kind="stable").index]
        dates = dates.loc[frame.index]
        cut = max(1, int(len(frame) * (1 - config.test_size)))
        boundary = dates.iloc[min(cut, len(dates) - 1)]
        train, test = frame.loc[dates < boundary], frame.loc[dates >= boundary]
        if train.empty or test.empty:
            raise ValueError("Temporal split needs at least two distinct timestamp groups")
        return train, test
    stratify = frame[config.target] if config.task == "classification" else None
    try:
        return tuple(train_test_split(frame, test_size=config.test_size, random_state=config.seed, stratify=stratify))
    except ValueError as error:
        raise ValueError(f"Cannot split dataset; check row/class counts: {error}") from error


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical),
    ])


def feature_frame(frame: pd.DataFrame, columns: list[str], numeric: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    result = frame[columns].copy()
    for column in columns:
        if column in numeric:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
            if np.isinf(result[column]).any():
                raise ValueError(f"Infinite feature values: {column}")
        else:
            result[column] = result[column].map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
    return result
