"""Load CSV and invoke an explicitly configured, trusted preparation script."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from ..config import Config


def prepare_dataset(config: Config, *, require_target: bool = True, input_path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(input_path or config.input, sep=config.delimiter)
    identity = "__toolbox_row_id__"
    if {"_row_id", identity}.intersection(frame.columns):
        raise ValueError("_row_id and __toolbox_row_id__ are reserved for row identity")
    frame.index.name = "_row_id"
    if config.strategy == "temporal":
        if config.time_column not in frame:
            raise ValueError(f"Missing time column: {config.time_column}")
        dates = pd.to_datetime(frame[config.time_column], errors="raise")
        if dates.isna().any():
            raise ValueError("Missing timestamps")
        # Rolling preparation must see chronology, not arbitrary CSV ordering.
        frame = frame.loc[dates.sort_values(kind="stable").index]
    if config.prepare:
        if config.prepare.startswith("builtin:"):
            from . import grades, stockmarket
            hooks = {"builtin:grades": grades.prepare, "builtin:stockmarket": stockmarket.prepare}
            if config.prepare not in hooks:
                raise ValueError(f"Unknown preparation hook: {config.prepare}")
            prepare = hooks[config.prepare]
        else:
            prepare = runpy.run_path(config.prepare).get("prepare")
        if not callable(prepare):
            raise ValueError("Preparation script must define prepare(frame)")
        original_ids = frame.index.copy()
        frame[identity] = frame.index
        frame = prepare(frame.copy())
        if (not isinstance(frame, pd.DataFrame) or identity not in frame or not frame.index.is_unique
                or not frame.index.isin(original_ids).all() or not np.array_equal(frame.index, frame[identity])):
            raise ValueError("Preparation hook must preserve unique row identity")
        frame = frame.drop(columns=[identity])
    frame = frame.drop(columns=list(config.drop), errors="ignore")
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Prepared data must have rows and unique columns")
    if require_target:
        if config.target not in frame or frame[config.target].isna().any():
            raise ValueError(f"Missing target column/values: {config.target}")
        if config.task == "regression" and not pd.api.types.is_numeric_dtype(frame[config.target]):
            raise ValueError("Regression target must be numeric")
    if np.isinf(frame.select_dtypes(include="number").to_numpy(dtype=float)).any():
        raise ValueError("Data contains infinite values")
    frame.index.name = "_row_id"
    return frame
