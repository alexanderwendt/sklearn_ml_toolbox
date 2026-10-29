"""Predict with the complete fitted pipeline; model files must be trusted."""
from pathlib import Path

import joblib
import pandas as pd

from ..config import Config
from ..data_generation.loader import prepare_dataset
from ..data_processing.preprocessing import feature_frame


def predict_csv(model_path: Path, input_path: Path, output_path: Path, config: Config | None = None) -> Path:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")
    model = joblib.load(model_path)
    frame = prepare_dataset(config, require_target=False, input_path=input_path) if config else pd.read_csv(input_path)
    if not config and "_row_id" in frame:
        if frame['_row_id'].isna().any() or not frame['_row_id'].is_unique:
            raise ValueError("Prediction row identity must be nonmissing and unique")
        frame = frame.set_index('_row_id')
    X = feature_frame(frame, model["features"], model["numeric"])
    prediction = model["pipeline"].predict(X)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predicted": prediction}, index=frame.index).to_csv(output_path, index_label="_row_id")
    return output_path
