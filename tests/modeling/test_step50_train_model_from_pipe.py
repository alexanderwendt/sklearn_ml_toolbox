import joblib
import numpy as np
import pandas as pd
from sklearn_ml_toolbox.config import Config
from sklearn_ml_toolbox.modeling.training import train_model


def test_trained_pipeline_roundtrip(tmp_path):
    frame = pd.DataFrame({'x': range(30), 'y': [0, 1] * 15})
    config = Config(tmp_path/'config.toml', tmp_path/'data.csv', 'y', 'classification', tmp_path, models=('linear',), search_iterations=1)
    model = train_model(frame, config)
    joblib.dump(model, tmp_path/'model.joblib')
    loaded = joblib.load(tmp_path/'model.joblib')
    np.testing.assert_array_equal(loaded['pipeline'].predict(frame[['x']]), model['pipeline'].predict(frame[['x']]))
