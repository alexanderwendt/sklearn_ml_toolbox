import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor


def test_regression_metrics_match_known_values(tmp_path):
    from sklearn_ml_toolbox.modeling.evaluation import evaluate_model
    pipe = DummyRegressor(strategy='constant', constant=2).fit([[0], [1]], [1, 3])
    model = {'pipeline': pipe, 'baseline': pipe, 'features': ['x'], 'numeric': ['x'], 'target': 'y', 'task': 'regression'}
    result = evaluate_model(model, pd.DataFrame({'x': [0, 1], 'y': [1., 3.]}), tmp_path)
    assert result['mae'] == 1
    assert result['rmse'] == 1
    assert result['r2'] == 0
    assert (tmp_path / 'graphs' / 'residuals.png').is_file()


def test_constant_target_metric_reason(tmp_path):
    from sklearn_ml_toolbox.modeling.evaluation import evaluate_model
    pipe = DummyRegressor().fit([[0], [1]], [2, 2])
    model = {'pipeline': pipe, 'baseline': pipe, 'features': ['x'], 'numeric': ['x'], 'target': 'y', 'task': 'regression'}
    result = evaluate_model(model, pd.DataFrame({'x': [0, 1], 'y': [2., 2.]}), tmp_path)
    assert result['r2'] is None
    assert 'r2' in result['undefined']
