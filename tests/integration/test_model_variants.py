from dataclasses import replace
import pandas as pd
import pytest

from sklearn_ml_toolbox.config import Config
from sklearn_ml_toolbox.modeling.training import train_model


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('model_name', ['svm', 'xgboost'])
def test_optional_model_variant_fits_and_predicts(tmp_path, task, model_name):
    config = Config(tmp_path/'config.toml', tmp_path/'data.csv', 'y', task, tmp_path, models=(model_name,), search_iterations=1)
    frame = pd.DataFrame({'x': range(40), 'y': [0]*20+[1]*20 if task == 'classification' else range(40)})
    model = train_model(frame, config)
    assert len(model['pipeline'].predict(frame[['x']])) == 40
    assert model_name in {item['model'] for item in model['cv_results']}


def test_temporal_training_uses_ordered_groups(tmp_path):
    config = Config(tmp_path/'config.toml', tmp_path/'data.csv', 'y', 'regression', tmp_path, models=('linear',), search_iterations=1, strategy='temporal', time_column='Date')
    frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=60), 'x': range(60), 'y': [2*i for i in range(60)]})
    model = train_model(frame, config)
    assert model['features'] == ['x']
    assert model['selected_model'] == 'linear'
    with pytest.raises(ValueError, match='sorted'):
        train_model(frame.iloc[::-1], config)
