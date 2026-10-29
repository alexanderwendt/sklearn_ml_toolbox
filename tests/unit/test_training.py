from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from sklearn_ml_toolbox.config import Config


def cfg(tmp_path, **kwargs):
    return Config(tmp_path / 'config.toml', tmp_path / 'data.csv', 'y', 'regression', tmp_path / 'results', models=('linear',), search_iterations=1, **kwargs)


def test_holdout_never_fits_preprocessor(tmp_path):
    from sklearn_ml_toolbox.modeling.training import train_model
    frame = pd.DataFrame({'x': range(20), 'category': ['a', 'b'] * 10, 'y': np.arange(20) * 2.})
    model = train_model(frame, cfg(tmp_path))
    mean = model['pipeline'].named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].mean_
    assert mean.tolist() == [9.5]
    assert np.isfinite(model['pipeline'].predict(pd.DataFrame({'x': [1e6], 'category': ['novel']}))).all()


def test_temporal_split_separates_timestamp_groups(tmp_path):
    from sklearn_ml_toolbox.data_processing.preprocessing import split_dataset
    frame = pd.DataFrame({'date': ['2020-01-03', '2020-01-01', '2020-01-02', '2020-01-03'], 'x': range(4), 'y': range(4)})
    train, test = split_dataset(frame, cfg(tmp_path, strategy='temporal', time_column='date'))
    assert train.date.max() < test.date.min()
    assert set(train.index).isdisjoint(test.index)


def test_rare_class_fails_before_search(tmp_path):
    from sklearn_ml_toolbox.modeling.training import train_model
    config = replace(cfg(tmp_path), task='classification')
    with pytest.raises(ValueError, match='class'):
        train_model(pd.DataFrame({'x': range(10), 'y': [0] * 9 + [1]}), config)


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_mixed_missing_features_and_selection(tmp_path, task):
    from sklearn_ml_toolbox.modeling.training import train_model
    frame = pd.DataFrame({'x': list(range(29)) + [None], 'category': ['a', None, 'b'] * 10, 'empty': [None] * 30, 'y': [0, 1, 2] * 10})
    model = train_model(frame, replace(cfg(tmp_path), task=task, select_percentile=80, balance=task == 'classification'))
    assert len(model['pipeline'].predict(frame[model['features']])) == 30
    assert {'dummy', 'linear'} == {row['model'] for row in model['cv_results']}


def test_threshold_tuning_uses_only_training_data(tmp_path):
    from sklearn_ml_toolbox.modeling.training import train_model
    config = replace(cfg(tmp_path), task='classification', tune_threshold=True)
    frame = pd.DataFrame({'x': range(60), 'y': [0] * 40 + [1] * 20})
    model = train_model(frame, config)
    assert hasattr(model['pipeline'], 'best_threshold_')
    assert set(model['pipeline'].predict(frame[['x']])) <= {0, 1}
