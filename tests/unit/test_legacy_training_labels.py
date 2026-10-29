from configparser import ConfigParser
from types import SimpleNamespace

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn_ml_toolbox.utils import execution_utils


def test_training_scorers_use_prepared_numeric_labels(tmp_path, monkeypatch):
    features = tmp_path / 'features.csv'
    outcomes = tmp_path / 'outcomes.csv'
    labels = tmp_path / 'labels.csv'
    raw_labels = tmp_path / 'raw_labels.csv'
    pd.DataFrame({'id': [0, 1, 2, 3], 'feature': [0., 1., 0., 1.]}).to_csv(features, sep=';', index=False)
    pd.DataFrame({'id': [0, 1, 2, 3], 'class': [0, 1, 0, 1]}).to_csv(outcomes, sep=';', index=False)
    labels.write_text('Negative;0\nPositive;1\n')
    raw_labels.write_text('neutral;0\npositive;1\nnegative;2\n')
    config = ConfigParser()
    config.read_dict({
        'Common': {'problem_type': 'classification'},
        'Paths': {'labels_path': str(raw_labels)},
        'Training': {
            'features_train_in': str(features), 'features_val_in': str(features),
            'outcomes_train_in': str(outcomes), 'outcomes_val_in': str(outcomes),
            'labels_in': str(labels), 'selected_feature_columns_in': 'unused',
            'refit_scorer_name': 'f1_score', 'average_method': 'macro',
        },
    })
    monkeypatch.setattr(execution_utils, 'load_feature_columns', lambda *args: ({}, {}, None))
    monkeypatch.setattr(execution_utils, 'Paths', lambda conf: SimpleNamespace(paths={}))

    X, y, _, _, classes, _, _, _, scorers, _ = execution_utils.load_training_input_input(config)
    assert classes == {0: 'Negative', 1: 'Positive'}
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert scorers['f1_score'](model, X, y) == 1.0
    assert scorers['recall_score'](model, X, y) == 1.0
    assert scorers['precision_score'](model, X, y) == 1.0
