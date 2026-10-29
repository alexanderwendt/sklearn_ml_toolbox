import numpy as np
import pandas as pd
import pytest


def test_step35_supports_multiclass_with_current_sklearn():
    from sklearn_ml_toolbox.data_processing.step35_perform_feature_selection import (
        predict_features_simple, execute_recursive_elimination_feature_selection)
    rng = np.random.default_rng(42)
    y = np.tile([0, 1, 2], 20)
    X = pd.DataFrame({'signal': y + rng.normal(0, .1, len(y)),
                      'noise': rng.normal(size=len(y))})
    assert predict_features_simple(X, y) > .9
    selected, _ = execute_recursive_elimination_feature_selection(X, y)
    assert list(selected) == ['signal']


def test_class_report_handles_integer_model_classes_and_float_targets(tmp_path):
    from sklearn.dummy import DummyClassifier
    from sklearn_ml_toolbox.modeling.evaluation import evaluate_model
    X = pd.DataFrame({'x': range(9)})
    pipe = DummyClassifier().fit(X, [0, 1, 2] * 3)
    model = {'pipeline': pipe, 'baseline': pipe, 'features': ['x'], 'numeric': ['x'],
             'task': 'classification', 'target': 'y'}
    result = evaluate_model(model, X.assign(y=np.tile([0., 1., 2.], 3)), tmp_path)
    assert result['classification_report']['0']['support'] == 3
    assert (tmp_path / 'graphs/class_f1-score.png').is_file()


@pytest.mark.parametrize('name', ['gnb', 'knn', 'svm_linear'])
def test_additional_classifiers_handle_categorical_features(tmp_path, name):
    from sklearn_ml_toolbox.config import Config
    from sklearn_ml_toolbox.modeling.training import train_model
    frame = pd.DataFrame({'x': np.arange(60), 'category': [f'c{i % 20}' for i in range(60)],
                          'y': np.tile([0, 1, 2], 20)})
    config = Config(tmp_path / 'config.toml', tmp_path / 'input.csv', 'y', 'classification',
                    tmp_path, models=(name,), folds=2, search_iterations=1)
    model = train_model(frame, config)
    assert name in model['candidates']
    assert len(model['candidates'][name].predict(frame[model['features']])) == 60


def test_parallel_coordinates_supports_five_classes(tmp_path):
    import seaborn as sns
    from sklearn_ml_toolbox.data_processing.step33_analyze_data import plot_parallel_coordinates
    frame = pd.DataFrame({'a': range(5), 'b': range(5, 10), 'class': range(5)})
    plot_parallel_coordinates(frame, ['a', 'b'], sns.color_palette('husl', 5), 'class', None, str(tmp_path))
    assert (tmp_path / 'Parallel_Coordinates.png').is_file()
