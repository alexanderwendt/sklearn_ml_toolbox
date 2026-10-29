import json

import numpy as np
import pandas as pd
import pytest

from sklearn_ml_toolbox.workflow import run_pipeline


@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_report_covers_all_features_models_and_splits(tmp_path, task):
    # Losing a candidate or mixing train/holdout predictions must fail this test.
    x = np.arange(60)
    pd.DataFrame({'x': x, 'category': ['a', 'b', None] * 20,
                  'empty': [None] * 60,
                  'y': x * 2. if task == 'regression' else x % 3}).to_csv(tmp_path / 'data.csv', index=False)
    config = tmp_path / 'config.toml'
    config.write_text(f'''[data]
path = "data.csv"
target = "y"
task = "{task}"
[training]
models = ["linear"]
folds = 2
search_iterations = 1
[output]
directory = "results"
''')
    run = run_pipeline(config)
    table = pd.read_csv(run / 'model_metrics.csv')
    assert set(zip(table.model, table.split)) == {
        ('dummy', 'train'), ('dummy', 'holdout'), ('linear', 'train'), ('linear', 'holdout')}
    for name in ('dummy', 'linear'):
        train = pd.read_csv(run / 'models' / name / 'train' / 'predictions.csv')
        holdout = pd.read_csv(run / 'models' / name / 'holdout' / 'predictions.csv')
        assert set(train._row_id).isdisjoint(holdout._row_id)
        assert len(train) + len(holdout) == 60
        importance = pd.read_csv(run / 'models' / name / 'feature_importance.csv')
        assert set(importance.feature) == {'x', 'category', 'empty'}
    features = pd.read_csv(run / 'feature_plots.csv')
    assert set(features.feature) == {'x', 'category', 'empty'}
    report = (run / 'report.html').read_text(encoding='utf-8')
    for path in run.rglob('*.png'):
        assert path.relative_to(run).as_posix() in report
        assert path.stat().st_size > 100
    if task == 'classification':
        metrics = json.loads((run / 'models/linear/holdout/metrics.json').read_text())
        assert len(metrics['per_class_curves']) == 3
        assert all(0 <= row['roc_auc'] <= 1 for row in metrics['per_class_curves'])
    else:
        assert (run / 'graphs/metric_rmse.png').exists()


def test_multiclass_curves_mark_absent_classes(tmp_path):
    from sklearn.dummy import DummyClassifier
    from sklearn_ml_toolbox.modeling.evaluation import evaluate_model
    pipe = DummyClassifier().fit(pd.DataFrame({'x': range(6)}), [0, 1, 2, 0, 1, 2])
    model = {'pipeline': pipe, 'baseline': pipe, 'features': ['x'], 'numeric': ['x'],
             'target': 'y', 'task': 'classification'}
    result = evaluate_model(model, pd.DataFrame({'x': [1, 2], 'y': [0, 0]}), tmp_path)
    assert all(row['roc_auc'] is None for row in result['per_class_curves'])
    assert all(row['reason'] for row in result['per_class_curves'])
