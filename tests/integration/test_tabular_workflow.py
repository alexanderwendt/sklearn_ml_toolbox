import json

import joblib
import numpy as np
import pandas as pd

from sklearn_ml_toolbox.workflow_runner import run_workflow


def test_complete_two_task_workflow_isolated_and_resumable(tmp_path):
    raw = tmp_path / 'data_raw'
    raw.mkdir()
    values = np.tile(np.arange(6), 12)
    pd.DataFrame({'StudentID': np.arange(len(values)), 'StudyTimeWeekly': values,
                  'GPA': 1 + values / 2, 'GradeClass': values % 3}).to_csv(raw / 'students.csv', index=False)
    config = tmp_path / 'workflow.toml'
    config.write_text('''[workflow]
version = 1
engine = "tabular"
inputs = ["data_raw/students.csv"]
[defaults.data]
path = "data_raw/students.csv"
prepare = "builtin:grades"
[defaults.training]
models = ["linear"]
folds = 2
search_iterations = 1
[defaults.analysis]
methods = ["pca", "lasso", "tree"]
[profiles.classification.data]
task = "classification"
target = "GradeClass"
drop = ["StudentID", "GPA"]
[profiles.regression.data]
task = "regression"
target = "GPA"
drop = ["StudentID", "GradeClass"]
''')
    run = run_workflow(config, until='classification.prepare')
    assert json.loads((run / 'manifest.json').read_text())['status'] == 'paused'
    assert run_workflow(config, resume=run) == run
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert len(manifest['attempts']) == 11
    for task in ('classification', 'regression'):
        folder = run / 'workspace/results' / task
        assert (folder / 'report.html').exists()
        model = joblib.load(folder / 'model.joblib')
        assert model['features'] == ['StudyTimeWeekly']
        assert set(model['candidates']) == {'dummy', 'linear'}
        holdout = pd.read_csv(folder / 'holdout.csv', index_col='_row_id')
        predictions = pd.read_csv(folder / 'inference/holdout_predictions.csv', index_col='_row_id')
        pd.testing.assert_index_equal(holdout.index, predictions.index)
        assert len(predictions) == 15
        diagnostics = json.loads((folder / 'analysis/manifest.json').read_text())
        assert set(diagnostics['row_ids']).isdisjoint(holdout.index)
        assert set(diagnostics['completed']) == {'lasso', 'tree'}
        assert 'pca' in diagnostics['skipped']  # This fixture has only one predictor.
        assert (folder / 'analysis/Lasso_Model_Weights.png').exists()
        assert 'Training-only feature diagnostics' in (folder / 'report.html').read_text()
    assert (run / 'workspace/results/report.html').exists()
    assert len(pd.read_csv(run / 'workspace/results/summary.csv')) == 2
    assert not (tmp_path / 'results').exists()
    attempts = len(manifest['attempts'])
    run_workflow(config, resume=run)
    assert len(json.loads((run / 'manifest.json').read_text())['attempts']) == attempts
