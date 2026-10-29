import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def command(*args, cwd=None):
    result = subprocess.run([sys.executable, '-m', 'sklearn_ml_toolbox.cli', *map(str, args)], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return Path(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_full_and_staged_workflow(tmp_path, task):
    from sklearn_ml_toolbox.prediction.inference import predict_csv
    folder = tmp_path / 'café samples'
    folder.mkdir()
    frame = pd.DataFrame({'x': range(60), 'category': ['a', 'b'] * 30, 'y': [0, 1, 2] * 20 if task == 'classification' else [i * 2 for i in range(60)]})
    frame.to_csv(folder / 'data.csv', index=False)
    config = folder / 'config.toml'
    config.write_text(f'[data]\npath="data.csv"\ntarget="y"\ntask="{task}"\n[training]\nmodels=["linear"]\nsearch_iterations=1\n')
    full = command('run', '--config', config, cwd=tmp_path)
    staged = command('prepare', '--config', config)
    command('train', '--config', config, '--run-dir', staged)
    command('evaluate', '--config', config, '--run-dir', staged)
    assert full != staged
    assert json.loads((full / 'metrics.json').read_text()) == json.loads((staged / 'metrics.json').read_text())
    assert (full / 'report.html').is_file()
    assert list((full / 'graphs').glob('*.png'))
    unseen = folder / 'new.csv'
    pd.DataFrame({'category': ['unseen'], 'x': [4]}).to_csv(unseen, index=False)
    output = predict_csv(full / 'model.joblib', unseen, folder / 'predictions.csv')
    assert len(pd.read_csv(output)) == 1
    pd.DataFrame({'category': ['a']}).to_csv(unseen, index=False)
    with pytest.raises(ValueError, match='Missing feature'):
        predict_csv(full / 'model.joblib', unseen, folder / 'bad.csv')


def test_cli_invalid_config_is_actionable(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'sklearn_ml_toolbox.cli', 'run', '--config', str(tmp_path / 'missing.toml')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'missing.toml' in result.stderr


def test_prepared_prediction_preserves_row_ids(tmp_path):
    import joblib
    from sklearn.dummy import DummyRegressor
    from sklearn_ml_toolbox.prediction.inference import predict_csv
    model = {'pipeline': DummyRegressor().fit(pd.DataFrame({'x': [1, 2]}), [3, 4]), 'features':['x'], 'numeric':['x']}
    joblib.dump(model, tmp_path/'model.joblib')
    pd.DataFrame({'_row_id': [20, 5], 'x':[1, 2]}).to_csv(tmp_path/'input.csv', index=False)
    predict_csv(tmp_path/'model.joblib', tmp_path/'input.csv', tmp_path/'output.csv')
    assert pd.read_csv(tmp_path/'output.csv')._row_id.tolist() == [20, 5]
