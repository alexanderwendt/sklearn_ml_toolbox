from copy import deepcopy
from pathlib import Path

import pytest

from sklearn_ml_toolbox import analysis_workflow as analysis
from sklearn_ml_toolbox import workflow_runner as runner
from sklearn_ml_toolbox.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def grades_config():
    return runner.read_workflow(ROOT / 'samples/grades/workflow.toml')


def test_grades_plan_runs_regression_and_preserves_omx_plan():
    config = grades_config()
    plan = runner.build_plan(config)
    assert [step['id'] for step in plan] == [
        'regression.prepare', 'regression.analyze', 'regression.train', 'regression.evaluate', 'regression.predict', 'report',
    ]
    omx = runner.read_workflow(ROOT / 'samples/debug_omxs30/workflow.toml')
    assert {'ols', 'lasso', 'elastic_net', 'forest'} <= set(plan[0]['models'])
    assert 'gnb' not in next(s for s in plan if s['profile'] == 'regression')['models']
    assert len(runner.build_plan(omx)) == 78


def test_task_and_model_selection_validate_incompatible_combinations():
    config = grades_config()
    plan = runner.build_plan(config, models=['ols'], tasks=['regression'])
    assert len(plan) == 6
    assert all(step['models'] == ['ols'] for step in plan)
    with pytest.raises(ValueError, match='only supports classification'):
        runner.build_plan(config, models=['gnb'])
    with pytest.raises(ValueError, match='Tasks must'):
        runner.build_plan(config, tasks=['unknown'])


def test_resolved_task_configs_point_to_snapshots_and_exclude_alternate_outcomes(tmp_path):
    config = grades_config()
    (tmp_path / 'data_raw').mkdir()
    (tmp_path / 'data_raw/grades.csv').write_text('GPA,Age\n3,16\n')
    (tmp_path / 'config').mkdir()
    for task, target, excluded in [('regression', 'GPA', 'GradeClass')]:
        step = next(s for s in runner.build_plan(config) if s['id'] == f'{task}.prepare')
        path = tmp_path / step['config']
        analysis.write_task_config(config, step, path)
        resolved = load_config(path)
        assert resolved.input == tmp_path / 'data_raw/grades.csv'
        assert resolved.target == target
        assert resolved.drop == ('StudentID', excluded)
        assert resolved.prepare == 'builtin:grades'
    invalid = deepcopy(config)
    invalid['defaults']['data']['path'] = '../../outside.csv'
    with pytest.raises(ValueError, match='relative to the dataset'):
        analysis.write_task_config(invalid, step, path)


def test_failed_stage_does_not_damage_committed_artifacts(tmp_path, monkeypatch):
    config_path = tmp_path / 'config.toml'
    config_path.write_text('[data]\npath="input.csv"\ntarget="target"\ntask="regression"\n')
    directory = tmp_path / 'results/regression'
    directory.mkdir(parents=True)
    (directory / 'model.joblib').write_bytes(b'original model')

    def fail(config, staging):
        (staging / 'model.joblib').write_bytes(b'damaged temporary copy')
        (staging / 'metrics.json').write_text('partial metrics')
        raise ValueError('plot failed')

    monkeypatch.setattr(analysis, 'evaluate_run', fail)
    with pytest.raises(ValueError, match='plot failed'):
        analysis.execute_stage(config_path, 'evaluate', directory)
    assert (directory / 'model.joblib').read_bytes() == b'original model'
    assert not (directory / 'metrics.json').exists()
    assert not list((tmp_path / '.staging').iterdir())
