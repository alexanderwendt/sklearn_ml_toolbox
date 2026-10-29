from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from sklearn_ml_toolbox import workflow_runner as runner


SAMPLE = Path(__file__).resolve().parents[2] / 'samples/debug_omxs30'
# Canonical effective configuration hashes captured from the verified legacy INIs
# before removing them. Originals are preserved in the November 2022 run baseline.
LEGACY_CONFIGS = {
    'debug_timedata_omxs30_datapreparation_train': 'a0c8381e437f92331fa1d6da94d91a703b5e399ba7640e1a871fd7f9e2850c3b',
    'debug_timedata_omxs30_infer_gnb': 'aee8553f025351487abe8fe1fb1b92ecd0e2d674e6245eebfe868cadace850ba',
    'debug_timedata_omxs30_infer_knn': '474558d158dfeac6ee16c146cd7444e37a735785f81309a58ceb3ba7e8219c2b',
    'debug_timedata_omxs30_infer_svm': '3af483e2efe824d7541b2cd4b51307debda7603f4272335f48804ec15b4b26a4',
    'debug_timedata_omxs30_infer_svm_linear': '39e45db3b08ceb9e9786f17144e54c829a21602740d096ffa7e31b19676bd4d9',
    'debug_timedata_omxs30_infer_xgboost': '5cfe49ebd37cdfa17f41cbae28323e200ff2ad9b0541a8d33803721e92ed7a94',
    'debug_timedata_omxs30_train_gnb': '2c5c7ce3cf2700ae155f841dac5c133efc6df538c2e031850122d074319813f4',
    'debug_timedata_omxs30_train_knn': '7718476d2810f5e0824a6f54ddc5e9dd936ab522a181d0e5dade78acff8f8f89',
    'debug_timedata_omxs30_train_svm': '4f84fe04fd4db6e2ca9f892def14f2d8e17e91776f1ff8879e99809882c8f5d6',
    'debug_timedata_omxs30_train_svm_linear': '0b38a35ee80f82c61475d40cb06bd84be3cce8f38a68d925324948b1f6cea050',
    'debug_timedata_omxs30_train_xgboost': 'ce3f99c35bd17743eac3f5223626aed3a639cebb95b0282bb143d3e1c0955554',
}


def test_generated_configs_match_every_legacy_value():
    config = runner.read_workflow(SAMPLE / 'workflow.toml')
    cases = [('prepare', None, 'debug_timedata_omxs30_datapreparation_train')]
    for model in config['models']:
        cases.extend([('train', model, f'debug_timedata_omxs30_train_{model}'),
                      ('infer', model, f'debug_timedata_omxs30_infer_{model}')])
    for profile, model, filename in cases:
        new = runner.resolved_config(config, profile, model)
        canonical = json.dumps({s: dict(new.items(s)) for s in new.sections()}, sort_keys=True)
        assert sha256(canonical.encode()).hexdigest() == LEGACY_CONFIGS[filename], filename


def test_plan_matches_legacy_commands_and_order():
    config = runner.read_workflow(SAMPLE / 'workflow.toml')
    plan = runner.build_plan(config)
    assert len(plan) == 78
    canonical = json.dumps([(s['module'], s['args']) for s in plan])
    assert sha256(canonical.encode()).hexdigest() == 'bb80f6870c71968ea20de8e6928ecada632f4e50a8005002ec39a99401341be4'


@pytest.fixture
def workflow(tmp_path):
    path = tmp_path / 'workflow.toml'
    path.write_bytes((SAMPLE / 'workflow.toml').read_bytes())
    for name in ('data_raw', 'annotations'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'input.txt').write_text('original input')
    return path


def test_new_runs_are_isolated_and_resume_skips_completed(workflow, monkeypatch):
    commands = []

    def execute(command, **kwargs):
        if command[2] != 'pip':
            commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', execute)
    first = runner.run_workflow(workflow, models=['svm'], until='prepare.groundtruth')
    second = runner.run_workflow(workflow, models=['svm'], until='prepare.groundtruth')
    assert first != second
    assert len(commands) == 2
    runner.run_workflow(workflow, resume=first, until='prepare.features')
    assert len(commands) == 3
    manifest = json.loads((first / 'manifest.json').read_text())
    assert manifest['completed'] == ['prepare.groundtruth', 'prepare.features']
    assert manifest['status'] == 'paused'
    assert (first / 'workspace/data_raw/input.txt').read_text() == 'original input'
    (first / 'workspace/data_raw/input.txt').write_text('changed')
    with pytest.raises(ValueError, match='snapshot changed'):
        runner.run_workflow(workflow, resume=first)


def test_failure_stops_and_retry_preserves_attempt_history(workflow, monkeypatch):
    def fail(command, **kwargs):
        return SimpleNamespace(returncode=0 if command[2] == 'pip' else 7)

    monkeypatch.setattr(runner.subprocess, 'run', fail)
    with pytest.raises(ValueError, match='failed'):
        runner.run_workflow(workflow, models=['gnb'])
    run = next((workflow.parent / '_old_reruns').iterdir())
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['completed'] == []
    assert len(manifest['attempts']) == 1
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    runner.run_workflow(workflow, resume=run, until='prepare.groundtruth')
    manifest = json.loads((run / 'manifest.json').read_text())
    assert [a['exit_code'] for a in manifest['attempts']] == [7, 0]


def test_dry_run_has_no_outputs_and_model_overrides_are_explicit(workflow):
    assert runner.run_workflow(workflow, models=['knn'], dry_run=True) is None
    assert not (workflow.parent / '_old_reruns').exists()
    config = runner.read_workflow(workflow)
    config['steps'] = {'knn.wide': {'args': ['--execute_wide=True']}}
    assert next(s for s in runner.build_plan(config, ['knn']) if s['id'] == 'knn.wide')['args'] == ['--execute_wide=True']
    config['steps']['typo'] = {'args': []}
    with pytest.raises(ValueError, match='Unknown step'):
        runner.build_plan(config)
