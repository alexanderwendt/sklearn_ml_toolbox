"""Preparation entry point works from a clean sample without legacy artifacts."""
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize('dry_run', [False, True])
def test_prepare_grades_in_isolated_run(tmp_path, dry_run):
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / 'project'
    script = project / 'scripts/prepare_samples.py'
    script.parent.mkdir(parents=True)
    shutil.copyfile(root / 'scripts/prepare_samples.py', script)
    sample = project / 'samples/grades'
    for name in ['run.py', 'workflow.toml', 'download_grades.py',
                 'steps/data_generation/generate_dataset.py']:
        destination = sample / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / 'samples/grades' / name, destination)
    if not dry_run:
        (sample / 'data_raw').mkdir()
        pd.DataFrame({
            'StudentID': range(40), 'GPA': [i / 10 for i in range(40)],
            'GradeClass': [i % 5 for i in range(40)], 'Absences': range(40),
            'Ethnicity': [i % 4 for i in range(40)],
        }).to_csv(sample / 'data_raw/Student_performance_data.csv', index=False)
    command = [sys.executable, str(script), 'grades']
    if dry_run:
        command.append('--dry-run')
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (sample / 'data_prepared').exists()
    if dry_run:
        assert not (sample / 'data_raw').exists()
        assert not (sample / '_reruns').exists()
        return
    runs = list((sample / '_reruns').iterdir())
    assert len(runs) == 1
    results = runs[0] / 'workspace/results/regression'
    prepared = pd.read_csv(results / 'prepared.csv')
    train = pd.read_csv(results / 'train.csv')
    holdout = pd.read_csv(results / 'holdout.csv')
    assert len(prepared) == 40
    assert 'StudentID' not in prepared and 'GradeClass' not in prepared
    assert set(train['_row_id']).isdisjoint(holdout['_row_id'])
    assert set(train['_row_id']) | set(holdout['_row_id']) == set(prepared['_row_id'])
    assert (results / 'split.joblib').exists()
    assert not (results / 'model.joblib').exists()
