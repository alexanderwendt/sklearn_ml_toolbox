"""Exercise the sample entry point without training models or network access."""
from io import BytesIO
from pathlib import Path
import runpy
import shutil
import sys
from zipfile import ZipFile

import pandas as pd
import pytest

from sklearn_ml_toolbox import cli


def test_run_downloads_and_prepares_missing_data(tmp_path, monkeypatch):
    sample = Path(__file__).resolve().parents[2] / 'samples/grades'
    for name in ['run.py', 'download_grades.py', 'steps/data_generation/generate_dataset.py']:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample / name, destination)
    archive = BytesIO()
    with ZipFile(archive, 'w') as bundle:
        bundle.writestr('students.csv', 'StudentID,GPA,GradeClass,ParentalSupport,ParentalEducation,Ethnicity\n1,3.2,1,2,3,0\n')
    requests = []

    def fetch(request, timeout):
        requests.append(request.full_url)
        return BytesIO(archive.getvalue())

    def workflow(args):
        frame = pd.read_csv(tmp_path / 'data_raw/grades.csv')
        assert frame['GPA'].tolist() == [3.2]
        assert 'GradeClass' not in frame and 'StudentID' not in frame
        assert (tmp_path / 'data_raw/source_metadata.json').exists()
        assert args == ['workflow', '--config', str(tmp_path / 'workflow.toml')]
        return 0

    monkeypatch.setattr('urllib.request.urlopen', fetch)
    monkeypatch.setattr(cli, 'main', workflow)
    monkeypatch.setattr(sys, 'argv', [str(tmp_path / 'run.py')])
    for _ in range(2):
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(tmp_path / 'run.py'), run_name='__main__')
        assert exit_info.value.code == 0
    assert len(requests) == 1
