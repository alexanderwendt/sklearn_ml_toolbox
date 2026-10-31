"""Offline first-run and schema tests for the five downloadable UCI samples."""
from io import BytesIO
import json
from pathlib import Path
import runpy
import shutil
import sys

import pandas as pd
import pytest

from sklearn_ml_toolbox import cli

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'automobile': 'price,symboling,normalized-losses,num-of-doors,num-of-cylinders,make\n10000,1,100,two,four,audi\n?,2,?,four,six,bmw\n',
    'bank_marketing': 'y,duration,pdays,age,job\nyes,300,-1,25,student\nno,100,10,35,technician\n',
    'credit_default': 'ID,Y,X1,X2,X3,X4,X6\n1,1,10000,1,2,1,-1\n2,0,20000,2,0,3,2\n',
    'german_credit': 'class,Attribute1,Attribute2\n1,A11,6\n2,A12,12\n',
    'real_estate': 'No,Y house price of unit area,X1 transaction date,X2 house age\n1,37.9,2012.917,32\n2,42.2,2013.5,19.5\n',
}


def sample_tree(tmp_path, name):
    folder = tmp_path / 'samples' / name
    folder.mkdir(parents=True)
    for file in ['run.py', 'download_data.py', 'workflow.toml', 'source_metadata.json']:
        shutil.copyfile(ROOT / 'samples' / name / file, folder / file)
    (tmp_path / 'scripts').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'scripts/download_samples.py', tmp_path / 'scripts/download_samples.py')
    return folder


@pytest.mark.parametrize('name', SOURCES)
def test_first_run_download_prepare_and_cached_rerun(tmp_path, monkeypatch, name):
    folder = sample_tree(tmp_path, name)
    requests, calls = [], []

    def fetch(url, timeout):
        requests.append(url)
        response = BytesIO(SOURCES[name].encode())
        response.headers = {}
        return response

    def workflow(args):
        calls.append(args)
        assert args[:3] == ['workflow', '--config', str(folder / 'workflow.toml')]
        assert args[3:] == ['--models', 'linear']
        assert (folder / 'data_raw/data.csv').exists()
        return 0

    monkeypatch.setattr('urllib.request.urlopen', fetch)
    monkeypatch.setattr(cli, 'main', workflow)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--models', 'linear'])
    monkeypatch.chdir(tmp_path)
    prepared_mtimes = []
    for _ in range(2):
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(folder / 'run.py'), run_name='__main__')
        assert result.value.code == 0
        prepared_mtimes.append((folder / 'data_raw/data.csv').stat().st_mtime_ns)
    assert len(requests) == 1
    assert len(calls) == 2
    assert prepared_mtimes[0] == prepared_mtimes[1]
    frame = pd.read_csv(folder / 'data_raw/data.csv')
    metadata = json.loads((folder / 'data_raw/preparation_metadata.json').read_text())
    assert not set(metadata['dropped_columns']).intersection(frame.columns)
    assert (folder / 'data_raw/original.csv').read_text() == SOURCES[name]
    if name == 'automobile':
        assert frame['num-of-doors'].tolist() == [2]
        assert frame['num-of-cylinders'].tolist() == [4]
        assert metadata['removed_unlabeled_rows'] == 1
    elif name == 'bank_marketing':
        assert frame['previously_contacted'].tolist() == [0, 1]
        assert pd.isna(frame['pdays'][0]) and frame['pdays'][1] == 10
        assert frame['y'].tolist() == ['yes', 'no']
    elif name == 'credit_default':
        assert frame['X2'].tolist() == ['category_1', 'category_2']
        assert frame['X3'].tolist() == ['category_2', 'category_0']
        assert frame['X6'].tolist() == ['category_-1', 'category_2']
        assert frame['X1'].tolist() == [10000, 20000]
    elif name == 'german_credit':
        assert frame['Attribute1'].tolist() == ['category_A11', 'category_A12']
        assert frame['class'].tolist() == [1, 2]
    else:
        assert frame['Y house price of unit area'].tolist() == [37.9, 42.2]


@pytest.mark.parametrize('name', SOURCES)
@pytest.mark.parametrize('args', [['--help'], ['--dry-run'], ['--resume=saved-run']])
def test_inspect_and_resume_do_not_download_or_prepare(tmp_path, monkeypatch, name, args):
    folder = sample_tree(tmp_path, name)
    monkeypatch.setattr(cli, 'main', lambda argv: 0)
    monkeypatch.setattr(sys, 'argv', ['run.py', *args])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(folder / 'run.py'), run_name='__main__')
    assert result.value.code == 0
    assert not (folder / 'data_raw').exists()


def test_invalid_refresh_preserves_original(tmp_path, monkeypatch):
    folder = sample_tree(tmp_path, 'real_estate')
    original = folder / 'data_raw/original.csv'
    original.parent.mkdir()
    original.write_text(SOURCES['real_estate'])
    response = BytesIO(b'<html>temporarily unavailable</html>')
    response.headers = {}
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: response)
    download = runpy.run_path(str(folder / 'download_data.py'))['download']
    with pytest.raises(ValueError, match='Expected nonempty'):
        download(overwrite=True)
    assert original.read_text() == SOURCES['real_estate']
    assert list(original.parent.iterdir()) == [original]


def test_automobile_accepts_numeric_uci_counts(tmp_path):
    folder = sample_tree(tmp_path, 'automobile')
    (folder / 'data_raw').mkdir()
    (folder / 'data_raw/original.csv').write_text(SOURCES['automobile'].replace('two,four', '2,4').replace('four,six', '4,6'))
    generate = runpy.run_path(str(folder / 'download_data.py'))['generate']
    frame = pd.read_csv(generate())
    assert frame['num-of-doors'].tolist() == [2]
    assert frame['num-of-cylinders'].tolist() == [4]


def test_unexpected_class_label_does_not_replace_prepared_data(tmp_path):
    folder = sample_tree(tmp_path, 'german_credit')
    (folder / 'data_raw').mkdir()
    (folder / 'data_raw/original.csv').write_text(SOURCES['german_credit'].replace('1,A11', '3,A11'))
    prepared = folder / 'data_raw/data.csv'
    prepared.write_bytes(b'existing prepared data')
    generate = runpy.run_path(str(folder / 'download_data.py'))['generate']
    with pytest.raises(ValueError, match='Unexpected german_credit target labels'):
        generate()
    assert prepared.read_bytes() == b'existing prepared data'
