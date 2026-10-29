import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


def module():
    spec = importlib.util.spec_from_file_location('downloads', Path(__file__).parents[2] / 'scripts' / 'download_samples.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_interrupted_download_does_not_create_destination(tmp_path):
    downloads = module()
    with patch.object(downloads, 'urlopen', side_effect=OSError('offline')):
        with pytest.raises(OSError):
            downloads.download_file('https://example.org/data.csv', tmp_path/'data.csv')
    assert not (tmp_path/'data.csv').exists()


def test_dataset_task_counts():
    datasets = module().DATASETS
    assert sum(item['task'] == 'classification' for item in datasets.values()) == 3
    assert sum(item['task'] == 'regression' for item in datasets.values()) == 2


def test_truncated_response_never_published(tmp_path):
    from io import BytesIO
    downloads = module()
    response = BytesIO(b'x,y\n1,2\n')
    response.headers = {'Content-Length': '1000'}
    with patch.object(downloads, 'urlopen', return_value=response):
        with pytest.raises(OSError, match='Incomplete'):
            downloads.download_file('https://example.org/data.csv', tmp_path/'data.csv')
    assert not (tmp_path/'data.csv').exists()
