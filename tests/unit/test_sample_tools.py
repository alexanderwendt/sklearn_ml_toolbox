import importlib.util
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location('sample_tools', Path(__file__).parents[2] / 'scripts' / 'migrate_samples.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_copy_preserves_original_and_detects_conflict(tmp_path):
    source, dest = tmp_path / 'original.csv', tmp_path / 'copy.csv'
    source.write_bytes(b'a,b\n1,2\n')
    migrate = module()
    record = migrate.copy_verified(source, dest)
    assert source.read_bytes() == dest.read_bytes()
    assert len(record['sha256']) == 64
    migrate.copy_verified(source, dest)
    dest.write_text('existing work')
    with pytest.raises(FileExistsError):
        migrate.copy_verified(source, dest)
    assert dest.read_text() == 'existing work'
