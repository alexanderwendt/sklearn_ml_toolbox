from pathlib import Path

import pytest


def write_config(tmp_path, extra=""):
    path = tmp_path / "config.toml"
    path.write_text('[data]\npath="data.csv"\ntarget="y"\ntask="regression"\n' + extra)
    return path


def test_config_paths_relative_to_file(tmp_path, monkeypatch):
    from sklearn_ml_toolbox.config import load_config
    folder = tmp_path / "café samples"
    folder.mkdir()
    path = write_config(folder)
    monkeypatch.chdir(tmp_path)
    config = load_config(path)
    assert config.input == folder / "data.csv"
    assert config.output == folder / "results"


@pytest.mark.parametrize("extra", ['typo=1', '[split]\ntest_size=1', '[training]\nfolds=1'])
def test_invalid_fields_rejected(tmp_path, extra):
    from sklearn_ml_toolbox.config import load_config
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, extra))


def test_invalid_task(tmp_path):
    from sklearn_ml_toolbox.config import load_config
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace('regression', 'guess'))
    with pytest.raises(ValueError, match="task"):
        load_config(path)
