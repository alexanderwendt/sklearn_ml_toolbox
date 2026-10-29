import pandas as pd
import pytest


def config(tmp_path, extra=""):
    from sklearn_ml_toolbox.config import load_config
    path = tmp_path / "config.toml"
    path.write_text('[data]\npath="data.csv"\ntarget="y"\ntask="regression"\n' + extra)
    return load_config(path)


def test_csv_without_hook_and_missing_labels(tmp_path):
    from sklearn_ml_toolbox.data_generation.loader import prepare_dataset
    pd.DataFrame({'x': [1, None, 3], 'y': [4, 5, 6]}).to_csv(tmp_path / 'data.csv', index=False)
    frame = prepare_dataset(config(tmp_path))
    assert frame.y.tolist() == [4, 5, 6]
    assert frame.index.tolist() == [0, 1, 2]
    pd.DataFrame({'x': [1], 'y': [None]}).to_csv(tmp_path / 'data.csv', index=False)
    with pytest.raises(ValueError, match="target"):
        prepare_dataset(config(tmp_path))


def test_hook_preserves_row_identity(tmp_path):
    from sklearn_ml_toolbox.data_generation.loader import prepare_dataset
    pd.DataFrame({'x': [10, 20, 30], 'y': [1, 2, 3]}).to_csv(tmp_path / 'data.csv', index=False)
    hook = tmp_path / 'prepare.py'
    hook.write_text('def prepare(frame):\n    return frame.iloc[::-1].iloc[:2]\n')
    frame = prepare_dataset(config(tmp_path, 'prepare="prepare.py"'))
    assert frame.index.tolist() == [2, 1]
    assert frame.y.tolist() == [3, 2]
    hook.write_text('def prepare(frame):\n    return frame.iloc[[0, 0]]\n')
    with pytest.raises(ValueError, match="identity"):
        prepare_dataset(config(tmp_path, 'prepare="prepare.py"'))


def test_grades_excludes_alternate_outcome(tmp_path):
    from sklearn_ml_toolbox.config import load_config
    from sklearn_ml_toolbox.data_generation.loader import prepare_dataset
    pd.DataFrame({'StudentID': [1, 2], 'Gender': [0, 1], 'GPA': [2., 3.], 'GradeClass': [3, 1]}).to_csv(tmp_path / 'data.csv', index=False)
    path = tmp_path / 'config.toml'
    path.write_text('[data]\npath="data.csv"\ntarget="GPA"\ntask="regression"\nprepare="builtin:grades"\ndrop=["StudentID", "GradeClass"]')
    frame = prepare_dataset(load_config(path))
    assert list(frame.columns) == ['Gender', 'GPA']
    assert not pd.api.types.is_numeric_dtype(frame.Gender)


def test_hook_reset_index_rejected(tmp_path):
    from sklearn_ml_toolbox.data_generation.loader import prepare_dataset
    pd.DataFrame({'x': [10, 20, 30], 'y': [1, 2, 3]}).to_csv(tmp_path/'data.csv', index=False)
    (tmp_path/'prepare.py').write_text('def prepare(frame):\n    return frame.iloc[::-1].reset_index(drop=True)\n')
    with pytest.raises(ValueError, match='identity'):
        prepare_dataset(config(tmp_path, 'prepare="prepare.py"'))


def test_unsorted_temporal_features_do_not_see_holdout(tmp_path):
    from sklearn_ml_toolbox.data_generation.loader import prepare_dataset
    from sklearn_ml_toolbox.data_processing.preprocessing import split_dataset
    frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=100), 'Close': range(100, 200), 'y': range(100)})
    frame.iloc[::-1].to_csv(tmp_path/'data.csv', index=False)
    cfg = config(tmp_path, 'prepare="builtin:stockmarket"\n[split]\nstrategy="temporal"\ntime_column="Date"')
    original, _ = split_dataset(prepare_dataset(cfg), cfg)
    frame.loc[80, 'Close'] = 99999
    frame.iloc[::-1].to_csv(tmp_path/'data.csv', index=False)
    changed, _ = split_dataset(prepare_dataset(cfg), cfg)
    pd.testing.assert_frame_equal(original, changed)
