import pandas as pd
from sklearn_ml_toolbox.config import Config
from sklearn_ml_toolbox.data_processing.preprocessing import split_dataset


def test_split_preserves_feature_target_alignment(tmp_path):
    frame = pd.DataFrame({'x': range(20), 'y': [i * 2 for i in range(20)]})
    config = Config(tmp_path/'config.toml', tmp_path/'data.csv', 'y', 'regression', tmp_path)
    train, holdout = split_dataset(frame, config)
    assert len(train) == 16 and len(holdout) == 4
    assert set(train.index).isdisjoint(holdout.index)
    assert (train.y == train.x * 2).all()
    assert (holdout.y == holdout.x * 2).all()
