from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from sklearn_ml_toolbox.config import Config, ANALYSIS_METHODS, load_config
from sklearn_ml_toolbox.data_processing.diagnostics import analyze_training, prepare_analysis_data


def setup_data(tmp_path):
    rng = np.random.default_rng(4)
    train = pd.DataFrame({'value': rng.normal(size=90), 'second': rng.normal(size=90),
                          'category': ['a', 'b', 'c'] * 30, 'constant': 1,
                          'target': [0, 1, 2] * 30}, index=np.arange(100, 190))
    config = Config(path=tmp_path / 'config.toml', input=tmp_path / 'data.csv',
                    output=tmp_path, target='target', task='classification', folds=3,
                    analysis_methods=ANALYSIS_METHODS, analysis_max_samples=60)
    return train, config


def test_all_classification_diagnostics_preserve_training_row_identity(tmp_path):
    train, config = setup_data(tmp_path)
    out = tmp_path / 'analysis'
    analyze_training(train, config, out)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert set(manifest['completed']) == set(ANALYSIS_METHODS)
    assert manifest['rows'] == 60
    assert set(manifest['row_ids']) <= set(train.index)
    assert all('target' not in name and 'constant' not in name for name in manifest['features'])
    assert set(pd.read_csv(out / 'Lasso_Model_Weights.csv').target) == {0, 1, 2}
    assert set(pd.read_csv(out / 'Backward_Elimination.csv').target) == {0, 1, 2}
    for name in ('UMAP_Supervised', 'UMAP_Unsupervised', 'T-SNE_Plot', 'PCA_Plot'):
        values = pd.read_csv(out / f'{name}.csv', index_col='_row_id')
        assert values.index.tolist() == manifest['row_ids']
        np.testing.assert_array_equal(values.target, train.loc[values.index, 'target'])
        assert np.isfinite(values[['component_1', 'component_2']]).all().all()
        assert (out / f'{name}.png').exists()


def test_regression_weights_and_subsampling_are_reproducible(tmp_path):
    train, config = setup_data(tmp_path)
    train['target'] = 2 * train.value - train.second
    config = replace(config, task='regression', analysis_methods=('lasso', 'tree', 'recursive', 'backward'))
    a, ya, _ = prepare_analysis_data(train, config)
    b, yb, _ = prepare_analysis_data(train, config)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(ya, yb)
    analyze_training(train, config, tmp_path / 'regression')
    weights = pd.read_csv(tmp_path / 'regression/Lasso_Model_Weights.csv')
    assert weights.target.unique().tolist() == ['target']
    assert weights.loc[weights.feature == 'numeric__value', 'coefficient'].iloc[0] > 0
    assert weights.loc[weights.feature == 'numeric__second', 'coefficient'].iloc[0] < 0


@pytest.mark.parametrize('analysis', ['methods=["unknown"]', 'max_samples=2', 'methods=["pca", "pca"]'])
def test_invalid_analysis_options_fail_early(tmp_path, analysis):
    path = tmp_path / 'config.toml'
    path.write_text('[data]\npath="data.csv"\ntarget="target"\ntask="classification"\n[analysis]\n' + analysis)
    with pytest.raises(ValueError, match='analysis.'):
        load_config(path)
