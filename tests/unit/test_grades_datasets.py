from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from sklearn_ml_toolbox.config import Config
from sklearn_ml_toolbox.modeling.training import train_model
from sklearn_ml_toolbox.data_generation.grades import prepare
from sklearn_ml_toolbox.data_processing.preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('dataset,target,excluded', [
    ('grades_class', 'GradeClass', 'GPA'), ('grade_reg', 'GPA', 'GradeClass')])
def test_generated_dataset_preserves_target_and_removes_leakage(tmp_path, dataset, target, excluded):
    generate = runpy.run_path(str(ROOT / 'samples' / dataset / 'steps/data_generation/generate_dataset.py'))['generate']
    source = ROOT / 'samples/grades/data_raw/Student_performance_data.csv'
    original = pd.read_csv(source)
    result = pd.read_csv(generate(source, tmp_path / 'data.csv'))
    pd.testing.assert_series_equal(result[target], original[target])
    assert excluded not in result and 'StudentID' not in result
    assert list(result.columns) == [c for c in original if c not in (excluded, 'StudentID')]
    prepared = prepare(result)
    predictors = prepared.drop(columns=[target])
    encoder = make_preprocessor(predictors).fit(predictors)
    names = encoder.get_feature_names_out().tolist()
    for column in ('ParentalSupport', 'ParentalEducation'):
        assert pd.api.types.is_numeric_dtype(prepared[column])
        assert [name for name in names if column in name] == [f'numeric__{column}']
    assert not pd.api.types.is_numeric_dtype(prepared['Ethnicity'])
    assert len([name for name in names if 'Ethnicity' in name]) > 1


def test_added_regressors_learn_continuous_target(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=80)
    frame = pd.DataFrame({'x': x, 'target': 2 * x + 1})
    config = Config(path=tmp_path / 'config.toml', input=tmp_path / 'data.csv',
                    output=tmp_path, target='target', task='regression',
                    models=('ols', 'lasso', 'elastic_net'), folds=3)
    result = train_model(frame, config)
    scores = {row['model']: row['cv_score'] for row in result['cv_results']}
    for name in config.models:
        assert scores[name] > scores['dummy']
    assert scores['ols'] > -1e-8
