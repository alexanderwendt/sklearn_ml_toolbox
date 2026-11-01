# Student GPA regression

Predict GPA from student characteristics. GradeClass and StudentID are removed
by `steps/data_generation/generate_dataset.py`, producing `data_raw/grades.csv`.
ParentalSupport and ParentalEducation are numeric ordered scores; Ethnicity is
categorical. A regression target can still have categorical predictors.

## Download and run

From the repository root:

```powershell
python samples/grades/run.py
```

Or run `python run.py` from this directory. The toolbox and the workflow's
optional dependencies must be installed in your Python environment.

`run.py` downloads the source if missing, automatically prepares `data_raw/grades.csv`,
and launches the workflow. Existing source files are reused. Resuming a run uses
its saved workspace without downloading or preparing data again.

Source: [Rabie El Kharoua's Students Performance Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).
The downloader saves the original CSV as `data_raw/Student_performance_data.csv`
and records its download URL, timestamp and SHA-256 in `data_raw/source_metadata.json`.
Existing source files are preserved unless `--overwrite` is supplied.

To download or refresh the source separately:

```powershell
python samples/grades/download_grades.py --overwrite
```

Models: OLS, Ridge (`linear`), Lasso, Elastic Net, random forest, RBF/linear SVR,
KNN and XGBoost, plus a dummy baseline. Selection uses training cross-validation
RMSE. Evaluation and saved-model predictions use held-out rows.

PCA, t-SNE, supervised/unsupervised UMAP, Lasso weights, tree importance,
recursive/backward selection and correlations use training rows only.

Each execution creates a new `_reruns/` folder with snapshots, logs, models,
plots and `workspace/results/report.html`. Resume is explicit via `--resume`.
Completed regression runs from `grade_reg` have been moved here; older combined
runs remain historical artifacts. The active workflow is regression only.

## Is `data_prepared/` necessary?

No. It contains legacy semicolon-separated features, outcomes and source copies
used by the old `config/*.ini` scripts. Neither `run.py`/`workflow.toml` nor
`config.toml` reads that directory. Existing local copies may be kept for historical
work, but they are not required for a fresh run and are ignored by Git.

Generate the current prepared data without analysis or model fitting, from the
repository root:

```powershell
python scripts/prepare_samples.py grades
# Equivalent from this dataset directory:
python run.py --until regression.prepare
```

This downloads the source when absent, removes the identifier and target-derived
GradeClass, and creates `_reruns/<run>/workspace/results/regression/` containing
`prepared.csv`, `train.csv`, `holdout.csv`, `split.joblib`, the config and manifest.
The CSVs include `_row_id` for tracking rows; it is not a predictor. These replace
the legacy separate feature/outcome CSVs. Encoding, imputation and scaling happen
later inside training folds. Continue with `python run.py --resume _reruns/<run>`.
The established directory name is `_reruns` (plural).

## Interpretation of the saved analysis

Evidence: run `20261031T113837Z_28a3749d`, under
`workspace/results/regression/` (`metrics.json`, `manifest.json`, `train.csv`,
and `analysis/Tree_Based_Importance.csv`). Nine regression candidates were compared;
training cross-validation selected Lasso. On the 20% holdout, RMSE was **0.197 GPA
points**, MAE **0.155**, and R² **0.953**, versus dummy RMSE **0.910**.

Absences dominate this dataset: training-row Pearson correlation with GPA is
**−0.919**, and the diagnostic tree assigns absences **84.7%** of its importance.
Study time and parental support have much smaller positive correlations (**0.194**
and **0.180**). The result is consistent with a strong, largely linear attendance
relationship in this benchmark; it does not establish that changing attendance
alone causes a GPA change. GradeClass is derived from GPA, so including it would
make the impressive score misleading. StudentID is likewise excluded.

The first two PCA components explain only **17.8%** of transformed predictor
variance. A diffuse unsupervised projection can coexist with excellent prediction:
the main source of feature variance need not be the main source of GPA signal.
These figures describe this source and split, not performance at another school.
