# Student GPA regression

Predict GPA from student characteristics. GradeClass and StudentID are removed
by `steps/data_generation/generate_dataset.py`, producing `_old/grades.csv`.
ParentalSupport and ParentalEducation are numeric ordered scores; Ethnicity is
categorical. A regression target can still have categorical predictors.

## Download and run

From the repository root:

```powershell
python samples/grades/run.py
```

Or run `python run.py` from this directory. The toolbox and the workflow's
optional dependencies must be installed in your Python environment.

`run.py` downloads the source if missing, automatically prepares `_old/grades.csv`,
and launches the workflow. Existing source files are reused. Resuming a run uses
its saved workspace without downloading or preparing data again.

Source: [Rabie El Kharoua's Students Performance Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).
The downloader saves the original CSV as `_old/Student_performance_data.csv`
and records its download URL, timestamp and SHA-256 in `source_metadata.json`.
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
