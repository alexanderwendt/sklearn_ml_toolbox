# GPA regression

Run `python samples/grade_reg/run.py` from the repository root.
The dataset-local `steps/data_generation/generate_dataset.py` generates
`data_raw/grade_reg.csv` from the original CSV in `samples/grades/data_raw/`.
It removes GradeClass and StudentID; the target is GPA.
ParentalSupport and ParentalEducation are ordered numeric scores (not hours).
Ethnicity remains categorical. Numeric scores are scaled within training folds.

Models: ordinary least squares (`ols`), Ridge (`linear`), Lasso, Elastic Net,
random forest, RBF SVR, linear SVR, KNN, XGBoost, plus a dummy baseline.
Selection uses training cross-validation RMSE; evaluation uses held-out rows.
All eight feature diagnostics remain enabled and use training rows only.
Each execution creates a new `_reruns/` directory containing snapshots, logs,
models, plots, tables and `workspace/results/report.html`. Resume is explicit.
