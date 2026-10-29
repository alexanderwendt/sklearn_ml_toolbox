# Grade classification

Run `python samples/grades_class/run.py` from the repository root.
The dataset-local `steps/data_generation/generate_dataset.py` generates
`data_raw/grades_class.csv` from the original CSV in `samples/grades/data_raw/`.
It removes GPA and StudentID; the target is GradeClass.
ParentalSupport and ParentalEducation are ordered numeric scores (not hours).
Ethnicity remains categorical. Numeric scores are scaled within training folds.

Models: SVM, XGBoost, linear SVM, KNN, Gaussian NB, plus a dummy baseline.
All eight feature diagnostics remain enabled and use training rows only.
Each execution creates a new `_reruns/` directory containing snapshots, logs,
models, plots, tables and `workspace/results/report.html`. Resume is explicit.
