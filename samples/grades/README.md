# Student grades

Run `python run.py` for GPA regression; `python run.py config_classification.toml` for GradeClass classification.

Both exclude StudentID and the alternate outcome. Shared `builtin:grades` hook treats demographic/support codes as categorical. No downstream script edits.

Original CSV supplied with this repository; its redistribution license has not been independently established. Existing raw file preserved; external working copy verified by SHA256. Do not infer an open-data license from the generated model report.

Reports: `results/regression/<run>/report.html` and `results/classification/<run>/report.html`. External verified copies/results live under `../sklearn_ml_toolbox_samples/grades` relative to repository root.

Each new report includes every input feature's distribution, missing-value count, and target relationship; CV scores and fit times; and separate training/holdout metrics and predictions for linear, forest, and dummy baseline models. Regression adds error/residual diagnostics. Classification adds per-class precision, recall, F1, support, confusion matrices, and one-vs-rest ROC/precision–recall curves. Per-model permutation importance covers every original feature. CSV tables and PNG plots are linked from the HTML report.

Run both tasks from the repository root:

```bash
.verify-venv/Scripts/python samples/grades/run.py
.verify-venv/Scripts/python samples/grades/run.py config_classification.toml
```

Training scores are in-sample diagnostics. Holdout metrics and permutation importance must not be used for tuning; importance is not causal. Financial backtesting and time-series plots from OMX do not apply to student grades. New runs above are written locally; external copies are not automatically synchronized.
