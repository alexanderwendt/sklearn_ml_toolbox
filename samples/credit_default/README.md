# Default of Credit Card Clients

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Attribution: I-Cheng Yeh. DOI: 10.24432/C55S3H. License: CC BY 4.0.

Taiwan credit default benchmark. Demographic predictors require review before real lending use.

## Target and preparation

Predict `Y`: next-month default (`1`) versus no default (`0`). These are labels,
so use binary classification even though the source stores numbers. All 30,000
rows are retained (6,636 defaults). Remove `ID`; use the 23 credit-limit,
demographic, repayment-status, bill-amount and payment-amount predictors.

`X2` (sex), `X3` (education), `X4` (marital status) and `X6`–`X11`
(repayment-status codes) become categorical strings. This preserves undocumented
codes as distinct categories rather than inventing numeric distances or dropping
rows. Credit limits, age, bills and payment amounts remain numeric.

## Run

With the toolbox installed, run `python samples/credit_default/run.py` from the
repository root, or `python run.py` here. It downloads missing source data,
prepares inputs and runs analysis, training, evaluation and prediction.
`python download_data.py` downloads/prepares only; add `--overwrite` to refresh.
The local script shares utilities in `scripts/download_samples.py`.

Source and prepared CSVs are `data_raw/original.csv` and `data_raw/data.csv`.
Download/preparation metadata beside them record URLs, hashes and row counts;
root `provenance.json` describes the historical import.

Results: `_reruns/<run>/workspace/results/report.html`. Options: `--models linear`,
`--dry-run`, `--resume <run-folder>`. Logistic regression and random forest are
compared using training cross-validation macro F1. The 20% holdout and CV folds
are stratified. PCA, tree importance and correlations use training rows only;
imputation, encoding and scaling are fitted inside training folds.
