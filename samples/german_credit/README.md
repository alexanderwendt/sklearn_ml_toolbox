# Statlog (German Credit Data)

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Attribution: Hans Hofmann. DOI: 10.24432/C5NC77. License: CC BY 4.0.

Historical credit-risk benchmark; macro F1 does not implement the original asymmetric cost matrix.

## Target and preparation

Predict `class`: `1` = good credit risk, `2` = bad credit risk. These numeric codes
are discrete labels, so use binary classification. All 1,000 rows are retained
(700 good, 300 bad), with 20 account, history, loan and applicant predictors.
Source qualitative codes are preserved as categorical strings; duration, amount,
age and counts remain numeric. Do not regress the arbitrary class codes.
Macro F1 model selection does not reproduce the source's 5:1 misclassification
cost for accepting a bad risk versus rejecting a good risk.

## Run

With the toolbox installed, run `python samples/german_credit/run.py` from the
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
