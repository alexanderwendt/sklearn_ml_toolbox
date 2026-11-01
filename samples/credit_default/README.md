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

## Interpretation of the saved analysis

Evidence: `_reruns/20261031T115141Z_c1822de8/workspace/results/classification/`
(`metrics.json`, `manifest.json`, and `analysis/`). This run evaluated **logistic
regression only**, plus a dummy baseline. Defaults comprise **22.1%** of the source.
Holdout accuracy is **81.7%** versus majority-baseline **77.9%**, macro F1 **0.675**
versus **0.438**, and balanced accuracy **0.650**.

Default precision **0.661** and recall **0.352** show that roughly two thirds of
actual defaults are missed at the default decision rule. ROC AUC **0.758** and
default average precision **0.528** show ranking value, but do not imply that
accuracy alone is sufficient for choosing a lending threshold.

The strongest individual diagnostic-tree feature is `X6=2` (September repayment
status: two-month delay, **6.5%** importance), followed by age `X5` and credit limit
`X1` (about **5.4%** and **5.3%**). Bills and payments distribute signal across many
related variables and months; a one-hot feature's importance is not the total
importance of its original variable. PCA's first two components explain **20.9%**
of transformed variance. Repayment history is informative, but substantial overlap
between classes remains. This historical Taiwanese cohort does not establish
performance on a new population or future economic period.
