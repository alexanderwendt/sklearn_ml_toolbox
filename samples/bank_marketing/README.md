# Bank Marketing

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

Attribution: S. Moro, P. Rita, P. Cortez. DOI: 10.24432/C5K306. License: CC BY 4.0.

Pre-contact subscription prediction: duration excluded. Random split is an IID benchmark, not a future-campaign backtest.

## Target and preparation

Predict `y` (`yes`/`no`, term-deposit subscription): binary classification, not
regression. The full older bank dataset has 45,211 rows, including 5,289 subscribers.
Use client attributes, contact context and campaign history. Remove `duration`,
which is only known after the call. Preserve source categories, including `unknown`.
Convert `pdays=-1` to missing and add `previously_contacted` to distinguish no prior
contact from an unknown interval. The prepared dataset has 16 predictors.
This is a contact-time IID benchmark, not a chronological campaign forecast.

## Run

With the toolbox installed, run `python samples/bank_marketing/run.py` from the
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

Evidence: `_reruns/20261031T115047Z_cc590ac6/workspace/results/classification/`
(`metrics.json`, `manifest.json`, and `analysis/`). This was a **logistic-regression
only** run plus the dummy baseline, not a comparison of all default models.
Subscribers are only **11.7%** of the source. Holdout accuracy **89.3%** is only
slightly above the majority baseline **88.3%**; macro F1 (**0.612** versus **0.469**)
and balanced accuracy (**0.584**) are more informative.

Subscriber precision is **0.662**, but recall is only **0.180**: most subscribers
are missed at the model's default decision rule. ROC AUC **0.772** and subscriber
average precision **0.411** indicate useful ranking signal despite weak recall.
Any contact-budget threshold should be chosen on training/validation data.

The diagnostic tree ranks the source field `day_of_week`, age, balance and campaign
count highest (roughly **10.7%, 10.5%, 10.4%, 7.9%**); previous campaign success is
also prominent. Numeric tree importance can favor variables with many split points,
so these are diagnostic associations, not proof of marketing effects. The first
two PCA components explain only **13.1%**, making a 2D projection incomplete.
Duration is deliberately excluded: using post-call information would answer a
different question. The random split does not demonstrate future-campaign accuracy.
