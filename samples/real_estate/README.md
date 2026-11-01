# Real Estate Valuation

Source: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set

Attribution: I-Cheng Yeh. DOI: 10.24432/C5J30W. License: CC BY 4.0.

Price per unit area, Taiwan. IID benchmark; nearby properties and dates may correlate.

## Target and preparation

Predict continuous `Y house price of unit area`: regression, in units of
10,000 New Taiwan dollars per ping (about 3.3 square metres). All 414 rows are
retained. Remove row identifier `No`. Keep six numeric predictors: transaction
date (fractional year), house age, MRT distance, convenience-store count,
latitude and longitude. Keep the source price scale without discretizing it.
The random split is an IID benchmark, not a spatial or future-price backtest.

## Run

With the toolbox installed, run `python samples/real_estate/run.py` from the
repository root, or `python run.py` here. It downloads missing source data,
prepares inputs and runs analysis, training, evaluation and prediction.
`python download_data.py` downloads/prepares only; add `--overwrite` to refresh.
The local script shares utilities in `scripts/download_samples.py`.

Source and prepared CSVs are `data_raw/original.csv` and `data_raw/data.csv`.
Download/preparation metadata beside them record URLs, hashes and row counts;
root `provenance.json` describes the historical import.

Results: `_reruns/<run>/workspace/results/report.html`. Options: `--models linear`,
`--dry-run`, `--resume <run-folder>`. Ridge and random forest are compared using
training cross-validation RMSE and a held-out 20% test set. PCA, tree importance
and correlations use training rows only; imputation, encoding and scaling are
fitted inside training folds.

## Interpretation of the saved analysis

Evidence: `_reruns/20261031T115316Z_2f502eeb/workspace/results/regression/`
(`metrics.json`, `manifest.json`, `train.csv`, and `analysis/`). The recorded run
used **Ridge only**, plus a dummy baseline. Holdout RMSE is **7.286**, MAE **5.295**
(10,000 NTD per ping), and R² **0.684**, versus dummy RMSE **13.114**.

Distance to the nearest MRT station has a strong negative training correlation
with price (**−0.657**); convenience-store count and latitude correlate positively
(**0.554**, **0.531**). MRT distance leads diagnostic-tree importance (**33.3%**),
followed by latitude (**17.4%**) and store count (**16.5%**). Accessibility and
location are central to this local market, but correlated location variables do
not establish a causal price premium for adding a store or station.

The first two PCA components explain **61.2%** of transformed variance, reflecting
substantial shared geographic structure. The 83-row random holdout still leaves
meaningful price error. Nearby transactions and dates can resemble training rows;
these results do not measure extrapolation to new districts or later market cycles.
