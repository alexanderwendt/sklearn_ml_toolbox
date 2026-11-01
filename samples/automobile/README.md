# Automobile

Source: https://archive.ics.uci.edu/dataset/10/automobile

Attribution: Jeffrey Schlimmer. DOI: 10.24432/C5B01C. License: CC BY 4.0.

Vehicle-price regression. Rows missing price excluded. Replaces California Housing for explicit UCI CC BY 4.0 provenance.

## Target and preparation

Predict continuous `price` from 23 vehicle-specification predictors: regression,
not classification of the source's insurance-risk `symboling` field. Exclude
`symboling` (partly derived from price) and `normalized-losses` (insurance outcomes).
The 205 source rows yield 201 labeled rows; four missing prices are excluded.
Door/cylinder counts are numeric; make, fuel, body and engine categories remain
categorical. Missing predictor values are retained for training-only imputation.

## Run

With the toolbox installed, run `python samples/automobile/run.py` from the
repository root, or `python run.py` here. This downloads missing source data,
regenerates training inputs and runs analysis, training, evaluation and prediction.
`python download_data.py` downloads/prepares only; add `--overwrite` to refresh.
The local script shares download/preparation utilities in `scripts/download_samples.py`.

`data_raw/original.csv` preserves the source; `data_raw/data.csv` is prepared.
Download and preparation metadata beside these files record URLs, hashes and row
counts. Root `provenance.json` describes the historical import.

Each run saves `_reruns/<run>/workspace/results/report.html`. Use `--models linear`
for a smaller run, `--dry-run` to inspect steps, or `--resume <run-folder>` to resume.
Defaults compare Ridge and random forest with training-only cross-validation and
a held-out 20% test set. PCA, tree importance and correlations use training rows;
encoding, scaling and imputation are fitted inside training folds.

## Interpretation of the saved analysis

Evidence: `_reruns/20261031T114929Z_9ed32282/workspace/results/regression/`
(`metrics.json`, `manifest.json`, `train.csv`, and `analysis/`). This run used
**Ridge only**, plus the dummy baseline; it is not a comparison with the current
random-forest default. Holdout RMSE was **4,295**, MAE **2,903** in source price
units, and R² **0.849**, versus dummy RMSE **11,489**.

Engine size, curb weight and horsepower have strong positive training correlations
with price (**0.870**, **0.835**, **0.803**). They also lead diagnostic tree importance
(**20.7%**, **15.5%**, **11.1%**): larger, heavier and more powerful cars tend to cost
more in this sample. These correlated specifications share information, so their
individual importances should not be read as independent causal effects.

The first two PCA components capture **26.8%** of transformed predictor variance;
two-dimensional plots omit much of the mixed numeric/categorical structure.
Only 201 labeled vehicles and a 41-row holdout underpin the reported accuracy.
The gap between MAE and RMSE indicates some much larger errors, so this is a useful
historical pricing benchmark rather than a precise estimate for every vehicle.
