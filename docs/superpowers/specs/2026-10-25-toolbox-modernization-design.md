# Toolbox modernization design

## Intent

CSV → dataset preparation → shared classification/regression pipeline → saved predictions, metrics, graphs. Minimal dataset-specific code; reproducible sample runs.

User-approved direction: replace old commands/configs with cleaner interface; migrate data; use sibling `sklearn_ml_toolbox_samples` repo when useful. Full project refactor, current-functionality tests, newest viable Python, working grades example, five additional free datasets (three classification, two regression), financial preference.

## Current findings

- `src/` already separates generation, processing, modeling, prediction; duplicated utility modules also exist at root.
- CLI argument parsing at import blocks library use and test collection.
- Grades classification config targets stock-market `LongTrend`; validation input references training data.
- Grades regression config targets `GPA`; CSV contains both `GPA` and `GradeClass`.
- Root regression smoke script references missing root-level commands.
- Existing tests use unittest and mocks; baseline execution still pending.
- `requirements.txt` has pre-existing user edits removing pins. Preserve intent; reconcile dependencies explicitly.
- External samples directory is separate Git repo containing `debug_omxs30`. Preserve existing files and history.

## Interface

Proposed builder choices, pending approval:

```bash
ml-toolbox run --config /path/to/samples/grades/config.toml
ml-toolbox prepare --config /path/to/samples/grades/config.toml
ml-toolbox train --config /path/to/samples/grades/config.toml
ml-toolbox evaluate --config /path/to/samples/grades/config.toml
ml-toolbox predict --model /path/to/model.joblib --input new.csv --output predictions.csv
```

One TOML config per task: input CSV, target, task type, optional preparation script, excluded columns, split strategy, seed, model/search budget, output directory. Paths relative to config; no machine-specific defaults. Errors identify field/path and remedy. CLI uses argparse; config uses standard-library tomllib.

Generic loader invokes explicit trusted local Python preparation function `prepare(frame) -> frame`. Dataset-specific feature/label generation lives in `src/data_generation`; external custom script paths supported. Ordinary CSV needs no custom script. Hooks perform deterministic row-level preparation; learned transforms belong inside training pipeline.

Sample folders contain config, runnable step script, provenance/readme, raw/prepared data, results. External samples repo becomes primary execution location; main `samples/` retains portable configs/instructions. Exact migration mapping recorded.

## Shared pipeline

- Retain existing stage boundaries; consolidate shared utilities after caller inventory. Remove superseded code only after behavior mapping/tests. No parallel replacement that leaves old functionality unexplained.
- Validate columns, target, numeric values, missing labels, task, split sizes, hook result. Preserve row identity for predictions.
- Split before fitting imputation, encoding, scaling, feature selection, resampling. All learned transforms inside sklearn Pipeline/ColumnTransformer; unseen categories supported.
- Stratified split/CV for classification; random split/KFold for independent regression; chronological holdout/TimeSeriesSplit for temporal data. Validate class counts; no silent fallback that invalidates evaluation.
- Compare dummy baseline and small task-appropriate model set. Cover existing SVM/XGBoost capabilities plus linear/tree baseline. Deterministic, bounded search on training folds; holdout never selects model or threshold.
- Save full fitted pipeline, feature schema, configuration, dependency versions, seed, source hash and split metadata. Inference reuses fitted preprocessing.
- Classification: accuracy, balanced accuracy, macro F1, per-class report, confusion matrix; ROC/PR only when labels and scores support them.
- Regression: MAE, RMSE, R², actual-vs-predicted and residual plots.
- Both: held-out prediction CSV, metrics JSON, model-comparison CSV, data-summary plots, readable HTML report, logs. Headless plotting; undefined metrics explicitly reported.
- Unique run directories prevent accidental overwrite. Training artifacts used for holdout scoring remain distinguishable from any full-data refit.

## Existing functionality and migration

Inventory generation, cleaning, analysis, feature selection, tuning, threshold selection, training, temporal inference, postprocessing and backtesting. Each capability gets migrated implementation/tests or explicit documented replacement. Financial-specific steps remain opt-in; arbitrary CSV users need only shared pipeline.

Copy raw data first; verify hashes before recording migration success. Never delete originals or overwrite conflicting destination files. Do not copy obsolete model pickles as usable new-runtime models; retain as historical artifacts and retrain. Keep external repo changes separately identifiable. Migration guide maps old scripts/INI sections to new commands/config fields; report ambiguous legacy mappings rather than invent targets.

## Grades

Run regression predicting `GPA`, excluding `StudentID` and `GradeClass`; also classification predicting `GradeClass`, excluding `StudentID` and `GPA`. Verify dataset semantics before declaring leakage-free evaluation; grade-derived features must not enter predictors. Report baseline comparison honestly; no promised accuracy target.

## Five additional datasets

Preferred candidates, subject to verified source, terms and download availability:

1. UCI Bank Marketing — subscription classification; exclude post-contact duration for pre-contact prediction.
2. UCI Default of Credit Card Clients — default classification.
3. UCI Statlog German Credit — credit-risk classification.
4. UCI Real Estate Valuation — property-price regression.
5. California Housing — median-house-value regression.

Use public unauthenticated sources where possible; Kaggle optional. Record original URL, attribution/license, retrieval date, checksum, target, exclusions, split rationale and caveats. Preserve downloaded originals; convert to CSV when necessary. If source/license blocks use, document replacement with same task count and financial preference. Download must be explicit, never triggered by unit tests.

## Runtime and packaging

Verify current stable CPython releases and actual dependency wheels before choosing exact version. Target newest stable version supported by required functionality; avoid prereleases. Document optional-library constraints instead of claiming unsupported compatibility. Installable pyproject package and console entry point; minimal core dependencies, optional finance/visualization extras where warranted. Replace obsolete environment recipes with verified commands and reproducible dependency resolution. No global environment mutation.

## Tests and acceptance

- Characterization tests for current meaningful behavior before migration; distinguish existing defects from intended semantics.
- Unit tests: configuration, preparation hook, cleaning, split, feature alignment, metrics, artifact paths, migration conflicts.
- Integration tests: binary/multiclass classification, regression, mixed categorical/numeric inputs, missing data, unseen categories, saved-model prediction, temporal split, CLI failures.
- Leakage checks: holdout never fits learned preprocessing or selects model; grades excludes alternate outcome.
- Run all migrated functionality tests and smoke-check installed CLI in isolated environment.
- Execute grades tasks and all five downloaded datasets through same shared CLI; record actual commands, metrics, graphs and failures. Compare dummy baselines without asserting guaranteed improvement.
- Report measured coverage and uncovered legacy functionality; no unverified claim of complete modernization.

## Scope limits

No GUI, cloud orchestration, tracking server, abstract plugin framework or speculative features. Configuration plus one optional preparation function sufficient. Existing operational behavior remains in scope; old CLI compatibility explicitly waived by user.

## Next gate

Independent builder review → user design approval → detailed implementation plan → implementation and verification.
