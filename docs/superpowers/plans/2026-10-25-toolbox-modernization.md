# Toolbox Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize existing toolbox into tested CSV-to-results classification/regression workflow; migrate data and run grades plus five public datasets.

**Architecture:** Keep generation/processing/modeling/prediction boundaries, package existing `src` modules as `sklearn_ml_toolbox` using setuptools package mapping. Shared stage functions power CLI and sample scripts. Dataset preparation hook stays separate from learned preprocessing.

**Tech Stack:** Newest verified stable CPython compatible with required dependencies; pandas, NumPy, scikit-learn, matplotlib, joblib, argparse, tomllib, pytest; optional XGBoost/finance/EDA dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-toolbox-modernization-design.md`

## Global Constraints

- Replace old commands/configs with cleaner interface; migrate data.
- Copy raw data first; verify hashes before recording migration success.
- Never delete originals or overwrite conflicting destination files.
- Paths relative to config; no machine-specific defaults.
- Holdout never selects model or threshold.
- Download must be explicit, never triggered by unit tests.
- No global environment mutation.
- Preserve pre-existing edits to `requirements.txt`; reconcile rather than restore old pins.
- External samples repo: `D:/210_Projects/23_Machine_Learning_Toolbox/sklearn_ml_toolbox_samples`; inspect its instructions/status before edits.
- Existing meaningful behavior stays in scope: label/feature generation, cleaning, EDA, selection, search, training, threshold tuning, temporal prediction/postprocessing/backtesting.
- Default shell Python currently 3.8.5; never mistake it for modern verification runtime.

## Review Focus

- Paths containing spaces/non-ASCII; config-relative resolution independent of current directory (Task 2).
- Rare classes and tiny/constant targets; explicit validation or undefined metric reasons, never misleading plots (Tasks 3/4).
- Reordered/missing input columns and novel categories at prediction; reorder valid schema, reject missing features (Task 4).
- Hook removes/reorders rows; preserve unique row identity and target alignment (Task 2).
- Interrupted downloads/copy conflicts; atomic writes and unchanged existing destination (Task 6).

## File Structure

- `pyproject.toml`, `src/__init__.py`, stage `__init__.py` files: installable package; mapped package root `src`.
- `src/config.py`, `src/cli.py`, `src/workflow.py`: validated config, CLI, stage orchestration.
- `src/data_generation/loader.py`, `grades.py`: generic CSV/hook loader and grades-specific preparation.
- `src/data_processing/preprocessing.py`, `analysis.py`: split/transform construction and reusable EDA.
- `src/modeling/training.py`, `evaluation.py`: model search, persisted pipeline, task-specific results.
- `src/prediction/inference.py`: saved-model schema validation and predictions.
- Existing `src/data_generation/step*.py`, `src/prediction/step*.py`, `src/utils/*.py`: migrate tested domain functions/imports; explicit replacement mapping before removals.
- `scripts/migrate_samples.py`, `scripts/download_samples.py`, `scripts/run_samples.py`: copy-first migration, source acquisition, common sample runner.
- `tests/unit/`, `tests/integration/`, existing tests: offline semantic tests; integration uses small synthetic CSVs.
- `samples/<dataset>/`: portable configs, runner, source notes; external matching folders: raw data and verified outputs.
- `docs/migration.md`, `docs/verification.md`, `docs/capabilities.md`, `README.md`: mapping, evidence, retained behaviors, usage.

### Task 1: Runtime, package and characterization baseline

**Files:** Create `pyproject.toml`, `src/__init__.py`, stage `__init__.py`, `docs/capabilities.md`, `tests/unit/test_imports.py`; modify requirements/environment recipes and existing tests only when their setup is invalid.

**Interfaces:** Produces importable `sklearn_ml_toolbox` package; isolated `.venv` interpreter used by every following command. Package mapping `sklearn_ml_toolbox = src`; child packages explicitly included.

- [ ] Inventory tracked files, callers, legacy scripts, optional dependencies and external repo status/instructions; fill capability table with old entry point, intended behavior, replacement/test. Correct design note: root `utils/` contains only bytecode, not duplicate source.
- [ ] Inspect installed interpreters/tooling and official CPython/PyPI metadata; select latest stable supported runtime using actual dependency resolution/wheels. Create isolated environment, record exact interpreter and dependency versions. No speculative version floor.
- [ ] Record baseline test collection/execution failures before fixes; avoid root `test_regression.py` import executing subprocesses. Characterize pure generation/cleaning functions with small deterministic inputs from existing tests.
- [ ] Add `test_imports_do_not_parse_arguments_or_write_files`: subprocess imports modules with an unrelated CLI argument from empty working directory; assert exit zero and no created artifacts. Run `.venv/Scripts/python -m pytest tests/unit/test_imports.py -q`; expect failure before import repairs.
- [ ] Add packaging, repair absolute imports, move CLI parsing and side effects into guarded main functions. Move optional heavy imports to relevant functions. Install editable package and necessary test dependencies.
- [ ] Run import and existing characterization tests; record actual remaining behavior defects, never bless them as expected outputs. Run `python -m pip check` with isolated interpreter.
- [ ] Review scoped diff; commit this independently testable change when commits are authorized.

### Task 2: Config and custom preparation contract

**Files:** Create `src/config.py`, `src/data_generation/loader.py`, `src/data_generation/grades.py`, `tests/unit/test_config.py`, `tests/unit/test_preparation.py`.

**Interfaces:** `load_config(path: Path) -> Config`; `Config` stores resolved paths, task, target, excluded columns, split, seed, models, search budget. `prepare_dataset(config: Config) -> pd.DataFrame`; trusted local hook `prepare(frame: pd.DataFrame) -> pd.DataFrame`.

- [ ] Add tests `test_config_paths_relative_to_file`, `test_unknown_fields_rejected`, `test_invalid_task_and_missing_target`, `test_hook_preserves_row_identity`, `test_csv_without_hook`, `test_grades_excludes_alternate_outcome`. Assert config works from different cwd with spaces/Unicode; invalid config identifies field; hook reorder retains IDs; duplicate/missing IDs rejected; classification drops GPA, regression drops GradeClass, both drop StudentID.
- [ ] Run `.venv/Scripts/python -m pytest tests/unit/test_config.py tests/unit/test_preparation.py -q`; expect missing-module failure.
- [ ] Implement TOML fields: `data.path`, `data.target`, `data.task`, optional `data.prepare`, `data.drop`, `data.delimiter`; `split.strategy`, `split.test_size`, `split.time_column`; `training.seed`, `training.models`, `training.folds`, `training.search_iterations`; `output.directory`. Defaults: test_size 0.2, seed 42, folds 3, search_iterations 10. Task and target explicit. Reject unknown keys.
- [ ] Implement CSV read, unique internal row IDs, hook loading, target/type/infinity validation, exclusions. Missing labels error; missing feature values allowed. Hooks may subset/reorder rows but cannot lose identity. Grades hook normalizes known categorical fields; target exclusion uses config.
- [ ] Run new tests and characterization tests; commit scoped change when authorized.

### Task 3: Leakage-safe splitting, preprocessing and model search

**Files:** Create `src/data_processing/preprocessing.py`, `src/modeling/training.py`, `tests/unit/test_training.py`, `tests/integration/test_training_tasks.py`.

**Interfaces:** `split_dataset(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]`; `train_model(train: pd.DataFrame, config: Config) -> dict` returns `pipeline`, `features`, `task`, `target`, `cv_results`, `seed`, `versions`. Persist dict with joblib; never deserialize untrusted files.

- [ ] Add tests `test_holdout_never_fits_preprocessor`, `test_temporal_split_orders_and_separates_timestamps`, `test_rare_class_fails_before_search`, `test_regression_does_not_stratify`, `test_mixed_features_and_missing_values`, `test_selection_and_resampling_fit_inside_cv`. Assert held-out extreme does not alter fitted scaler; unseen category predicts; split row IDs disjoint; chronological folds never train on future data.
- [ ] Run `.venv/Scripts/python -m pytest tests/unit/test_training.py tests/integration/test_training_tasks.py -q`; expect failure.
- [ ] Implement stratified/random/temporal split and matching CV; prohibit overlapping timestamp groups. Impute/encode/scale inside ColumnTransformer/Pipeline; optional selectors and imbalanced resampling also inside folds. Reject impossible class/fold counts and invalid regression targets.
- [ ] Implement dummy plus linear, random forest and SVM candidates for both tasks; optional XGBoost. Bounded parameter search; macro F1 classification, negative RMSE regression. Dummy excluded from parameter tuning but included in comparison. Record CV scores/time; selection uses training folds only.
- [ ] Migrate wide/narrow search and feature-selection capabilities to explicit config/options or tested reusable helpers; maintain capability table. Threshold tuning consumes training out-of-fold scores only.
- [ ] Run tests, including binary/multiclass/regression integration; commit scoped change when authorized.

### Task 4: Results, persisted inference and shared CLI

**Files:** Create `src/modeling/evaluation.py`, `src/data_processing/analysis.py`, `src/prediction/inference.py`, `src/workflow.py`, `src/cli.py`, `tests/unit/test_evaluation.py`, `tests/integration/test_cli.py`; modify `pyproject.toml` entry point.

**Interfaces:** `evaluate_model(model: dict, holdout: pd.DataFrame, output: Path) -> dict`; `predict_csv(model_path: Path, input_path: Path, output_path: Path) -> Path`; `run_pipeline(config_path: Path) -> Path`; `main(argv: list[str] | None = None) -> int`.

- [ ] Add tests `test_regression_metrics_match_known_values`, `test_multiclass_metrics_and_graphs`, `test_constant_target_metric_reason`, `test_saved_model_predictions_equal_original`, `test_inference_reorders_columns_and_rejects_missing`, `test_cli_run_from_other_directory`, `test_staged_run_matches_full_run`, `test_run_never_overwrites_previous_outputs`. Assert JSON contains no NaN; metric exclusions have reasons; report escapes user column labels.
- [ ] Run `.venv/Scripts/python -m pytest tests/unit/test_evaluation.py tests/integration/test_cli.py -q`; expect failure.
- [ ] Implement metrics/plots per design using headless matplotlib; reuse verified plotting functions. EDA includes missingness, distributions, correlations; optional PCA/UMAP/t-SNE and existing analysis helpers retained/tested where applicable.
- [ ] Implement unique run directories containing prepared/split CSVs, model.joblib, predictions.csv, metrics.json, comparison.csv, report.html, graphs, manifest.json and run.log. Manifest includes input/config hashes, versions, seed and split row IDs.
- [ ] Implement `run`, `prepare`, `train`, `evaluate`, `predict` CLI. Staged commands accept `--run-dir` identifying artifacts; train/evaluate verify manifest/config compatibility. Predict expects prepared features; expose `--config` for replaying deterministic raw-data preparation without requiring target. Errors return nonzero with actionable messages.
- [ ] Run offline full/staged workflow tests and installed `ml-toolbox --help`; commit scoped change when authorized.

### Task 5: Existing financial/domain functionality and interface migration

**Files:** Modify existing generation/prediction/util modules and batch scripts; `filepaths.py`, `hyperparam.py`, root utility scripts; replace root `test_regression.py`; create `tests/unit/test_financial_features.py`, `tests/integration/test_temporal_workflow.py`, `docs/migration.md`.

**Interfaces:** Existing domain helpers become import-safe package functions; dataset preparation uses Task 2 contract. Shared model/evaluation uses Tasks 3/4 interfaces. Finance operations explicit `analyze-temporal`, `postprocess`, `backtest` CLI subcommands with config and run directory.

- [ ] Expand characterization tests for smoothing, moving-average direction, labels, dimension trimming, temporal alignment and signal postprocessing; read every caller before changing helpers. Document any future-aware label generation separately from causal features.
- [ ] Add `test_feature_generation_does_not_use_future_rows` comparing prefix features before/after appending future rows; `test_backtest_signal_applied_next_bar` asserts no same-bar hindsight; `test_temporal_workflow_keeps_prediction_index` asserts timestamps survive model outputs/postprocessing.
- [ ] Run finance/temporal tests to expose defects before repairs.
- [ ] Consolidate redundant feature variants and obsolete plotting dependencies without losing validation/semantics; adapt finance helpers to new config/artifact contracts. Preserve backtest costs/execution timing; test deterministic miniature OHLC dataset with finance extras installed.
- [ ] Finish capability mapping for every script including image conversion and annotation tools; retain standalone import-safe tools where generic runner does not fit. Delete only superseded tracked implementation with tested replacement. Do not delete `_old` or unfamiliar files as cleanup.
- [ ] Replace broken root smoke script and template commands with portable CLI instructions/runners; document old INI-to-TOML mapping, retraining requirement and ambiguous migration errors.
- [ ] Run characterization plus temporal integration tests; commit scoped change when authorized.

### Task 6: Copy-first migration, dataset acquisition and sample configs

**Files:** Create `scripts/migrate_samples.py`, `scripts/download_samples.py`, `scripts/run_samples.py`, `tests/unit/test_sample_tools.py`; create configs/runners/source notes in `samples/{grades,bank_marketing,credit_default,german_credit,real_estate,california_housing}/`; migrate into matching external folders.

**Interfaces:** `copy_verified(source: Path, destination: Path) -> dict` returns source/destination/SHA256; `download_dataset(name: str, destination: Path) -> Path` explicitly acquires and validates CSV; scripts expose guarded argparse mains. Run script invokes installed shared CLI for each config.

- [ ] Add offline tests `test_copy_preserves_original_and_checks_hash`, `test_conflict_does_not_overwrite`, `test_interrupted_download_keeps_existing_file`, `test_archive_traversal_rejected`, `test_sample_configs_have_three_classification_two_regression`. Mock transport, not preparation semantics.
- [ ] Run `.venv/Scripts/python -m pytest tests/unit/test_sample_tools.py -q`; expect failure.
- [ ] Implement atomic destination writes, checksums and migration manifest. Validate archive members before extraction; no secrets or credential requirements. Existing matching hashes are idempotent; conflicts fail with paths.
- [ ] Verify original provider URLs and reuse licenses for five candidates; record sources/terms/retrieval/checksums. Download originals, convert workbook/text formats to CSV where needed using explicit optional dependencies. Source unavailable → documented same-task financial/economic replacement.
- [ ] Copy grades and existing external stock-market data into new layout without replacing originals; produce configs for grades GPA regression and GradeClass classification. Exclude alternate outcomes and identifiers. Bank duration excluded for pre-contact task. Record split limitations for all datasets.
- [ ] Generate per-sample runnable step scripts invoking shared commands plus README; run migration/download smoke checks and offline tests; commit only intended source/config/provenance artifacts when authorized.

### Task 7: Execute samples, complete verification and documentation

**Files:** Update `README.md`, `docs/verification.md`, `docs/capabilities.md`, `docs/migration.md`, environment recipes, `.opencode/project.json`; create `.github/workflows/tests.yml`; outputs in external sample result directories.

**Interfaces:** All commands from Task 4; all sample configs from Task 6. Project profile commands refer to verified installed runtime/test entry points.

- [ ] Run grades regression/classification and all five additional datasets using same installed CLI and bounded budgets. Save actual run directories, metrics, plots and baseline comparisons; investigate failures, rerun only affected checks. Do not promise model beats baseline.
- [ ] Run full offline test suite with coverage, finance-extra tests, `pip check`, build wheel and install it in fresh environment; smoke-test CLI outside repo. Record exact commands/interpreter/results and uncovered capabilities.
- [ ] Configure CI for verified modern runtime on Windows/Linux, offline tests and wheel build; avoid dataset downloads in default tests.
- [ ] Update setup/quickstart, new-CSV walkthrough, custom hook example, sample commands, optional dependencies, metric interpretation and migration guide. Update capability matrix: no unexplained legacy entry or unsupported claim.
- [ ] Set `.opencode/project.json` lane commands to verified commands; record final runtime choice and reproducible dependency snapshot. Remove stale environment guidance only after working replacement verified.
- [ ] Review both repository diffs/status; ensure user edits, raw originals and unrelated external data preserved. Obtain independent whole-change review; fix findings and rerun relevant tests before completion.
- [ ] Commit reviewed scoped changes when authorized; report complete/failed/skipped checks honestly with output paths and remaining risks.

## Self-review

- Spec coverage: workflow/config Task 2/4; leakage/modeling Task 3; existing behavior Task 1/5; migration/datasets Task 6; runtime Task 1/7; complete runs/docs Task 7.
- Interfaces consistent: Config → prepared frame → split → model dict → evaluation/inference; row IDs preserved through stages.
- Five review-focus cases explicitly assigned tests.
- Runtime/dependency versions and source licenses require live verification during execution; no guessed compatibility promises.

## Execution Status

- [x] Design approved by user (`continue`).
- [x] Implementation plan written and self-reviewed.
- [x] User selects execution method: native, current directory.
- [x] Shared pipeline, samples, packaging and local verification complete; see `docs/verification.md`.
- [ ] Exhaustive specialist legacy workflow verification (retained scope gaps listed in `docs/capabilities.md`).
