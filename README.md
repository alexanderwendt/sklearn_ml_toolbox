# SKLearn Machine Learning Toolbox

**CSV → preparation → classification/regression → predictions, metrics and graphs.**

Use one config per dataset. Customize only a preparation function when your CSV needs domain-specific features. Shared processing, model selection, evaluation and prediction need no edits.

## Install

Python **3.15.1 or newer**. Verified on Windows with Python 3.15.1, scikit-learn 1.9.1, pandas 3.1.0 and NumPy 2.5.4.

```bash
uv python install 3.15.1
uv venv --python 3.15.1
uv pip install -e .
```

Activate `.venv` (`source .venv/Scripts/activate` in Windows Git Bash; `source .venv/bin/activate` on Linux/macOS). Without uv, use `python -m venv .venv`, activate it, then `python -m pip install -e .` with Python 3.15.1+.

Optional packages:

```bash
uv pip install -e ".[test,finance,analysis]"
```

`finance`: XGBoost and historical backtesting. `analysis`: historical EDA, UMAP, resampling. Core pipeline needs neither. `requirements-lock.txt` records resolved versions for this Windows verification; install with `uv pip install -r requirements-lock.txt` then `uv pip install -e . --no-deps` for replication. Conda YAML files are alternate recipes, not locally verified environments.

This checkout's verified environment is `.verify-venv`; run `.verify-venv/Scripts/ml-toolbox.exe` directly without activation. `.venv` was used only to bootstrap tools.

## Your CSV

Create `config.toml` beside your CSV:

```toml
[data]
path = "data.csv"
target = "price"
task = "regression" # or "classification"
drop = ["customer_id"]

[training]
models = ["linear", "forest"]
seed = 42
folds = 3
search_iterations = 3

[output]
directory = "results"
```

```bash
ml-toolbox run --config /path/to/config.toml
```

Every path inside a config resolves relative to that config, regardless of working directory. Each run creates a new output folder and prints its path. Open **`report.html`** there.

Outputs: prepared/train/holdout CSVs, held-out predictions, metrics JSON, CV model comparison, trained model, graphs, data summary, config snapshot, source/artifact hashes and run log. Classification includes confusion matrix and binary ROC/PR when defined. Regression includes residual and actual-vs-predicted plots, MAE, RMSE and R². Dummy baseline reported alongside selected model.

## Custom preparation

Add a script under `src/data_generation` or your sample folder:

```python
def prepare(frame):
    frame["income_per_person"] = frame["income"] / frame["household_size"]
    return frame
```

Point config to it:

```toml
[data]
path = "raw.csv"
target = "price"
task = "regression"
prepare = "prepare.py"
```

The hook receives/returns a DataFrame. Preserve its index and reserved `__toolbox_row_id__` column, even when filtering/reordering; the loader removes that column afterward. It must support unlabeled prediction data. Hooks are **trusted executable Python**, not sandboxed plugins. Use only deterministic row-level or causal transformations here. Learned imputation, encoding, scaling and selection belong in the shared pipeline, where they fit only training folds. Validate domain-specific divisors/missing values in your hook.

Built-ins: `builtin:grades` converts categorical codes; `builtin:stockmarket` adds causal trailing returns/mean ratios. Explicitly exclude identifiers and target-derived columns in config. No automatic leakage detector can infer their meaning.

## Run individual steps

```bash
ml-toolbox prepare --config config.toml
# Copy the printed run directory into subsequent commands:
ml-toolbox train --config config.toml --run-dir results/<run>
ml-toolbox evaluate --config config.toml --run-dir results/<run>
ml-toolbox predict --model results/<run>/model.joblib --input new.csv --output predictions.csv
```

Add `--config config.toml` to `predict` when replaying raw-data preparation. Without it, input must contain the prepared feature columns. Predictions preserve source row IDs; column ordering is corrected, missing columns rejected, novel categories accepted. Load **only trusted model/joblib files**: deserialization can execute code.

Reusing an existing trained/evaluated run or prediction output fails rather than overwriting it. Changed config/input/hook/split/model artifacts require a new run.

### Extra controls

- Models: `linear`, `forest`, `svm`, optional `xgboost` (classification labels must be encoded 0..n-1).
- `training.select_percentile = 50`: select within folds.
- `training.balance = true`: oversample classification training folds; requires analysis extra.
- `training.tune_threshold = true`: binary threshold tuning on training folds, optimizing balanced accuracy.
- `split.strategy = "temporal"`, `split.time_column = "Date"`: chronological holdout/CV, never splitting equal timestamps between sides. Default is random, stratified for classification.
- `data.delimiter = ";"`: alternate CSV separator.

For overlapping future-label horizons, purge boundary rows in dataset preparation/design; basic chronological splitting alone does not remove horizon overlap. No financial sample result here constitutes a future-trading backtest.

## Samples

Configs and runnable `run.py` files live in `samples/`. Download five free, attributed UCI datasets to either local samples or a separate directory:

```bash
python scripts/download_samples.py --destination ../sklearn_ml_toolbox_samples
python scripts/run_samples.py --samples ../sklearn_ml_toolbox_samples
# Or one dataset:
ml-toolbox run --config ../sklearn_ml_toolbox_samples/bank_marketing/config.toml
```

| Sample | Task | Target |
|---|---|---|
| grades | Regression + classification | GPA / GradeClass |
| bank_marketing | Classification | Term deposit subscription |
| credit_default | Classification | Credit-card default |
| german_credit | Classification | Credit risk |
| real_estate | Regression | Property price per unit area |
| automobile | Regression | Vehicle price |

Grades is already available locally; run `python samples/grades/run.py`, or add `config_classification.toml` as its argument. Alternate outcome and StudentID are excluded. External copies and all seven completed reports are under `../sklearn_ml_toolbox_samples`; see its `sample_results.json` and [verification report](docs/verification.md).

Downloads are explicit and never part of unit tests. Original CSVs, source metadata, licenses and SHA256 checksums accompany each downloaded sample. Credit/property benchmarks use IID holdouts and may not represent future cohorts, geography or deployment conditions.

## Financial utilities

```bash
ml-toolbox postprocess --input aligned_prices_predictions.csv --output signals.csv --threshold 0.5
ml-toolbox backtest --input signals.csv --output backtest.csv --cost 0.001
ml-toolbox analyze-temporal --input chronological.csv --output temporal.csv
```

Postprocessing thresholds numeric `predicted` into `signal`. Backtest needs `Close` and binary `signal`, chronological rows, and uses the preceding row's signal for the next close-to-close return. It charges turnover costs; it is a long/flat research calculation, not an intrabar trading simulator. Historical specialist scripts remain available as package modules; see capability inventory.

## Tests and migration

```bash
python -m pytest --cov=src
python -m pylint src scripts samples --recursive=y --errors-only
python -m pip check
python -m build
```

Pylint is included in the `test` extra. Configure the IDE interpreter and Pylint
executable to use `.venv` (Python 3.15.1+), rather than an older system or Conda
environment. Generated `_reruns` snapshots are excluded from linting. Omit
`--errors-only` to also see the existing style and refactoring diagnostics.

To refresh all declared libraries to compatible releases and record the result:

```bash
uv pip compile pyproject.toml --all-extras --python .venv/Scripts/python.exe --upgrade --output-file requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe -e . --no-deps
```

Use `.venv/bin/python` on Linux/macOS. The lock records the Windows environment;
rerun the tests and `pip check` after refreshing it. The historical decision-boundary
optimizer additionally requires a working NLopt build for the selected Python and
platform; it is not a dependency of the normal workflows.

[Migration guide](docs/migration.md) · [Capability inventory](docs/capabilities.md) · [Verification evidence](docs/verification.md)
