# Migration

## Interface mapping

| Old stage | New interface |
|---|---|
| step20/21 label/features | CSV + optional `data.prepare` hook |
| step22 alignment | Hook preserves row index; loader validates |
| step30/31 cleaning/conversion | Loader validation + pipeline imputation/encoding |
| step32/33/34 analysis | Automatic report; `analyze-temporal`; specialist modules retained |
| step35 feature selection | `training.select_percentile` fitted inside CV |
| step36 split | `split` section; random/temporal |
| step42 timing | `comparison.csv` model training/search durations |
| step43/44 search | `training.models`, `folds`, `search_iterations` |
| step45 threshold | `training.tune_threshold` on training folds |
| step50 train | `ml-toolbox train` |
| step60/61 evaluate | `ml-toolbox evaluate`; appropriate task metrics |
| step70 inference | `ml-toolbox predict` |
| step71/72 | `postprocess`, `backtest`; historical strategy retained |

INI `[Common].class_name` → `data.target`; `problem_type` → `data.task`; source path → `data.path`; `[Preparation].test_size` → `split.test_size`. Results/models paths collapse into `output.directory/<unique run>`. Explicit task required: numeric labels cannot reliably distinguish classification from regression.

Old model pickles are historical artifacts, not compatible new-runtime models. Retrain. Existing INI files/batch templates are historical examples; no automatic conversion of arbitrary user expressions, custom model classes or ambiguous target names. Grades' stock-market target was corrected explicitly in new configs.

## Data migration

```bash
python scripts/migrate_samples.py samples/grades/data_raw ../sklearn_ml_toolbox_samples/grades/data_raw
```

Copies verified by SHA256; matching files reused, differing destinations rejected. Originals preserved. Destination receives `migration_manifest.json`. Copy entire raw-data directories similarly; do not treat old trained models as migrated usable models.

External `debug_omxs30` remains historical source. New `omxs30/data_raw` copy preserves source filenames and records hashes; label annotations remain separate until domain-specific merge is chosen. Existing untracked `_old_data` and old scripts untouched.

## Compatibility changes

- Python 3.15.1+, installable package `sklearn_ml_toolbox`.
- Legacy source modules now use relative package imports; invoke as `python -m sklearn_ml_toolbox.<stage>.<module>` rather than executing source file paths. New projects should use shared CLI.
- Removed obsolete scikit-plot dependency; precision/recall and ROC use sklearn displays.
- Removed incompatible pandas-ta dependency; consumed SMA/RSI/stochastic/MACD indicators implemented with pandas. Warmup values remain missing; flat RSI/stochastic neutral. Numerical changes require regenerated features/retraining.
- Fixed price-normalization rolling warmup, timestamp-index preservation and flat-price division; fixed simultaneous head/tail trimming and nondefault-index NaN removal.
- `requirements.txt` delegates to pyproject; previous user intent to unpin obsolete dependencies preserved. `requirements-lock.txt` supplies verified resolution separately.
- Conda recipes modernized; uv/venv is verified installation route.

## Limits

Import-safety checks cover all package modules. Not every historical INI workflow, nonlinear decision-boundary visualization, stock strategy or image-conversion path has been executed on real data. Historical model search/helper implementations are retained for migration reference; shared CLI is verified production path. See coverage/capability report before relying on a specialist function.
