# Student grades: complete analysis workflow

Run from this directory with the `sklearn315` environment active:

```powershell
python run.py
```

Or from the repository root:

```powershell
conda run --no-capture-output -n sklearn315 python samples/grades/run.py
```

One command runs both tasks using `workflow.toml`:

| Task | Target | Models |
| --- | --- | --- |
| Classification | GradeClass | SVM, XGBoost, linear SVM, KNN, Gaussian NB (same families and order as OMX S30) |
| Regression | GPA | Ridge, random forest, SVR, linear SVR, KNN, XGBoost |

Both include a dummy baseline, an 80/20 split, five-fold training cross-validation,
and up to ten hyperparameter candidates per model (bounded by the search space).
StudentID and the alternate outcome are excluded. The `builtin:grades` preparation
hook treats demographic/support codes as categories. Preprocessing is fitted
inside training folds; holdout results do not select the winning model.

## One folder per execution

Every fresh invocation creates `_reruns/<UTC timestamp>_<unique id>/`:

```text
workflow.toml                   # original configuration snapshot
manifest.json                   # plan, progress, checksums, attempt history
environment.txt                 # package versions
logs/                           # separate stdout/stderr log per stage
workspace/
    data_raw/                   # immutable input snapshot
    config/
        classification.toml     # generated effective task configurations
        regression.toml
    results/
        report.html             # START HERE: links to both complete reports
        summary.csv             # selected-model metrics per task
        classification/
            report.html
            prepared.csv, train.csv, holdout.csv, split.joblib
            comparison.csv, model_metrics.csv, metrics.json
            model.joblib        # selected pipeline and all fitted candidates
            graphs/
            analysis/           # PCA, UMAP, t-SNE and exploratory feature rankings
            models/<model>/     # per-model train/holdout diagnostics and importance
            inference/holdout_predictions.csv
        regression/             # same layout, regression-specific diagnostics
```

Reports contain:

- Feature distributions, missing-value counts, target relationships, and correlations.
- Cross-validation scores, selected parameters, and search/fit times.
- Training and holdout metrics, predictions, and permutation importance for every model.
- Classification: confusion matrices, per-class precision/recall/F1/support,
  and one-vs-rest ROC/precision–recall curves where scores are available.
- Regression: MAE, RMSE, R², prediction plots, and residual diagnostics.
- Predictions loaded from the saved winning pipeline on the holdout CSV. These
  exercise saved-model inference; they are not a separate unseen student dataset.

## Dimensionality reduction and feature diagnostics

Both tasks now include an `analyze` stage between preparation and training.
The classification model list is exactly the five OMX S30 model families, plus
the shared dummy baseline. Tabular preprocessing and the bounded tabular search
remain appropriate to grades; this does not import the stock-market feature
engineering, binary threshold tuning, or trading backtests.

The analysis stage uses a reproducible sample of at most 1,000 **training** rows.
It imputes and one-hot encodes categories, omits constant predictors, and standardizes
the encoded features. Holdout rows are excluded. Each method emits PNG graphs and
CSV values in `analysis/`, all included in the final HTML report:

| Method | Main outputs |
| --- | --- |
| PCA | `PCA_Plot`, `PCA_Variance_Coverage`, `PCA_Heatmap`, signed loadings |
| t-SNE | `T-SNE_Plot` and coordinates |
| UMAP | `UMAP_Unsupervised`, `UMAP_Supervised` and coordinates |
| LassoCV | `Lasso_Model_Weights` with coefficients, selected features, and alpha |
| ExtraTrees | `Tree_Based_Importance` and selection at importance > 0.01 |
| Recursive elimination | `Recursive_Feature_Ranking` |
| Backward OLS elimination | `Backward_Elimination`, selected features and p-values |
| Correlation / hierarchy | Pearson and Spearman matrices; Ward feature dendrogram |

Lasso and backward OLS analyze one-vs-rest class indicators for classification,
and GPA directly for regression. Classification labels are not treated as numeric
distances. Supervised UMAP uses training targets (categorical for classification,
continuous for regression); apparent separation is not an accuracy measurement.
These projections and feature lists are exploratory and do not silently alter
the model-training features. The fitted analysis preprocessing and selected row
IDs are saved for reproducibility.

Control analysis in `workflow.toml`:

```toml
[defaults.analysis]
methods = ["pca", "tsne", "umap", "lasso", "tree", "recursive", "backward", "correlation"]
max_samples = 1000
```

Use `methods = []` to disable this stage, or override it in
`[profiles.classification.analysis]` / `[profiles.regression.analysis]`.
Dependencies are declared in the existing extras: `pip install -e ".[finance,analysis]"`.

## Select tasks, models, or stages

```powershell
python run.py --dry-run
python run.py --tasks classification
python run.py --tasks regression
python run.py --models linear,forest
python run.py --tasks classification --models gnb,knn
python run.py --until classification.prepare
python run.py --tasks classification --until classification.analyze
python run.py --resume _reruns/<run-folder>
python run.py --resume _reruns/<run-folder> --until regression.train
```

`--until` includes preceding steps. The full workflow now has 11 stages.
Task order is classification then regression,
followed by the combined report. Resume retains the original task/model selection
and skips completed steps. Each task stage commits its outputs only after success,
so a failed evaluation can be retried without damaging the trained model.
Logs preserve each attempt. Input/config snapshots are checked before resuming.

The shared `ml-toolbox workflow` runner now supports both the OMX legacy-step
engine and this tabular engine. The tabular engine reuses the existing preparation,
training, evaluation, reporting, and inference implementations. Financial
backtesting and market-specific temporal indicators are not grades analysis steps.

The older `config*.toml`, `config/*.ini`, and `data_prepared/` files are retained as
historical material and for the standalone legacy diagnostics helper. The new
runner does not read them. Edit only `workflow.toml` for new runs; generated configs
belong to their run. The root `test_regression.py` helper also uses the isolated
workflow with the regression task selected.

## Interpretation and provenance

The first complete run of this workflow finished all nine stages successfully:
[`_reruns/20261029T195724Z_8147fb19/workspace/results/report.html`](_reruns/20261029T195724Z_8147fb19/workspace/results/report.html).
It generated 358 classification graphs and 111 regression graphs. The selected
models were XGBoost for GradeClass (holdout accuracy 0.6889, macro F1 0.5430) and
Ridge for GPA (holdout RMSE 0.1964, R² 0.9534).

Training scores are in-sample diagnostics. Holdout metrics and permutation
importance are for evaluation, not tuning; importance is predictive, not causal.
Classification and regression metrics are different quantities, not a common ranking.

Original CSV supplied with this repository; its redistribution license has not
been independently established. Raw data is preserved. External copies under
`../sklearn_ml_toolbox_samples/grades` are not modified or synchronized automatically.
