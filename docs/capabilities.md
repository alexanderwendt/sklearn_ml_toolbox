# Capability inventory

Shared CSV workflow is modern interface. Historical modules remain import-safe; old CLI compatibility was waived, but source retained for specialist behavior/reference.

| Capability / existing source | Current implementation | Evidence |
|---|---|---|
| Ground truth step20, annotation step20 | Retained domain functions + custom hook contract | top/bottom signal merge test; annotation real-data workflow unexecuted |
| Full/reduced step21 financial features | Retained; pandas indicators replace pandas-ta | causal indicator tests, normalization tests; not every legacy feature combination covered |
| step22 alignment/NaN cleanup | Corrected helpers + loader row identity | trim both ends, NaN index, hook reorder tests |
| step30 cleaning | Validation in loader; legacy cleaning helper retained | constant-column helper, missing target/infinity validation |
| step31 conversion | `feature_frame` + ColumnTransformer | mixed numeric/category/missing values tests; old conversion characterization |
| step32 embedding search | Retained specialist module | import smoke only |
| step33 EDA | New summary/correlation/missingness graphs; old PCA/UMAP/t-SNE retained | all real sample reports; specialist embeddings unexecuted |
| step34 temporal EDA | `analyze-temporal`; old lag/correlation module retained | import smoke; temporal report command available |
| step35 selection | CV-internal SelectPercentile; old Lasso/tree/RFE helpers retained | selection inside fitted pipeline; specialist selectors unexecuted |
| step36 split | Shared random/stratified/chronological splitter | alignment, class counts, timestamp groups |
| step42 training time | Per-model durations in comparison.csv | seven real runs |
| step43/44 search | Bounded training-fold randomized search; linear/forest/SVM/XGBoost | linear/forest real samples; model variants smoke-tested separately |
| step45 precision/recall threshold | TunedThresholdClassifierCV using training CV | binary threshold test |
| step50 final training | Complete fitted preprocessing/model artifact | saved-model predictions match roundtrip |
| step60/61 evaluation | Held-out task-specific metrics, baseline, graphs | known-value regression metrics, multiclass CLI, real binary samples |
| step70 prediction | Schema-aware persisted pipeline inference | reordered/missing fields and novel categories |
| step71 postprocessing | Explicit threshold command; historical event logic retained | threshold CSV and event-forward-fill tests |
| step72 backtesting | Explicit next-bar close-to-close benchmark; historical Backtesting.py strategy retained | signal timing and turnover cost tests; old strategy not run |
| utils/filepaths | Legacy paths packaged; new config-relative paths replace for shared CLI | import smoke + config cwd tests |
| utils/decisionboundaryplot | Historical advanced optimizer; nlopt imported on demand | import smoke only; optimizer requires nlopt |
| root hyperparam.py | Historical interface retained | no active shared workflow dependency |
| root image/time-series utilities and `_old` | Retained historical tools | not covered by modern pipeline tests |

## Baseline

Original seven tests could not collect: broken `utils` imports and missing old dependencies. Price-normalization test compared its own reimplementation rather than production output; split/train tests asserted mock calls. Replaced those with actual normalization, split-alignment and persisted-model checks. Existing successful semantic tests retained.

Full modernization of every experimental legacy branch is **not claimed**. Verified shared pipeline, samples, packaging and selected legacy fixes are distinguished from retained specialist source above.
