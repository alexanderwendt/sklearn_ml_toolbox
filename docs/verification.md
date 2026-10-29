# Verification — 2026-10-25

## Environment

- Isolated `.verify-venv`: CPython **3.15.1**, latest stable listed by python.org when checked.
- NumPy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, joblib 1.6.0.
- Optional XGBoost 3.4.1, Backtesting 0.6.6, analysis dependencies installed. Resolver snapshot: `requirements-lock.txt`.
- Global Python unchanged. `.venv` bootstraps uv; `.python` stores local managed runtime. Main verified interpreter is `.verify-venv/Scripts/python`.

## Checks

| Check | Observed result |
|---|---|
| Original test collection | Seven errors: broken imports / unavailable old dependencies |
| `.verify-venv/Scripts/python -m pytest --cov=src --cov-report=json --cov-report=term` | **50 passed**, 24 sklearn warnings from deliberately constant/perfectly separating feature-selection fixture |
| New core modules coverage | **87.7%**, 479/546 executable lines |
| All `src` coverage including retained experimental code | **33.1%**; specialist legacy functions largely uncovered |
| `.venv/Scripts/uv pip check --python .verify-venv/Scripts/python.exe` | All 56 installed packages compatible |
| `.verify-venv/Scripts/python -m build` | sdist and wheel built |
| Fresh `.wheel-venv`, core-only wheel installation | CLI help and full Automobile run succeeded outside repository cwd |
| Independent whole-change review | Four Important findings reproduced RED then fixed GREEN; one Minor deferred |
| Windows/Linux CI | Workflow added; remote CI not executed locally |
| Conda environment recipes | Updated, not instantiated |

Core coverage includes config, CLI, workflow, preparation hooks/indicators, preprocessing, training/evaluation, inference and temporal utility. Coverage includes subprocess CLI runs; it is not a claim of exhaustive behavioral verification.

## Actual sample results

Same generic pipeline, seed 42, 80/20 holdout, three training folds, linear/forest candidates, three parameter samples per candidate. Baselines use training data only. Classification comparison: macro F1; regression comparison: RMSE.

| Dataset | Task | Main held-out metrics | Dummy comparison |
|---|---|---|---|
| Grades | Regression | MAE 0.1551; RMSE **0.1964**; R² **0.9534** | RMSE 0.9097 |
| Grades | Classification | Accuracy **0.7223**; macro F1 **0.5497** | macro F1 0.1346 |
| Bank Marketing | Classification | Accuracy 0.8966; macro F1 **0.6320** | macro F1 0.4689 |
| Credit Default | Classification | Accuracy 0.8178; macro F1 **0.6781** | macro F1 0.4378 |
| German Credit | Classification | Accuracy 0.7800; macro F1 **0.7210** | macro F1 0.4118 |
| Real Estate | Regression | RMSE **5.4609**; R² **0.8222** | RMSE 13.1137 |
| Automobile | Regression | RMSE **2351.0246**; R² **0.9548** | RMSE 11488.7134 |

Grades excludes alternate outcome (`GradeClass` for GPA regression, GPA for GradeClass classification), plus StudentID. Financial/economic datasets are IID educational benchmarks; scores do not establish future-cohort or trading performance.

## Result locations

Root: `D:/210_Projects/23_Machine_Learning_Toolbox/sklearn_ml_toolbox_samples`.

- `grades/results/regression/20261025T190749-2eaf6a36/report.html`
- `grades/results/classification/20261025T190754-c6cee7ae/report.html`
- `bank_marketing/results/classification/20261025T190640-fe76173a/report.html`
- `credit_default/results/classification/20261025T190712-a73da365/report.html`
- `german_credit/results/classification/20261025T190746-ec10499f/report.html`
- `real_estate/results/regression/20261025T190757-3be40036/report.html`
- `automobile/results/regression/20261025T190638-bfe4ba3f/report.html`
- Fresh-wheel Automobile run: `automobile/results/regression/20261025T190857-af3bff2c/report.html`.

External `sample_results.json` lists seven final shared-run results. Every run includes predictions, metrics, graphs, full preprocessing/model, manifest and log. Previous runs preserved.

## Review fixes

1. Temporal data sorted before preparation; mutating holdout prices no longer changes causal training features.
2. Prepared CSV `_row_id` preserved through prediction; duplicate/missing IDs rejected.
3. Hooks must preserve index and internal identity column; reverse/reset-index rejected.
4. HTTP Content-Length checked before publishing downloads; truncated body rejected.

Each reproducer failed before its fix; full 50-test suite passed afterward.

## Rulings I made

- Work in current `gemini-refactor` branch, per user choice; no worktree isolation.
- Leave changes uncommitted pending authorization; recovery relies on current diff/ledger rather than commits.
- Interleave legacy repairs with new pipeline because baseline tests did not collect; some historical workflows remain unexecuted.
- Use `.verify-venv` for latest patch runtime, keeping bootstrap `.venv`; activation instructions must distinguish them.
- Replace incompatible pandas-ta with consumed pandas indicators; numerical/warmup differences require regenerated features and retraining.
- Replace California Housing with UCI Automobile price regression for explicit verified licensing; five new datasets still comprise three classification and two regression tasks.
- Temporal utilities accept explicit aligned CSV input/output rather than infer joins from run directories; callers must supply chronological aligned rows.
- Prediction preparation permits missing target; custom hooks must work on unlabeled input.
- Root `utils/` had only bytecode, not duplicate source; source consolidation there unnecessary.

## Deferred minors / unverified scope

- Re-running cached dataset acquisition on a later calendar day conflicts with existing `provenance.json` retrieval date. Existing downloaded datasets remain usable; run training directly. Downloader resume-date fix deferred.
- Specialist legacy INI pipelines, image tools and nonlinear decision-boundary optimizer not fully exercised. See `capabilities.md`; full-project coverage is intentionally reported rather than hidden.
- `.run/*.xml` changed externally during execution; preserved without modification/review. Pre-existing external `_old_data` and historical scripts also preserved.
