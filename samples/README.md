# Downloadable dataset workflows

Install the toolbox in a compatible Python environment from the repository root:

```powershell
python -m pip install -e .
python samples/automobile/run.py
```

Each of the following samples downloads its public UCI source when missing,
automatically prepares the data, and runs an isolated training/evaluation workflow.
You can also change into a dataset directory and execute `python run.py`.

| Directory | Task | Target | Prepared rows | Predictors |
|---|---|---|---:|---:|
| `automobile` | Regression | Vehicle price | 201 | 23 |
| `bank_marketing` | Classification | Term-deposit subscription (`yes`/`no`) | 45,211 | 16 |
| `credit_default` | Classification | Next-month default (`0`/`1`) | 30,000 | 23 |
| `german_credit` | Classification | Good/bad credit risk (`1`/`2`) | 1,000 | 20 |
| `real_estate` | Regression | Property price per unit area | 414 | 6 |

Dataset READMEs explain predictor choices, source attribution and preparation.
Preparation is deterministic: missing targets are excluded, IDs/leaking predictors
are removed and categorical codes are preserved. Imputation, one-hot encoding and
scaling are learned within training folds, not from the full downloaded dataset.

## Commands inside a dataset directory

```powershell
python run.py                       # download, prepare, analyze, train and evaluate
python download_data.py             # download and prepare only
python download_data.py --overwrite # refresh source and regenerate inputs
python run.py --models linear       # Ridge or logistic regression only
python run.py --dry-run             # inspect steps without downloading
python run.py --resume <run-folder>  # continue an existing isolated run
```

Source CSVs, prepared CSVs and current checksum/provenance records are in
`data_raw/`. Existing downloads are reused, while prepared inputs are regenerated
on every fresh execution. Reports are saved to
`_reruns/<run>/workspace/results/report.html` within each sample.

`scripts/download_samples.py --destination samples` downloads and prepares all five
without training. Local `download_data.py` scripts share this implementation.
The older `config.toml` files remain usable with `ml-toolbox run --config ...`
after preparation; `run.py` uses `workflow.toml` for isolated, resumable execution.

See the separate instructions in `grades/README.md` and `debug_omxs30/README.md`
for those workflows.
