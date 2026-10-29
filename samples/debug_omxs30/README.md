# OMX S30 workflow

Activate `sklearn315` and run from this directory:

```powershell
python run.py
```

Or from the repository root:

```powershell
conda run --no-capture-output -n sklearn315 python samples/debug_omxs30/run.py
```

`workflow.toml` is the configuration source for this runner. It merges shared
defaults, preparation/training/inference profiles, and small model overrides.
The shared runner calls the same numbered step modules, with the same settings
and ordering as the verified batch workflow. Superseded batch/shell scripts, job
templates, and hand-maintained INI configs have been removed. Use `run.py` for all
new runs. Historical copies remain inside the preserved run folders.

The dataset root now contains only `run.py`, `workflow.toml`, this README,
`annotations/`, the two required CSV inputs and provenance in `data_raw/`, and
`_old_reruns`. Generated outputs belong exclusively inside a run folder.

## Every new execution has its own folder

```text
_reruns/<UTC timestamp>_<unique id>/
    workflow.toml          # frozen workflow config
    manifest.json          # full plan, progress, attempts, checksums
    environment.txt        # installed package versions
    logs/                  # one log per attempted step
    runtime/models/        # snapshot of legacy search definitions
    workspace/
        annotations/       # copied inputs
        data_raw/
        config/            # generated INI files (do not edit)
        prepared-data/
        models/
        results/
```

No previous prepared data or model is reused in a fresh run. Only an explicit
`--resume` continues an existing run. A failed step stops execution; its log path
is printed. Resume checks the configuration and input snapshots and skips completed
steps. Step implementation code comes from the active installation, allowing a
compatibility fix before resuming; package versions are recorded at run creation.

## Controls

```powershell
python run.py --dry-run
python run.py --models svm,knn
python run.py --until prepare.split
python run.py --resume _reruns/<run-folder>
python run.py --resume _reruns/<run-folder> --until svm.evaluate_validation
```

`--until` executes all preceding steps, then pauses. `--dry-run` lists exact step
IDs without creating files. A resume retains the original model list. New run IDs
include a unique suffix, so launches in the same second cannot share outputs.

To override the arguments of a particular step, add its exact ID to the TOML:

```toml
[steps."svm.wide"]
args = ["--execute_wide=True"] # replaces default arguments; removes -debug
```

Model tables override INI sections in the training profile; inference automatically
uses each model's identity. Keep generated configs with the run for diagnosis.
This adapter currently preserves the OMX S30 numbered workflow; adapting dataset
preparation and regression behavior is a later step. No other dataset is migrated.

## Completed runs and baselines

The initial verified five-model run is preserved at
`_old_reruns`.

The first full run through the shared runner completed all 78 steps:
`_old_reruns`.
Its `migration-comparison/` contains the comparison with the initial verified run;
all 19 prepared-data artifacts are byte-identical. Model and result files can differ
between training runs; the comparison records those differences.

The November 2022 baseline is preserved at `_old_reruns`.
The older `_old_data` archive was moved to `_old_reruns`.
Duplicate root-level outputs and superseded partial verification attempts were removed.
