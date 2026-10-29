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

The dataset root contains `run.py`, `workflow.toml`, this README, dataset-specific
Python code in `steps/`, `annotations/`, and the required CSV inputs and provenance
in `data_raw/`. Generated outputs belong exclusively inside a run folder.

## Every new execution has its own folder

```text
_reruns/<UTC timestamp>_<unique id>/
    workflow.toml          # frozen workflow config
    manifest.json          # full plan, progress, attempts, checksums
    environment.txt        # installed package versions
    logs/                  # one log per attempted step
    runtime/models/        # snapshot of legacy search definitions
    runtime/dataset/steps/ # snapshot of dataset-specific Python code
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
is printed. Resume checks configuration, input, and local-code snapshots and skips
completed steps. Dataset-specific code executes from the run's snapshot: editing
the live `steps/` directory affects new runs only. Shared package steps come from
the active installation; package versions are recorded at run creation.

## Dataset-local Python steps

Dataset-specific implementations are grouped by responsibility outside the general
toolbox package:

```text
steps/
    data_generation/
        step20_generate_groundtruth_stockmarket.py
        step20_generate_groundtruth_stockmarket_from_annotation.py
        step21_generate_features.py
        step21_generate_features_reduced_lt.py
    prediction/
        step70_predict_temporal_data.py
        step71_value_postprocessing.py
        step72_backtesting.py
```

They import shared utilities from the installed `sklearn_ml_toolbox`.
Generic saved-model CSV inference stays in `src/prediction/inference.py`. The
dataset-independent close/signal research helper stays in `src/prediction/temporal.py`;
the OMX signal smoothing, temporal plots, SMA reference strategy, and model-specific
backtests belong to the sample's prediction steps.
The shared runner has no hard-coded stock-market generation module names. Instead,
`workflow.toml` supplies the implementation for its ground-truth and feature steps:

```toml
[workflow]
code = ["steps"] # copied and hashed with each run

[steps."prepare.groundtruth"]
script = "steps/data_generation/step20_generate_groundtruth_stockmarket.py"

[steps."prepare.features"]
script = "steps/data_generation/step21_generate_features.py"
args = ["-debug"]

[steps."infer_prepare.features"]
script = "steps/data_generation/step21_generate_features.py"
args = ["-debug"]

[steps."*.postprocess"]
script = "steps/prediction/step71_value_postprocessing.py"

[steps."*.backtest"]
script = "steps/prediction/step72_backtesting.py"

[steps."*.inference.predict"]
script = "steps/prediction/step70_predict_temporal_data.py"
```

These tables already exist; edit their values rather than adding duplicate tables.
Wildcard bindings avoid repeating the same prediction scripts for every model.
Matching wildcard tables merge in declaration order; an exact step ID (such as
`svm.backtest`) overrides them. Standard config-section arguments are retained
unless explicitly overridden with `args`.
The usual `python run.py` still executes the entire workflow. To run just the
generation stages in a new isolated folder:

```powershell
python run.py --until prepare.dimensions
```

To use annotated outcomes, change `prepare.groundtruth.script` to
`steps/data_generation/step20_generate_groundtruth_stockmarket_from_annotation.py` and configure
`Paths.outcomes_source`, `Generation.outcome_col`, and `Common.class_number` in
the preparation profile. Include the annotation CSV in the copied inputs.
To use reduced features, change both feature bindings to
`steps/data_generation/step21_generate_features_reduced_lt.py` so training and inference agree.
These alternatives retain their existing behavior; they are not the default profile.

Each file also remains directly executable with `--config_path` (and `-debug` for
feature generation). For manual debugging, use a run's `workspace/` as the working
directory and pass its generated INI, e.g. `--config_path=config/prepare.ini`.
An absolute script path works from that directory. Direct execution writes to the
paths specified by that INI; `run.py` is the entry point that creates isolated runs.

Other datasets can supply their own `script` paths using the same mechanism.
An installed or snapshotted package can instead be selected with
`module = "package.module"`; `module` and `script` are mutually exclusive.
Older runs keep their original manifests/configs: this relocation does not rewrite
their history. Start a new run with this workflow configuration after the migration.

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
