"""Durable stage artifacts shared by full runs and individual commands."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import platform
from uuid import uuid4

import joblib
import pandas as pd

from .config import Config, load_config
from .data_generation.loader import prepare_dataset
from .data_processing.preprocessing import split_dataset
from .data_processing.analysis import analyze
from .modeling.training import train_model
from .modeling.evaluation import evaluate_model
from .modeling.reporting import compare_models, write_report


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def log(run, message):
    with (run / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def prepare_run(config: Config, run: Path | None = None) -> Path:
    frame = prepare_dataset(config)
    train, holdout = split_dataset(frame, config)
    run = Path(run) if run is not None else config.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    run.mkdir(parents=True)
    for name, part in (("prepared", frame), ("train", train), ("holdout", holdout)):
        part.to_csv(run / f"{name}.csv", index_label="_row_id")
    # CSV cannot preserve category/numeric distinctions; joblib preserves the stage schema.
    joblib.dump({"train": train, "holdout": holdout}, run / "split.joblib")
    (run / "config.toml").write_bytes(config.path.read_bytes())
    manifest = {"config": asdict(config), "config_sha256": digest(config.path), "input_sha256": digest(config.input), "python": platform.python_version(), "train_ids": train.index.tolist(), "holdout_ids": holdout.index.tolist(), "split_sha256": digest(run / "split.joblib"), "stage": "prepared"}
    if config.prepare and not config.prepare.startswith("builtin:"):
        manifest["hook_sha256"] = digest(Path(config.prepare))
    write_json(run / "manifest.json", manifest)
    log(run, "Prepared and split dataset")
    return run


def validate_run(config: Config, run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != digest(config.path) or manifest["input_sha256"] != digest(config.input):
        raise ValueError("Config/input changed since preparation; start a new run")
    if manifest["split_sha256"] != digest(run / "split.joblib"):
        raise ValueError("Split artifact changed; start a new run")
    if "hook_sha256" in manifest and manifest["hook_sha256"] != digest(Path(config.prepare)):
        raise ValueError("Preparation hook changed; start a new run")
    return manifest


def train_run(config: Config, run: Path) -> Path:
    manifest = validate_run(config, run)
    if (run / "model.joblib").exists():
        raise FileExistsError("Model already exists; start a new run")
    parts = joblib.load(run / "split.joblib")
    model = train_model(parts["train"], config)
    joblib.dump(model, run / "model.joblib")
    pd.DataFrame(model["cv_results"]).to_csv(run / "comparison.csv", index=False)
    manifest.update(stage="trained", model_sha256=digest(run / "model.joblib"), versions=model["versions"], selected_model=model["selected_model"])
    write_json(run / "manifest.json", manifest)
    log(run, f"Trained {model['selected_model']} using training-fold selection")
    return run


def analyze_run(config: Config, run: Path) -> Path:
    validate_run(config, run)
    from .data_processing.diagnostics import analyze_training
    parts = joblib.load(run / 'split.joblib')
    analyze_training(parts['train'], config, run / 'analysis')
    log(run, 'Completed training-only dimensionality reduction and feature diagnostics')
    return run


def evaluate_run(config: Config, run: Path) -> Path:
    manifest = validate_run(config, run)
    if manifest.get("model_sha256") != digest(run / "model.joblib"):
        raise ValueError("Model artifact changed; start a new run")
    if (run / "metrics.json").exists():
        raise FileExistsError("Evaluation already exists; start a new run")
    parts = joblib.load(run / "split.joblib")
    model = joblib.load(run / "model.joblib")
    if config.analysis_methods and not (run / 'analysis/manifest.json').exists():
        analyze_run(config, run)
    result = evaluate_model(model, parts["holdout"], run)
    analyze(parts["train"][model["features"]], run, parts["train"][config.target], config.task)
    table = compare_models(model, parts, run)
    write_report(config, model, result, table, run)
    manifest["stage"] = "evaluated"
    write_json(run / "manifest.json", manifest)
    log(run, "Evaluated holdout; wrote metrics, predictions and report")
    return run


def run_pipeline(config_path: Path) -> Path:
    config = load_config(config_path)
    run = prepare_run(config)
    train_run(config, run)
    return evaluate_run(config, run)
