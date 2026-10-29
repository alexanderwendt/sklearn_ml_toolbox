"""Small, strict, config-relative TOML interface."""
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Config:
    path: Path
    input: Path
    target: str
    task: str
    output: Path
    prepare: str | None = None
    drop: tuple[str, ...] = ()
    delimiter: str = ","
    strategy: str = "random"
    test_size: float = 0.2
    time_column: str | None = None
    seed: int = 42
    models: tuple[str, ...] = ("linear", "forest")
    folds: int = 3
    search_iterations: int = 10
    select_percentile: int = 100
    balance: bool = False
    tune_threshold: bool = False


def load_config(path: Path) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    allowed = {
        "data": {"path", "target", "task", "prepare", "drop", "delimiter"},
        "split": {"strategy", "test_size", "time_column"},
        "training": {"seed", "models", "folds", "search_iterations", "select_percentile", "balance", "tune_threshold"},
        "output": {"directory"},
    }
    for section, values in raw.items():
        if section not in allowed or not isinstance(values, dict):
            raise ValueError(f"Unknown configuration section: {section}")
        if unknown := values.keys() - allowed[section]:
            raise ValueError(f"Unknown {section} fields: {sorted(unknown)}")
    data = raw.get("data", {})
    for key in ("path", "target", "task"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} must be a nonempty string")
    if data["task"] not in {"classification", "regression"}:
        raise ValueError("data.task must be classification or regression")
    split, training = raw.get("split", {}), raw.get("training", {})
    strategy = split.get("strategy", "random")
    if strategy not in {"random", "temporal"}:
        raise ValueError("split.strategy must be random or temporal")
    if strategy == "temporal" and not split.get("time_column"):
        raise ValueError("split.time_column required for temporal split")
    size = split.get("test_size", 0.2)
    if type(size) not in (int, float) or not 0 < size < 1:
        raise ValueError("split.test_size must be between 0 and 1")
    for key, default, minimum in (("folds", 3, 2), ("search_iterations", 10, 1), ("seed", 42, 0)):
        value = training.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"training.{key} must be an integer >= {minimum}")
    models = training.get("models", ["linear", "forest"])
    if not isinstance(models, list) or not models or any(m not in ("linear", "forest", "svm", "svm_linear", "knn", "gnb", "xgboost") for m in models):
        raise ValueError("training.models must contain linear, forest, svm, svm_linear, knn, gnb or xgboost")
    if data["task"] != "classification" and "gnb" in models:
        raise ValueError("gnb only supports classification")
    drop = data.get("drop", [])
    if not isinstance(drop, list) or any(not isinstance(v, str) for v in drop) or data["target"] in drop:
        raise ValueError("data.drop must be column names, excluding target")
    percentile = training.get("select_percentile", 100)
    if type(percentile) is not int or not 1 <= percentile <= 100:
        raise ValueError("training.select_percentile must be 1..100")
    if type(training.get("balance", False)) is not bool:
        raise ValueError("training.balance must be boolean")
    if type(training.get("tune_threshold", False)) is not bool:
        raise ValueError("training.tune_threshold must be boolean")
    delimiter = data.get("delimiter", ",")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("data.delimiter must be one character")
    hook = data.get("prepare")
    if hook is not None:
        if not isinstance(hook, str) or not hook:
            raise ValueError("data.prepare must be a script path or builtin:name")
        if not hook.startswith("builtin:"):
            hook = str((path.parent / hook).resolve())
    return Config(
        path=path, input=(path.parent / data["path"]).resolve(), target=data["target"],
        task=data["task"], output=(path.parent / raw.get("output", {}).get("directory", "results")).resolve(),
        prepare=hook, drop=tuple(drop), delimiter=delimiter, strategy=strategy, test_size=size,
        time_column=split.get("time_column"), seed=training.get("seed", 42), models=tuple(models),
        folds=training.get("folds", 3), search_iterations=training.get("search_iterations", 10),
        select_percentile=percentile, balance=training.get("balance", False), tune_threshold=training.get("tune_threshold", False),
    )
