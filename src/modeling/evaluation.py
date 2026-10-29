"""Held-out evaluation and headless, task-appropriate graphs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from ..data_processing.preprocessing import feature_frame


def scores(task, truth, predicted):
    if task == "classification":
        return {"accuracy": metrics.accuracy_score(truth, predicted), "balanced_accuracy": metrics.balanced_accuracy_score(truth, predicted), "macro_f1": metrics.f1_score(truth, predicted, average="macro", zero_division=0), "classification_report": metrics.classification_report(truth, predicted, output_dict=True, zero_division=0)}
    undefined = {}
    r2 = None
    if len(truth) < 2 or np.var(truth) == 0:
        undefined["r2"] = "Requires at least two observations and nonconstant target"
    else:
        r2 = metrics.r2_score(truth, predicted)
    return {"mae": metrics.mean_absolute_error(truth, predicted), "rmse": metrics.root_mean_squared_error(truth, predicted), "r2": r2, "undefined": undefined}


def save_plot(path, draw):
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        draw(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def evaluate_model(model: dict, holdout: pd.DataFrame, output: Path, split="holdout") -> dict:
    output = Path(output)
    graphs = output / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    X = feature_frame(holdout, model["features"], model["numeric"])
    truth = holdout[model["target"]]
    predicted = model["pipeline"].predict(X)
    result = scores(model["task"], truth, predicted)
    result["baseline"] = scores(model["task"], truth, model["baseline"].predict(X))
    predictions = pd.DataFrame({"actual": truth, "predicted": predicted}, index=holdout.index)
    predictions.to_csv(output / "predictions.csv", index_label="_row_id")
    if model["task"] == "classification":
        pipe = model["pipeline"]
        classes = pipe.classes_
        result["classification_report"] = metrics.classification_report(
            truth, predicted, labels=classes, target_names=[str(label) for label in classes],
            output_dict=True, zero_division=0)
        save_plot(graphs / "confusion_matrix.png", lambda ax: metrics.ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=classes, ax=ax, colorbar=False))
        save_plot(graphs / "confusion_matrix_normalized.png", lambda ax: metrics.ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=classes, normalize="true", ax=ax, colorbar=False))
        report = pd.DataFrame(result["classification_report"]).T
        report.to_csv(output / "classification_report.csv")
        for metric in ("precision", "recall", "f1-score", "support"):
            save_plot(graphs / f"class_{metric}.png", lambda ax: report.loc[[str(label) for label in classes if str(label) in report.index], metric].plot.bar(ax=ax, title=f"{split}: per-class {metric}"))
        if len(classes) == 2 and truth.nunique() == 2 and (hasattr(pipe, "predict_proba") or hasattr(pipe, "decision_function")):
            values = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)
            binary = (truth == classes[1]).astype(int)
            result["roc_auc"] = metrics.roc_auc_score(binary, values)
            result["average_precision"] = metrics.average_precision_score(binary, values)
            save_plot(graphs / "roc.png", lambda ax: metrics.RocCurveDisplay.from_predictions(binary, values, ax=ax))
            save_plot(graphs / "precision_recall.png", lambda ax: metrics.PrecisionRecallDisplay.from_predictions(binary, values, ax=ax))
        if hasattr(pipe, "predict_proba") or hasattr(pipe, "decision_function"):
            values = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else pipe.decision_function(X)
            if values.ndim == 1:
                values = np.column_stack((-values, values))
            result["per_class_curves"] = []
            for index, label in enumerate(classes):
                binary = (truth == label).astype(int)
                row = {"class": str(label), "roc_auc": None, "average_precision": None}
                if binary.nunique() < 2:
                    row["reason"] = "Requires positive and negative examples in this split"
                else:
                    score = values[:, index]
                    row.update(roc_auc=metrics.roc_auc_score(binary, score), average_precision=metrics.average_precision_score(binary, score))
                    save_plot(graphs / f"class_{index}_roc.png", lambda ax: metrics.RocCurveDisplay.from_predictions(binary, score, ax=ax, name=f"{split}: {label} vs rest"))
                    save_plot(graphs / f"class_{index}_precision_recall.png", lambda ax: metrics.PrecisionRecallDisplay.from_predictions(binary, score, ax=ax, name=f"{split}: {label} vs rest"))
                result["per_class_curves"].append(row)
            curves = pd.DataFrame(result["per_class_curves"]).set_index("class")
            curves.to_csv(output / "class_curves.csv")
            for metric in ("roc_auc", "average_precision"):
                if curves[metric].notna().any():
                    save_plot(graphs / f"class_{metric}.png", lambda ax: curves[metric].plot.bar(ax=ax, title=f"{split}: one-vs-rest {metric}"))
        else:
            result["score_curves_note"] = "Estimator does not expose probabilities or decision scores"
    else:
        def actual_plot(ax):
            ax.scatter(truth, predicted, s=9, alpha=0.5)
            low, high = min(truth.min(), predicted.min()), max(truth.max(), predicted.max())
            ax.plot([low, high], [low, high], "k--")
            ax.set(xlabel="Actual", ylabel="Predicted", title=f"{split}: predictions")
        def residual_plot(ax):
            ax.scatter(predicted, truth - predicted, s=9, alpha=0.5)
            ax.axhline(0, color="black")
            ax.set(xlabel="Predicted", ylabel="Actual minus predicted", title="Residuals")
        save_plot(graphs / "actual_vs_predicted.png", actual_plot)
        save_plot(graphs / "residuals.png", residual_plot)
        save_plot(graphs / "residual_distribution.png", lambda ax: pd.Series(truth - predicted).plot.hist(ax=ax, bins=30, title=f"{split}: residual distribution"))
        def error_plot(ax):
            ax.scatter(truth, np.abs(truth - predicted), s=9, alpha=0.5)
            ax.set(xlabel="Actual", ylabel="Absolute error", title=f"{split}: absolute error")
        save_plot(graphs / "absolute_error.png", error_plot)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
