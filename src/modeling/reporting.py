"""Per-candidate diagnostics; model selection remains training-fold only."""
from html import escape
from urllib.parse import quote

import pandas as pd
from sklearn.inspection import permutation_importance

from ..data_processing.preprocessing import feature_frame
from .evaluation import evaluate_model, save_plot


def compare_models(model, parts, output):
    rows = []
    candidates = model.get("candidates", {model["selected_model"]: model["pipeline"]})
    for name, pipeline in candidates.items():
        directory = output / "models" / name
        candidate = {**model, "pipeline": pipeline}
        for split, frame in parts.items():
            result = evaluate_model(candidate, frame, directory / split, split=split)
            rows.append({"model": name, "split": split, **{
                key: value for key, value in result.items()
                if isinstance(value, (int, float)) or value is None
            }})
        frame = parts["holdout"]
        X = feature_frame(frame, model["features"], model["numeric"])
        scorer = "f1_macro" if model["task"] == "classification" else "neg_root_mean_squared_error"
        importance = permutation_importance(pipeline, X, frame[model["target"]],
                                            scoring=scorer, n_repeats=5,
                                            random_state=model["seed"], n_jobs=1)
        table = pd.DataFrame({"feature": model["features"], "mean": importance.importances_mean,
                              "std": importance.importances_std}).sort_values("mean")
        table.to_csv(directory / "feature_importance.csv", index=False)
        save_plot(directory / "feature_importance.png", lambda ax: table.set_index("feature")["mean"].plot.barh(
            ax=ax, xerr=table["std"].to_numpy(), title=f"{name}: holdout permutation importance ({scorer})"))
    table = pd.DataFrame(rows)
    table.to_csv(output / "model_metrics.csv", index=False)
    for metric in table.columns.difference(["model", "split"]):
        if table[metric].notna().any():
            save_plot(output / "graphs" / f"metric_{metric}.png", lambda ax: table.pivot(
                index="model", columns="split", values=metric).plot.bar(ax=ax, title=metric))
    comparison = pd.DataFrame(model["cv_results"]).set_index("model")
    for column, title in (("cv_score", f"Training CV: {comparison.scorer.iloc[0]} (higher is better)"),
                          ("seconds", "Search and fit time (seconds)")):
        save_plot(output / "graphs" / f"{column}.png", lambda ax: comparison[column].plot.bar(ax=ax, title=title))
    return table


def write_report(config, model, result, table, output):
    import json

    sections = []
    for directory in sorted({path.parent for path in output.rglob("*.png")}):
        title = directory.relative_to(output).as_posix()
        images = []
        for path in sorted(directory.glob("*.png")):
            relative = path.relative_to(output).as_posix()
            images.append(f'<figure><img loading="lazy" src="{quote(relative)}" alt="{escape(path.stem)}"><figcaption>{escape(path.stem)}</figcaption></figure>')
        sections.append(f'<details open><summary>{escape(title)}</summary>{"".join(images)}</details>')
    links = []
    for path in sorted(output.rglob("*")):
        if path.suffix in (".csv", ".json", ".toml", ".log"):
            relative = path.relative_to(output).as_posix()
            links.append(f'<li><a href="{quote(relative)}">{escape(relative)}</a></li>')
    report = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{escape(config.target)} results</title>
<style>body{{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:1em}}img{{max-width:100%}}pre{{white-space:pre-wrap}}table{{border-collapse:collapse}}td,th{{padding:.4em}}summary{{font-size:1.3em;font-weight:bold;margin:1em 0}}figure{{display:inline-block;max-width:48%;vertical-align:top;margin:1%}}@media(max-width:700px){{figure{{max-width:100%}}}}</style></head><body>
<h1>{escape(config.target)}: {escape(config.task)}</h1>
<p>Selected model: {escape(model['selected_model'])}. Selection uses training cross-validation only.
Training scores are in-sample diagnostics, not generalization estimates. Holdout scores and permutation
importance are diagnostic only: do not use them to tune models. Importance is predictive, not causal;
correlated features can share importance. Error bars show standard deviation over five permutations.</p>
<h2>Selected model: holdout metrics</h2><pre>{escape(json.dumps(result, indent=2))}</pre>
<h2>All models: training versus holdout</h2>{table.to_html(index=False, float_format=lambda value: f'{value:.4f}')}
<h2>Plots</h2>{''.join(sections)}<h2>Tables and artifacts</h2><ul>{''.join(links)}</ul></body></html>'''
    (output / "report.html").write_text(report, encoding="utf-8")
