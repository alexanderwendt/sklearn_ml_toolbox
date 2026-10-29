"""Small descriptive report, independent of fitted model selection."""
from pathlib import Path

import pandas as pd

from ..modeling.evaluation import save_plot


def analyze(frame: pd.DataFrame, output: Path, target=None, task=None) -> None:
    graphs = output / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    frame.describe(include="all").to_csv(output / "data_summary.csv")
    feature_plots = []
    for index, name in enumerate(frame.columns):
        values = frame[name]
        prefix = f"feature_{index:03d}"
        numeric_feature = pd.api.types.is_numeric_dtype(values)
        def distribution(ax):
            if numeric_feature and values.notna().any():
                values.plot.hist(ax=ax, bins=30)
            elif values.notna().any():
                values.fillna("(missing)").value_counts().plot.bar(ax=ax)
            else:
                ax.text(0.5, 0.5, "All values missing", ha="center")
            ax.set_title(f"Training distribution: {name}")
        save_plot(graphs / f"{prefix}_distribution.png", distribution)
        save_plot(graphs / f"{prefix}_missing.png", lambda ax: pd.Series(
            {"present": values.notna().sum(), "missing": values.isna().sum()}
        ).plot.bar(ax=ax, title=f"Training missing values: {name}"))
        if target is not None:
            def relationship(ax):
                if numeric_feature and task == "regression":
                    ax.scatter(values, target, s=9, alpha=0.4)
                    ax.set(xlabel=name, ylabel=target.name)
                elif numeric_feature:
                    groups = [(label, values[target == label].dropna()) for label in sorted(target.unique())]
                    groups = [(label, group) for label, group in groups if not group.empty]
                    if groups:
                        ax.boxplot([group for _, group in groups], tick_labels=[str(label) for label, _ in groups])
                    ax.set(xlabel=target.name, ylabel=name)
                elif task == "classification":
                    pd.crosstab(values.fillna("(missing)"), target, normalize="index").plot.bar(stacked=True, ax=ax)
                    ax.set(ylabel="Class fraction")
                else:
                    target.groupby(values.fillna("(missing)")).mean().plot.bar(ax=ax)
                    ax.set(ylabel=f"Mean {target.name}")
                ax.set_title(f"Training target relationship: {name}")
            save_plot(graphs / f"{prefix}_target.png", relationship)
        feature_plots.append({"feature": name, "plot_prefix": prefix})
    pd.DataFrame(feature_plots).to_csv(output / "feature_plots.csv", index=False)
    save_plot(graphs / "missing_values.png", lambda ax: frame.isna().mean().sort_values().tail(30).plot.barh(ax=ax, title="Missing fraction"))
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        save_plot(graphs / "distribution.png", lambda ax: numeric.iloc[:, 0].plot.hist(ax=ax, bins=30, title=str(numeric.columns[0])))
    if len(numeric.columns) > 1:
        def correlation(ax):
            corr = numeric.iloc[:, :20].corr()
            ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
            ax.set_yticks(range(len(corr)), corr.columns, fontsize=6)
            ax.set_title("Training feature correlation")
        save_plot(graphs / "correlation.png", correlation)
