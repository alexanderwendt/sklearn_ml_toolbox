"""Bounded training-fold model selection; no holdout access."""
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..config import Config
from ..data_processing.preprocessing import feature_frame, make_preprocessor


def train_model(train: pd.DataFrame, config: Config) -> dict:
    classification = config.task == "classification"
    y = train[config.target]
    if y.isna().any() or len(train) <= config.folds:
        raise ValueError("Training needs nonmissing targets and more rows than folds")
    if classification and (y.nunique() < 2 or y.value_counts().min() < config.folds):
        raise ValueError("Each class needs at least training.folds rows, with at least two classes")
    features = [name for name in train if name not in (config.target, config.time_column)]
    if not features:
        raise ValueError("No predictor columns remain")
    numeric = train[features].select_dtypes(include="number").columns.tolist()
    X = feature_frame(train, features, numeric)
    if config.strategy == "temporal":
        dates = pd.to_datetime(train[config.time_column])
        if not dates.is_monotonic_increasing:
            raise ValueError("Temporal training data must be sorted")
        groups = dates.drop_duplicates().to_numpy()
        cv = [(np.flatnonzero(dates.isin(groups[a])), np.flatnonzero(dates.isin(groups[b]))) for a, b in TimeSeriesSplit(config.folds).split(groups)]
        if classification and any(y.iloc[a].nunique() < 2 for a, _ in cv):
            raise ValueError("Temporal training folds must contain at least two classes")
    else:
        cv = (StratifiedKFold if classification else KFold)(config.folds, shuffle=True, random_state=config.seed)
    scoring = "f1_macro" if classification else "neg_root_mean_squared_error"
    candidates = {
        "dummy": (DummyClassifier(strategy="prior") if classification else DummyRegressor(), {}),
        "linear": (LogisticRegression(max_iter=2000, random_state=config.seed) if classification else Ridge(solver="lsqr"), {"model__C" if classification else "model__alpha": [0.1, 1.0, 10.0]}),
        "forest": ((RandomForestClassifier if classification else RandomForestRegressor)(n_estimators=100, random_state=config.seed, n_jobs=1), {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3, 8]}),
        "svm": (SVC() if classification else SVR(), {"model__C": [0.1, 1., 10.], "model__gamma": ["scale", 0.01]}),
        "svm_linear": (SVC(kernel="linear") if classification else SVR(kernel="linear"), {"model__C": [0.1, 1., 10.]}),
        "knn": ((KNeighborsClassifier if classification else KNeighborsRegressor)(), {"model__n_neighbors": [3, 5, 11], "model__weights": ["uniform", "distance"]}),
        "gnb": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-7, 1e-5]}),
    }
    if "xgboost" in config.models:
        from xgboost import XGBClassifier, XGBRegressor
        if classification and list(np.unique(y)) != list(range(y.nunique())):
            raise ValueError("XGBoost classification requires labels encoded as 0..n-1; use preparation hook")
        candidates["xgboost"] = ((XGBClassifier if classification else XGBRegressor)(n_estimators=100, random_state=config.seed, n_jobs=1), {"model__max_depth": [3, 6], "model__learning_rate": [0.05, 0.1]})
    results, fitted = [], {}
    for name in dict.fromkeys(("dummy", *config.models)):
        estimator, params = candidates[name]
        steps = [("preprocess", make_preprocessor(X))]
        if name == "gnb":
            steps[0][1].set_params(sparse_threshold=0)
        if config.select_percentile < 100:
            steps.append(("selection", SelectPercentile(f_classif if classification else f_regression, percentile=config.select_percentile)))
        pipe_type = Pipeline
        if config.balance:
            if not classification:
                raise ValueError("training.balance only supports classification")
            from imblearn.over_sampling import RandomOverSampler
            from imblearn.pipeline import Pipeline as BalancedPipeline
            pipe_type = BalancedPipeline
            steps.append(("balance", RandomOverSampler(random_state=config.seed)))
        pipe = pipe_type([*steps, ("model", estimator)])
        started = perf_counter()
        if params:
            count = int(np.prod([len(values) for values in params.values()]))
            search = RandomizedSearchCV(pipe, params, n_iter=min(config.search_iterations, count), scoring=scoring, cv=cv, random_state=config.seed, n_jobs=1, error_score="raise")
            search.fit(X, y)
            fitted[name], score = search.best_estimator_, search.best_score_
            best_params = search.best_params_
        else:
            score = float(cross_val_score(pipe, X, y, cv=cv, scoring=scoring, error_score="raise").mean())
            fitted[name] = pipe.fit(X, y)
            best_params = {}
        results.append({"model": name, "cv_score": float(score), "scorer": scoring, "seconds": perf_counter() - started, "parameters": best_params})
    winner = max(results, key=lambda row: row["cv_score"])["model"]
    if config.tune_threshold:
        from sklearn.model_selection import TunedThresholdClassifierCV
        if not classification or y.nunique() != 2:
            raise ValueError("Threshold tuning requires binary classification")
        fitted[winner] = TunedThresholdClassifierCV(fitted[winner], scoring="balanced_accuracy", cv=cv, n_jobs=1).fit(X, y)
    return {"pipeline": fitted[winner], "candidates": fitted, "baseline": fitted["dummy"], "selected_model": winner, "features": features, "numeric": numeric, "task": config.task, "target": config.target, "cv_results": results, "seed": config.seed, "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}}
