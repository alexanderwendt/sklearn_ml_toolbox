"""Training-only projections and exploratory feature rankings for tabular tasks."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .preprocessing import feature_frame, make_preprocessor
from ..modeling.evaluation import save_plot


def prepare_analysis_data(train, config):
    """Bound work and fit preprocessing only to the chosen training rows."""
    frame = train
    if len(frame) > config.analysis_max_samples:
        frame, _ = train_test_split(frame, train_size=config.analysis_max_samples,
                                    random_state=config.seed,
                                    stratify=frame[config.target] if config.task == 'classification' else None)
    frame = frame.sort_index()
    features = [name for name in frame if name not in (config.target, config.time_column)]
    numeric = frame[features].select_dtypes(include='number').columns.tolist()
    X = feature_frame(frame, features, numeric)
    preprocessor = make_preprocessor(X).set_params(sparse_threshold=0, categorical__encode__drop='first')
    values = preprocessor.fit_transform(X)
    names = preprocessor.get_feature_names_out()
    keep = np.std(values, axis=0) > 0
    if not keep.any() or len(frame) < 4:
        raise ValueError('Analysis needs at least four training rows and a nonconstant predictor')
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(values[:, keep]), columns=names[keep], index=frame.index)
    return X, frame[config.target], {'preprocessor': preprocessor, 'nonconstant': keep, 'scaler': scaler}


def save_embedding(values, y, name, config, output):
    frame = pd.DataFrame(values[:, :2], index=y.index, columns=['component_1', 'component_2'])
    frame['target'] = y
    frame.to_csv(output / f'{name}.csv', index_label='_row_id')

    def draw(ax):
        if config.task == 'classification':
            for label in sorted(y.unique()):
                chosen = y == label
                ax.scatter(values[chosen, 0], values[chosen, 1], s=12, alpha=.6, label=str(label))
            ax.legend(title=config.target)
        else:
            scatter = ax.scatter(values[:, 0], values[:, 1], c=y, s=12, alpha=.6, cmap='viridis')
            ax.figure.colorbar(scatter, ax=ax, label=config.target)
        ax.set(title=f'{name}: training data', xlabel='Component 1', ylabel='Component 2')
    save_plot(output / f'{name}.png', draw)


def targets(y, task):
    # Class numbers are labels, not continuous distances. Analyze each indicator.
    return [(str(label), (y == label).astype(float)) for label in sorted(y.unique())] if task == 'classification' else [(str(y.name), y)]


def lasso_weights(X, y, config, output):
    rows = []
    for label, response in targets(y, config.task):
        fit = LassoCV(cv=min(config.folds, len(X)), max_iter=100000).fit(X, response)
        rows.extend({'feature': feature, 'target': label, 'coefficient': float(coefficient),
                     'selected': bool(coefficient != 0), 'alpha': float(fit.alpha_)}
                    for feature, coefficient in zip(X.columns, fit.coef_))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'Lasso_Model_Weights.csv', index=False)
    weights = table.pivot(index='feature', columns='target', values='coefficient')
    save_plot(output / 'Lasso_Model_Weights.png', lambda ax: weights.plot.barh(
        ax=ax, title='Lasso weights (one-vs-rest indicators for classification)', figsize=(10, max(6, len(weights) * .25))))


def tree_importance(X, y, config, output):
    cls = ExtraTreesClassifier if config.task == 'classification' else ExtraTreesRegressor
    model = cls(n_estimators=50, random_state=config.seed, n_jobs=1).fit(X, y)
    table = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    table['selected'] = table.importance > .01
    table.to_csv(output / 'Tree_Based_Importance.csv', index=False)
    save_plot(output / 'Tree_Based_Importance.png', lambda ax: table.sort_values('importance').set_index('feature').importance.plot.barh(
        ax=ax, title='ExtraTrees feature importance (threshold 0.01)', figsize=(10, max(6, len(table) * .25))))


def recursive_selection(X, y, config, output):
    model = LogisticRegression(max_iter=2000, random_state=config.seed) if config.task == 'classification' else Ridge()
    if X.shape[1] < 2:
        table = pd.DataFrame({'feature': X.columns, 'rank': 1, 'selected': True})
    else:
        selection = RFE(model, n_features_to_select=max(1, X.shape[1] // 2)).fit(X, y)
        table = pd.DataFrame({'feature': X.columns, 'rank': selection.ranking_, 'selected': selection.support_})
    table.to_csv(output / 'Recursive_Feature_Ranking.csv', index=False)
    save_plot(output / 'Recursive_Feature_Ranking.png', lambda ax: table.sort_values('rank', ascending=False).set_index('feature')['rank'].plot.barh(
        ax=ax, title='Recursive elimination: rank 1 = selected', figsize=(10, max(6, len(table) * .25))))


def backward_selection(X, y, config, output):
    import statsmodels.api as sm
    from scipy.linalg import qr

    # Remove exact linear dependencies before interpreting OLS p-values.
    _, triangular, pivots = qr(X.to_numpy(), mode='economic', pivoting=True)
    rank = np.linalg.matrix_rank(triangular)
    independent = list(X.columns[np.sort(pivots[:rank])])
    records = []
    for label, response in targets(y, config.task):
        columns = independent.copy()
        removed = {feature: np.nan for feature in X.columns if feature not in columns}
        while columns:
            fitted = sm.OLS(response, sm.add_constant(X[columns], has_constant='add')).fit()
            pvalues = fitted.pvalues.drop('const').fillna(1.)
            worst = pvalues.idxmax()
            if pvalues[worst] <= .05:
                break
            removed[worst] = float(pvalues[worst])
            columns.remove(worst)
        for feature in X.columns:
            records.append({'feature': feature, 'target': label, 'selected': feature in columns,
                            'p_value': float(pvalues[feature]) if feature in columns else removed[feature],
                            'linearly_dependent': feature not in independent})
    table = pd.DataFrame(records)
    table.to_csv(output / 'Backward_Elimination.csv', index=False)
    selected = table.groupby('feature').selected.sum().sort_values()
    save_plot(output / 'Backward_Elimination.png', lambda ax: selected.plot.barh(
        ax=ax, title='Backward OLS elimination: number of targets selecting each feature', figsize=(10, max(6, len(selected) * .25))))


def correlations(X, y, config, output):
    from scipy.cluster.hierarchy import dendrogram, linkage

    combined = X.copy()
    for label, response in targets(y, config.task):
        combined[f'target:{label}'] = response
    for method in ('pearson', 'spearman'):
        corr = combined.corr(method=method)
        corr.to_csv(output / f'{method}_correlation.csv')
        def draw(ax):
            ax.figure.set_size_inches(12, 10)
            image = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
            ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
            ax.set_yticks(range(len(corr)), corr.columns, fontsize=6)
            ax.figure.colorbar(image, ax=ax)
            ax.set_title(f'Training {method} correlations')
        save_plot(output / f'{method}_correlation.png', draw)
    if X.shape[1] > 1:
        tree = linkage(X.to_numpy().T, method='ward')
        pd.DataFrame(tree, columns=['left', 'right', 'distance', 'count']).to_csv(output / 'Hierarchical_Linkage.csv', index=False)
        def hierarchy(ax):
            ax.figure.set_size_inches(12, 8)
            dendrogram(tree, labels=X.columns.tolist(), ax=ax, leaf_rotation=90, leaf_font_size=7)
            ax.set_title('Training feature hierarchy (Ward linkage)')
        save_plot(output / 'Hierarchical_Linkage.png', hierarchy)


def analyze_training(train, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    X, y, preprocessing = prepare_analysis_data(train, config)
    X.to_csv(output / 'analysis_features.csv', index_label='_row_id')
    y.to_csv(output / 'analysis_targets.csv', index_label='_row_id')
    joblib.dump(preprocessing, output / 'preprocessing.joblib')
    completed, skipped = [], {}
    for method in config.analysis_methods:
        print(f'Analysis: {method} ({len(X)} training rows, {X.shape[1]} encoded features)', flush=True)
        if method in ('pca', 'tsne', 'umap') and min(X.shape) < 2:
            skipped[method] = 'Requires at least two nonconstant features and two rows'
            continue
        if method == 'pca':
            model = PCA().fit(X)
            save_embedding(model.transform(X), y, 'PCA_Plot', config, output)
            variance = pd.DataFrame({'component': np.arange(1, len(model.explained_variance_ratio_) + 1),
                                    'explained_variance': model.explained_variance_ratio_,
                                    'cumulative_variance': model.explained_variance_ratio_.cumsum()})
            variance.to_csv(output / 'PCA_Variance_Coverage.csv', index=False)
            pd.DataFrame(model.components_, columns=X.columns).to_csv(output / 'PCA_Loadings.csv', index_label='component')
            save_plot(output / 'PCA_Variance_Coverage.png', lambda ax: variance.set_index('component').plot(ax=ax, title='PCA explained variance'))
            def loadings(ax):
                image = ax.imshow(model.components_, aspect='auto', cmap='coolwarm')
                ax.set_xticks(range(X.shape[1]), X.columns, rotation=90, fontsize=6)
                ax.figure.colorbar(image, ax=ax)
                ax.set(title='PCA signed loadings', ylabel='Component')
            save_plot(output / 'PCA_Heatmap.png', loadings)
        elif method == 'tsne':
            values = TSNE(n_components=2, perplexity=min(10., len(X) - 1), max_iter=1000, random_state=config.seed).fit_transform(X)
            save_embedding(values, y, 'T-SNE_Plot', config, output)
        elif method == 'umap':
            from umap import UMAP
            options = dict(n_neighbors=min(5, len(X) - 1), random_state=config.seed, init='random', n_jobs=1)
            save_embedding(UMAP(**options).fit_transform(X), y, 'UMAP_Unsupervised', config, output)
            target_metric = 'categorical' if config.task == 'classification' else 'euclidean'
            response = pd.factorize(y, sort=True)[0] if config.task == 'classification' else y.to_numpy()
            values = UMAP(**options, target_metric=target_metric).fit_transform(X, y=response)
            save_embedding(values, y, 'UMAP_Supervised', config, output)
        else:
            {'lasso': lasso_weights, 'tree': tree_importance, 'recursive': recursive_selection,
             'backward': backward_selection, 'correlation': correlations}[method](X, y, config, output)
        completed.append(method)
    (output / 'manifest.json').write_text(json.dumps({
        'source': 'training split only', 'rows': len(X), 'row_ids': X.index.tolist(),
        'features': X.columns.tolist(), 'seed': config.seed, 'completed': completed, 'skipped': skipped,
        'purpose': 'Exploratory diagnostics; projections and selected features are not fed into training.',
        'classification_targets': 'Lasso and backward OLS use one-vs-rest indicators, not ordinal class numbers.',
        'supervised_umap': 'Uses training labels; apparent separation is not predictive performance.'
    }, indent=2), encoding='utf-8')
