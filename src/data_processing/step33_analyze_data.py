#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Data analysis.

This script performs a comprehensive analysis of the prepared data. It includes
visualizations like correlation matrices, pair plots, hierarchical linkage plots,
and dimensionality reduction plots (PCA, t-SNE, UMAP). The goal is to gain
insights into the data and the relationships between the features.

Inputs:
    - Features and outcomes from the previous steps (loaded via `sup.load_features`).

Outputs:
    - A variety of plots saved in the `data_preparation` subdirectory of the
      results directory, including:
        - `Correlation_Strength.png`
        - `Spearman_Correlation_Plot.png`
        - `Pairplot.png`
        - `Hierarchical_Linkage.png`
        - `T-SNE_Plot.png`
        - `UMAP_Supervised.png` and `UMAP_Unsupervised.png`
        - `PCA_Variance_Coverage.png` and `PCA_Plot.png`

Main Functions:
    - `analyse_features`: Orchestrates the entire analysis, calling the various
      plotting functions.
    - `plot_pca`, `plot_umap`, `plot_t_sne`, `plot_parallel_coordinates`,
      `plot_hierarchical_linkage`, `plot_correlation_bar`,
      `plot_spearman_correlation_matrix`, `plot_correlation_matrix2`,
      `plot_correlation_matrix`: Functions for generating the different
      visualizations.
    - `main`: Loads the data and calls `analyse_features`.

License_info: ISC
ISC License

Copyright (c) 2020, Alexander Wendt

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import os
import warnings
import sys
import traceback

# Libs
from pandas.plotting import register_matplotlib_converters
import argparse
import pandas as pd
import matplotlib as m
from matplotlib import ticker
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap.umap_ as umap  # Work around from https://github.com/lmcinnes/umap/issues/24

import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial import distance

import seaborn as sns

from sklearn import preprocessing

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup

__author__ = "Alexander Wendt"
__copyright__ = (
    "Copyright 2020, Christian Doppler Laboratory for " "Embedded Machine Learning"
)
__credits__ = [""]
__license__ = "ISC"
__version__ = "0.2.0"
__maintainer__ = "Alexander Wendt"
__email__ = "alexander.wendt@tuwien.ac.at"
__status__ = "Experiental"

register_matplotlib_converters()

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description="Step 3 - Analyze Data")
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)

args = parser.parse_args()


def unique_cols(df):
    """
    Check if all values in a column are the same.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    np.ndarray
        A boolean array indicating which columns have all the same values.
    """
    a = df.values
    return (a[0] == a).all(0)


def analyse_features(features, y, class_labels, conf, image_save_directory):
    """
    Perform a comprehensive analysis of the features.

    Parameters
    ----------
    features : pd.DataFrame
        The features DataFrame.
    y : np.ndarray
        The outcomes array.
    class_labels : dict
        A dictionary mapping class labels to integer values.
    conf : configparser.ConfigParser
        The configuration object.
    image_save_directory : str
        The directory where the plots will be saved.
    """

    features_reduced = features.loc[:, np.invert(unique_cols(features))]
    print("Reduce columns that are duplicated in terms of values.")
    features = features_reduced

    X_scaled = rescale_features(features)
    if y is not None:
        y_scaled = rescale_outcomes(conf, features, y)
        print("Merged features and outcomes to use in correlation matrix")
        total_values_scaled = X_scaled.join(y_scaled)

        plot_correlation_matrix2(conf, image_save_directory, total_values_scaled)

        plot_correlation_bar(X_scaled, conf, image_save_directory, y_scaled)
    else:
        total_values_scaled = X_scaled
        print("Only features will be used in for correlations.")

    plot_correlation_matrix(features, image_save_directory, total_values_scaled)
    plot_spearman_correlation_matrix(image_save_directory, total_values_scaled)

    try:
        plot_hierarchical_linkage(X_scaled, conf, image_save_directory)
    except:
        warnings.warn("Cannot execute hiearchical linkage")
        traceback.print_exc()

    X_train_index_subset = sup.get_random_data_subset_index(1000, features)
    X_train_scaled_subset = X_scaled.iloc[X_train_index_subset, :]

    if y is not None:
        df_y = pd.DataFrame(
            data=y.reshape(-1, 1),
            index=features.index,
            columns=[conf["Common"].get("class_name")],
        )
        total_values = features.join(df_y)
        print("Merged features and outcomes to use in correlation matrix unscaled")
        y_train_subset = np.array(y[X_train_index_subset]).flatten()

        feature_plot = list(range(0, 10, 1))
        cols = total_values.columns[feature_plot]
        print(feature_plot)
        print(cols)

        comparison_name = conf["Common"].get("class_name")
        print("Class name: ", comparison_name)

        df_fv = total_values.iloc[X_train_index_subset, :]

        m.rc_file_defaults()
        colors = ["#2e8ad8", "#cd3785", "#c64c00", "#889a00"]
        plot_parallel_coordinates(
            df_fv, cols, colors, comparison_name, conf, image_save_directory
        )
    else:
        y_train_subset = None
        warnings.warn("No y value. Parallel coordinates will not be calculated.")

    plot_t_sne(X_train_scaled_subset, y_train_subset, class_labels, image_save_directory)
    plot_umap(X_scaled, class_labels, image_save_directory, y)

    try:
        plot_pca(X_scaled, class_labels, image_save_directory, y)
    except:
        warnings.warn("Cannot execute PCA")
        traceback.print_exc()


def plot_pca(X_scaled, class_labels, image_save_directory, y):
    """
    Plot the PCA results.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        The scaled features DataFrame.
    class_labels : dict
        A dictionary mapping class labels to integer values.
    image_save_directory : str
        The directory where the plots will be saved.
    y : np.ndarray
        The outcomes array.
    """

    m.rc_file_defaults()
    pca_trafo = PCA().fit(X_scaled)
    pca_values = pca_trafo.transform(X_scaled)
    targets = np.array(y).flatten()
    fig, ax1 = plt.subplots(figsize=(10, 8))
    plt.semilogy(pca_trafo.explained_variance_ratio_, "--o")
    ax2 = ax1.twinx()
    plt.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), "--o", color="green")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance")
    plt.xticks(np.arange(0, len(pca_trafo.explained_variance_ratio_)))
    plt.hlines(
        0.95,
        0,
        len(pca_trafo.explained_variance_ratio_.cumsum()),
        colors="red",
        linestyles="solid",
        label="95% variance covered",
    )

    vis.save_figure(
        plt.gcf(),
        image_save_directory=image_save_directory,
        filename="PCA_Variance_Coverage",
    )

    fig = plt.figure()
    sns.heatmap(
        np.log(pca_trafo.inverse_transform(np.eye(X_scaled.shape[1]))),
        cmap="hot",
        cbar=True,
    )
    necessary_components = pca_trafo.explained_variance_ratio_.cumsum()[
        pca_trafo.explained_variance_ratio_.cumsum() < 0.95
    ]
    print(
        "95% variance covered with the {} first components. Values={}".format(
            len(necessary_components), necessary_components
        )
    )

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="PCA_Heatmap"
    )

    plt.figure(figsize=(10, 10))
    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(
            pca_values[idx, 0],
            pca_values[idx, 1],
            label=class_labels[t],
            edgecolor="none",
            alpha=0.5,
        )
    plt.legend(labels=class_labels.values(), bbox_to_anchor=(1, 1))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="PCA_Plot"
    )


def plot_umap(X_scaled, class_labels, image_save_directory, y):
    """
    Plot the UMAP results.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        The scaled features DataFrame.
    class_labels : dict
        A dictionary mapping class labels to integer values.
    image_save_directory : str
        The directory where the plots will be saved.
    y : np.ndarray
        The outcomes array.
    """

    sns.set(style="white", context="poster")
    embeddingUnsupervised = umap.UMAP(
        n_neighbors=5, random_state=42, init="random"
    ).fit_transform(X_scaled)

    if y is not None:
        embeddingSupervised = umap.UMAP(
            n_neighbors=5, random_state=42, init="random"
        ).fit_transform(X_scaled, y=y)
        vis.plotUmap(
            embeddingSupervised,
            y,
            list(class_labels.values()),
            "Dataset supervised clustering",
        )

        vis.save_figure(
            plt.gcf(), image_save_directory=image_save_directory, filename="UMAP_Supervised"
        )
        print("Plot UMAP supervised")

        vis.plotUmap(
            embeddingUnsupervised,
            y,
            list(class_labels.values()),
            "Dataset unsupervised clustering",
            cmapString="RdYlGn",
        )
        print("Plot UMAP unsupervised with class labels")
    else:
        warnings.warn("No y values.")
        vis.plotUmap(
            embeddingUnsupervised,
            None,
            None,
            "Dataset unsupervised clustering",
            cmapString="RdYlGn",
        )
        print("Plot UMAP unsupervised without class labels")

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="UMAP_Unsupervised"
    )
    print("Plot UMAP unsupervised")


def plot_t_sne(X_scaled_subset, y_scaled_subset, class_labels, image_save_directory):
    """
    Plot the t-SNE results.

    Parameters
    ----------
    X_scaled_subset : np.ndarray
        The scaled subset of the features.
    y_scaled_subset : np.ndarray
        The subset of the outcomes.
    class_labels : dict
        A dictionary mapping class labels to integer values.
    image_save_directory : str
        The directory where the plots will be saved.
    """
    np.random.seed(0)
    X_embedded = TSNE(
        n_components=2,
        perplexity=10.0,
        early_exaggeration=100.0,
        n_iter=5000,
        n_iter_without_progress=1000,
        learning_rate=10,
    ).fit_transform(X_scaled_subset)
    m.rc_file_defaults()
    plt.figure(figsize=(10, 10))
    texts = []

    if y_scaled_subset is not None and class_labels is not None:
        print("Plot t-sne with known classes")
        for i, t in enumerate(set(y_scaled_subset)):
            idx = y_scaled_subset == t
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=class_labels[t])
        plt.legend(bbox_to_anchor=(1, 1))
    else:
        print("Plot t-sne without known classes")
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="T-SNE_Plot"
    )


def plot_parallel_coordinates(
    df, cols, colours, comparison_name, conf, image_save_directory
):
    """
    Plot parallel coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to plot.
    cols : list
        The columns to plot.
    colours : list
        The colors to use for the different classes.
    comparison_name : str
        The name of the comparison column.
    conf : configparser.ConfigParser
        The configuration object.
    image_save_directory : str
        The directory where the plots will be saved.
    """
    x = [i for i, _ in enumerate(cols)]

    colours = {
        df[comparison_name].astype("category").cat.categories[i]: colours[i]
        for i, _ in enumerate(df[comparison_name].astype("category").cat.categories)
    }

    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(15, 5))

    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    for i, ax in enumerate(axes):
        for idx in df.index:
            mpg_category = df.loc[idx, comparison_name]
            ax.plot(x, df.loc[idx, cols], colours[mpg_category])
        ax.set_xlim([x[i], x[i + 1]])

    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks - 1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks - 1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]])

    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])

    plt.subplots_adjust(wspace=0)

    plt.legend(
        [
            plt.Line2D((0, 1), (0, 0), color=colours[cat])
            for cat in df[comparison_name].astype("category").cat.categories
        ],
        df[comparison_name].astype("category").cat.categories,
        bbox_to_anchor=(1.2, 1),
        loc=2,
        borderaxespad=0.0,
    )

    plt.title("Values of attributes by category")
    vis.save_figure(
        plt.gcf(),
        image_save_directory=image_save_directory,
        filename="Parallel_Coordinates",
    )


def plot_hierarchical_linkage(X_scaled, conf, image_save_directory):
    """
    Plot hierarchical linkage.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        The scaled features DataFrame.
    conf : configparser.ConfigParser
        The configuration object.
    image_save_directory : str
        The directory where the plots will be saved.
    """
    corr_matrix = X_scaled.corr()
    correlations_array = np.asarray(corr_matrix)
    linkage = hierarchy.linkage(distance.pdist(correlations_array), method="average")
    g = sns.clustermap(
        corr_matrix,
        row_linkage=linkage,
        col_linkage=linkage,
        row_cluster=True,
        col_cluster=True,
        figsize=(8, 8),
        cmap=plt.get_cmap("coolwarm"),
    )
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    label_order = corr_matrix.iloc[:, g.dendrogram_row.reordered_ind].columns

    vis.save_figure(
        plt.gcf(),
        image_save_directory=image_save_directory,
        filename="Hierarchical_Linkage",
    )


def plot_correlation_bar(X_scaled, conf, image_save_directory, y_scaled):
    """
    Plot correlation bar.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        The scaled features DataFrame.
    conf : configparser.ConfigParser
        The configuration object.
    image_save_directory : str
        The directory where the plots will be saved.
    y_scaled : pd.DataFrame
        The scaled outcomes DataFrame.
    """
    m.rc_file_defaults()
    corr = X_scaled.corrwith(y_scaled[conf["Common"].get("class_name")], axis=0)
    corr.sort_values().plot.barh(
        color="blue", title="Strength of Correlation", figsize=(10, 25)
    )
    print(corr)
    plt.gcf()

    vis.save_figure(
        plt.gcf(),
        image_save_directory=image_save_directory,
        filename="Correlation_Strength",
    )


def plot_spearman_correlation_matrix(image_save_directory, total_values):
    """
    Plot Spearman correlation matrix.

    Parameters
    ----------
    image_save_directory : str
        The directory where the plots will be saved.
    total_values : pd.DataFrame
        The DataFrame containing all values.
    """
    matfig = plt.figure(figsize=(20, 20))
    plt.matshow(
        total_values.corr(method="spearman"), fignum=1, cmap=plt.get_cmap("coolwarm")
    )
    plt.xticks(range(len(total_values.columns)), total_values.columns)
    plt.yticks(range(len(total_values.columns)), total_values.columns)
    plt.xticks(rotation=90)
    plt.colorbar()

    vis.save_figure(
        plt.gcf(),
        image_save_directory=image_save_directory,
        filename="Spearman_Correlation_Plot",
    )


def plot_correlation_matrix2(conf, image_save_directory, total_values):
    """
    Plot pairplot.

    Parameters
    ----------
    conf : configparser.ConfigParser
        The configuration object.
    image_save_directory : str
        The directory where the plots will be saved.
    total_values : pd.DataFrame
        The DataFrame containing all values.
    """
    feature_plot = list(range(0, 10, 1))
    feature_plot.extend([-1])
    g = sns.pairplot(
        total_values.iloc[0:1000, feature_plot],
        hue=conf["Common"].get("class_name"),
        diag_kind="hist",
    )
    g.map_upper(sns.regplot)
    g.map_lower(sns.residplot)
    g.map_diag(plt.hist)
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)
    g.add_legend()
    g.set(alpha=0.5)

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="Pairplot"
    )


def plot_correlation_matrix(features, image_save_directory, total_values):
    """
    Plot scatter matrix.

    Parameters
    ----------
    features : pd.DataFrame
        The features DataFrame.
    image_save_directory : str
        The directory where the plots will be saved.
    total_values : pd.DataFrame
        The DataFrame containing all values.
    """
    feature_plot = list(range(0, 10, 1))
    feature_plot.extend([-4, -3, -2, -1])
    print(feature_plot)
    print(total_values.columns[feature_plot])

    if np.linalg.cond(total_values.iloc[:, feature_plot]) < 1 / sys.float_info.epsilon:
        m.rc_file_defaults()
        axs = pd.plotting.scatter_matrix(
            total_values.iloc[:, feature_plot],
            figsize=(15, 15),
            alpha=0.2,
            diagonal="kde",
        )
        n = len(features.iloc[:, feature_plot].columns)
        for i in range(n):
            for j in range(n):
                ax = axs[i, j]
                ax.xaxis.label.set_rotation(90)
                ax.yaxis.label.set_rotation(0)
                ax.yaxis.labelpad = 50
        vis.save_figure(
            plt.gcf(), image_save_directory=image_save_directory, filename="Scatter-Matrix"
        )
    else:
        warnings.warn("Inputmatrix is singular and cannot be calculated. ")


def rescale_features(features):
    """
    Rescale features using StandardScaler.

    Parameters
    ----------
    features : pd.DataFrame
        The features DataFrame.

    Returns
    -------
    pd.DataFrame
        The scaled features DataFrame.
    """

    scaler = preprocessing.StandardScaler()
    scaler.fit(features)
    X_scaled = pd.DataFrame(
        data=scaler.transform(features), index=features.index, columns=features.columns
    )
    print("Unscaled values")
    print(features.iloc[0:2, :])
    print("Scaled values")
    print(X_scaled.iloc[0:2, :])

    return X_scaled


def rescale_outcomes(conf, features, y):
    """
    Rescale outcomes using StandardScaler.

    Parameters
    ----------
    conf : configparser.ConfigParser
        The configuration object.
    features : pd.DataFrame
        The features DataFrame.
    y : np.ndarray
        The outcomes array.

    Returns
    -------
    pd.DataFrame
        The scaled outcomes DataFrame.
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(y.reshape(-1, 1))
    y_scaled = pd.DataFrame(
        data=scaler.transform(y.reshape(-1, 1)),
        index=features.index,
        columns=[conf["Common"].get("class_name")],
    )
    print("Unscaled values")
    print(y[0:10])
    print("Scaled values")
    print(y_scaled.iloc[0:10, :])

    return y_scaled


def main(config_path):
    """
    Main function to execute the script.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    conf = sup.load_config(config_path)
    features, y, df_y, class_labels = sup.load_features(conf)

    image_save_directory = conf["Paths"].get("results_directory") + "/data_preparation"

    print(
        "WARNING: If a singular matrix occurs in a calculation, probably the outcome is "
        "only one value."
    )
    analyse_features(features, y, class_labels, conf, image_save_directory)


if __name__ == "__main__":
    main(args.config_path)

    print("=== Program end ===")
