#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Data analysis
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
#from __future__ import print_function

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

from pandas.plotting import register_matplotlib_converters

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)
#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 3 - Analyze Data')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()

def unique_cols(df):
    a = df.values
    return (a[0] == a).all(0)

def analyse_features(features, y, class_labels, conf, image_save_directory):
    '''


    '''



    # Feature Visualization
    # Here, feature selection and visulization of datasets is performed Methods - Feature visualization through
    # t - SNE - Feature visualization and analysis through PCA


    ### Standardize Data for Feature Selection and Visualization
    # Z - Normalize the data around zero and divided by standard deviation.Fit the normalizer on the training data and
    # transform the training and the test data.The reason is that the scaler only must depend on the training data,
    # in order to prevent leakage of information from the test data.

    # === Remove Columns that are all the same
    features_reduced = features.loc[:, np.invert(unique_cols(features))]
    print("Reduce columns that are duplicated in terms of values.")
    features = features_reduced

    # === Select the best type of scaler ===#
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

    ### Feature and Outcomes Correlation Matrix
    plot_correlation_matrix(features, image_save_directory, total_values_scaled)

    #fixme: Class names not correct shown
    plot_spearman_correlation_matrix(image_save_directory, total_values_scaled)

    #from tabulate import tabulate
    #print(tabulate(X_scaled, headers='keys', tablefmt='psql'))

    try:
        plot_hierarchical_linkage(X_scaled, conf, image_save_directory)
    except:
        warnings.warn("Cannot execute hiearchical linkage")
        traceback.print_exc()

    ### Feature visualization with Parallel Coordinates
    # Select a random subset to visualize
    import random

    # Reduce the training set with the number of samples randomly chosen
    X_train_index_subset = sup.get_random_data_subset_index(1000, features)
    X_train_scaled_subset = X_scaled.iloc[X_train_index_subset, :]

    if y is not None:
        df_y = y_scaled = pd.DataFrame(data=y.reshape(-1, 1), index=features.index, columns=[conf['Common'].get('class_name')])
        total_values = features.join(df_y)
        print("Merged features and outcomes to use in correlation matrix unscaled")
        y_train_subset = np.array(y[X_train_index_subset]).flatten()

        # Select column values to use in the correlation plot
        feature_plot = list(range(0, 10, 1))
        # cols = ['MA2Norm', 'MA50Norm', 'MA200Norm', 'MA400Norm', 'MA200NormDiff', 'MA400NormDiff']
        cols = total_values.columns[feature_plot]
        print(feature_plot)
        print(cols)

        comparison_name = conf['Common'].get('class_name')
        print("Class name: ", comparison_name)

        df_fv = total_values.iloc[X_train_index_subset, :]

        # Use parallel coordinates to visualize the classes and all features for plotting
        # https://plot.ly/python/parallel-coordinates-plot/
        # http://benalexkeen.com/parallel-coordinates-in-matplotlib/

        m.rc_file_defaults()  # Reset sns
        colors = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']
        plot_parallel_coordinates(df_fv, cols, colors, comparison_name, conf, image_save_directory)
        #plot_parallel_coordinates(X_train_index_subset, cols, comparison_name, total_values)
    else:
        y_train_subset = None
        warnings.warn("No y value. Parallel coordinates will not be calculated.")

    #### t-SNE Parameter Grid Search
    #calibrate_tsne = False
    #if calibrate_tsne:
    #    find_tsne_parmeters(X_train_scaled_subset, y_train_scaled_subset, class_labels)

    # t-SNE plot
    plot_t_sne(X_train_scaled_subset, y_train_subset, class_labels, image_save_directory)

    ### UMAP Cluster Analysis
    plot_umap(X_scaled, class_labels, image_save_directory, y)

    ### PCA Analysis
    try:
        plot_pca(X_scaled, class_labels, image_save_directory, y)
    except:
        warnings.warn("Cannot execute PCA")
        traceback.print_exc()


def plot_pca(X_scaled, class_labels, image_save_directory, y):

    m.rc_file_defaults()  # Reset sns
    pca_trafo = PCA().fit(X_scaled)
    pca_values = pca_trafo.transform(X_scaled)
    # from adjustText import adjust_text
    targets = np.array(y).flatten()
    fig, ax1 = plt.subplots(figsize=(10, 8))
    plt.semilogy(pca_trafo.explained_variance_ratio_, '--o')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plt.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), '--o', color='green');
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance")
    plt.xticks(np.arange(0, len(pca_trafo.explained_variance_ratio_)))
    plt.hlines(0.95, 0, len(pca_trafo.explained_variance_ratio_.cumsum()), colors='red', linestyles='solid',
               label='95% variance covered')

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='PCA_Variance_Coverage')


    fig = plt.figure()
    sns.heatmap(np.log(pca_trafo.inverse_transform(np.eye(X_scaled.shape[1]))), cmap="hot", cbar=True)
    necessary_components = pca_trafo.explained_variance_ratio_.cumsum()[pca_trafo.explained_variance_ratio_.cumsum() < 0.95]
    print("95% variance covered with the {} first components. Values={}".format(len(necessary_components), necessary_components))

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='PCA_Heatmap')

    plt.figure(figsize=(10, 10))
    # plt.scatter(pca_values[:,0], pca_values[:,1], c=targets, edgecolor='none', label=class_labels.values(), alpha=0.5)
    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(pca_values[idx, 0], pca_values[idx, 1], label=class_labels[t], edgecolor='none', alpha=0.5)
    plt.legend(labels=class_labels.values(), bbox_to_anchor=(1, 1));
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='PCA_Plot')


def plot_umap(X_scaled, class_labels, image_save_directory, y):
    # Use a supervised / unsupervised analysis to make the clusters

    sns.set(style='white', context='poster')
    # import umap
    # %time #Time of the whole cell
    embeddingUnsupervised = umap.UMAP(n_neighbors=5, random_state=42, init='random').fit_transform(X_scaled)
    # %time #Time of the whole cell

    if y is not None:
        embeddingSupervised = umap.UMAP(n_neighbors=5, random_state=42, init='random').fit_transform(X_scaled, y=y)
        vis.plotUmap(embeddingSupervised, y, list(class_labels.values()), 'Dataset supervised clustering')

        vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='UMAP_Supervised')
        print("Plot UMAP supervised")

        vis.plotUmap(embeddingUnsupervised, y, list(class_labels.values()), 'Dataset unsupervised clustering', cmapString='RdYlGn')
        print("Plot UMAP unsupervised with class labels")
    else:
        warnings.warn("No y values.")
        vis.plotUmap(embeddingUnsupervised, None, None, 'Dataset unsupervised clustering', cmapString='RdYlGn')
        print("Plot UMAP unsupervised without class labels")

    
    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='UMAP_Unsupervised')
    print("Plot UMAP unsupervised")
        


def plot_t_sne(X_scaled_subset, y_scaled_subset, class_labels, image_save_directory):
    ### Visualize Data with t-SNE
    # Select a random subset to visualize
    import random
    # Reduce the training set with the number of samples randomly chosen
    # X_train_index_subset = sup.get_data_subset_index(1000, X_scaled)
    np.random.seed(0)
    # X_embedded = TSNE(n_components=2, perplexity=5.0, early_exaggeration=12.0, n_iter=5000,
    #                  n_iter_without_progress=1000, learning_rate=10).fit_transform(embedded)
    X_embedded = TSNE(n_components=2, perplexity=10.0, early_exaggeration=100.0, n_iter=5000,
                      n_iter_without_progress=1000, learning_rate=10).fit_transform(
        X_scaled_subset)
    #### Plot t-SNE with best parameters
    m.rc_file_defaults()  # Reset sns
    # Plot with texts added to the graphs
    # from adjustText import adjust_text
    #targets = np.array(y[X_train_index_subset]).flatten()
    plt.figure(figsize=(10, 10))
    texts = []

    if y_scaled_subset is not None and class_labels is not None:
        print("Plot t-sne with known classes")
        for i, t in enumerate(set(y_scaled_subset)):
            idx = y_scaled_subset == t
            # for x, y in zip(X_embedded[idx, 0], X_embedded[idx, 1]):
            # texts.append(plt.text(x, y, t))
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=class_labels[t])
        # adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1,1), expand_text=(1,1), arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
        plt.legend(bbox_to_anchor=(1, 1));
    else:
        print("Plot t-sne without known classes")
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])



    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='T-SNE_Plot')

    #if image_save_directory:
    #    if not os.path.isdir(image_save_directory):
    #        os.makedirs(image_save_directory)
    #    plt.savefig(os.path.join(image_save_directory, 'T-SNE_Plot'), dpi=300)

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close()


def plot_parallel_coordinates(df, cols, colours, comparison_name, conf, image_save_directory):
    x = [i for i, _ in enumerate(cols)]

    # create dict of categories: colours
    colours = {df[comparison_name].astype('category').cat.categories[i]: colours[i]
               for i, _ in enumerate(df[comparison_name].astype('category').cat.categories)}

    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(15, 5))

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            mpg_category = df.loc[idx, comparison_name]
            ax.plot(x, df.loc[idx, cols], colours[mpg_category])
        ax.set_xlim([x[i], x[i + 1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
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

    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    # Add legend to plot
    plt.legend(
        [plt.Line2D((0, 1), (0, 0), color=colours[cat]) for cat in
         df[comparison_name].astype('category').cat.categories],
        df[comparison_name].astype('category').cat.categories,
        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

    plt.title("Values of attributes by category")
    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='Parallel_Coordinates')

    #if image_save_directory:
    #    if not os.path.isdir(image_save_directory):
    #        os.makedirs(image_save_directory)
    #    plt.savefig(os.path.join(image_save_directory, 'Parallel_Coordinates'), dpi=300)

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close()


def plot_hierarchical_linkage(X_scaled, conf, image_save_directory):
    '''


    '''
    corr_matrix = X_scaled.corr()
    correlations_array = np.asarray(corr_matrix)
    linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
    g = sns.clustermap(corr_matrix, row_linkage=linkage, col_linkage=linkage, row_cluster=True, col_cluster=True,
                       figsize=(8, 8), cmap=plt.get_cmap('coolwarm'))
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    label_order = corr_matrix.iloc[:, g.dendrogram_row.reordered_ind].columns

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='Hierarchical_Linkage')

    #if image_save_directory:
    #    if not os.path.isdir(image_save_directory):
    #       os.makedirs(image_save_directory)
    #    plt.savefig(os.path.join(image_save_directory, 'Hierarchical_Linkage'), dpi=300)

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close()


def plot_correlation_bar(X_scaled, conf, image_save_directory, y_scaled):
    m.rc_file_defaults()  # Reset sns
    corr = X_scaled.corrwith(y_scaled[conf['Common'].get('class_name')], axis=0)
    corr.sort_values().plot.barh(color='blue', title='Strength of Correlation', figsize=(10, 25))
    print(corr)
    plt.gcf()

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='Correlation_Strength')

    #if image_save_directory:
    #    if not os.path.isdir(image_save_directory):
    #        os.makedirs(image_save_directory)
    #    plt.savefig(os.path.join(image_save_directory, conf['Common'].get('dataset_name') + '_Correlation_Strength'), dpi=300)

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close()


def plot_spearman_correlation_matrix(image_save_directory, total_values):
    # http://benalexkeen.com/correlation-in-python/
    matfig = plt.figure(figsize=(20, 20))
    plt.matshow(total_values.corr(method='spearman'), fignum=1,
                cmap=plt.get_cmap(
                    'coolwarm'))  # Use spearman correlation instead of pearson to have a robust correlation
    plt.xticks(range(len(total_values.columns)), total_values.columns)
    plt.yticks(range(len(total_values.columns)), total_values.columns)
    plt.xticks(rotation=90)
    plt.colorbar()

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='Spearman_Correlation_Plot')

    #if image_save_directory:
    #    if not os.path.isdir(image_save_directory):
    #        os.makedirs(image_save_directory)
    #    plt.savefig(os.path.join(image_save_directory, 'Spearman_Correlation_Plot'), dpi=300)

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close()


def plot_correlation_matrix2(conf, image_save_directory, total_values):

    # https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6
    feature_plot = list(range(0, 10, 1))
    feature_plot.extend([-1])
    g = sns.pairplot(total_values.iloc[0:1000, feature_plot], hue=conf['Common'].get('class_name'), diag_kind="hist")
    # total_values.columns[-1]
    g.map_upper(sns.regplot)
    g.map_lower(sns.residplot)
    g.map_diag(plt.hist)
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)
    # FIXME: Legend is incorrect shown. Only numbers instead of class names
    g.add_legend()
    g.set(alpha=0.5)

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='Pairplot')

    #if image_save_directory:
    #    if not os.path.isdir(image_save_directory):
    #        os.makedirs(image_save_directory)
    #    plt.savefig(os.path.join(image_save_directory, 'Pairplot'), dpi=300)

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close()

def plot_correlation_matrix(features, image_save_directory, total_values):
    # Select column values to use in the correlation plot
    feature_plot = list(range(0, 10, 1))
    # Select outcomes to show
    feature_plot.extend([-4, -3, -2, -1])
    print(feature_plot)
    print(total_values.columns[feature_plot])
    # http://benalexkeen.com/correlation-in-python/
    # https://stackoverflow.com/questions/26975089/making-the-labels-of-the-scatterplot-vertical-and-horizontal-in-pandas

    #Check if the matrix is singular
    if np.linalg.cond(total_values.iloc[:, feature_plot]) < 1/sys.float_info.epsilon:
        m.rc_file_defaults()  # Reset sns
        axs = pd.plotting.scatter_matrix(total_values.iloc[:, feature_plot], figsize=(15, 15), alpha=0.2, diagonal='kde')
        n = len(features.iloc[:, feature_plot].columns)
        for i in range(n):
            for j in range(n):
                # to get the axis of subplots
                ax = axs[i, j]
                # to make x axis name vertical
                ax.xaxis.label.set_rotation(90)
                # to make y axis name horizontal
                ax.yaxis.label.set_rotation(0)
                # to make sure y axis names are outside the plot area
                ax.yaxis.labelpad = 50
        # plt.yticks(rotation=90)
        vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename="Scatter-Matrix")
    else:
        warnings.warn("Inputmatrix is singular and cannot be calculated. ")


def rescale_features(features):
    '''


    '''

    scaler = preprocessing.StandardScaler()  # Because normal distribution. Don't use minmax scaler for PCA or unsupervised learning
    # as the axis shall be centered and not shifted.
    scaler.fit(features)
    # Use this scaler also for the test data at the end
    X_scaled = pd.DataFrame(data=scaler.transform(features), index=features.index, columns=features.columns)
    print("Unscaled values")
    print(features.iloc[0:2, :])
    print("Scaled values")
    print(X_scaled.iloc[0:2, :])

    return X_scaled

def rescale_outcomes(conf, features, y):
    '''
    
    
    '''
    scaler = preprocessing.StandardScaler()
    scaler.fit(y.reshape(-1, 1))
    y_scaled = pd.DataFrame(data=scaler.transform(y.reshape(-1, 1)), index=features.index, columns=[conf['Common'].get('class_name')])
    print("Unscaled values")
    print(y[0:10])
    print("Scaled values")
    print(y_scaled.iloc[0:10, :])

    return y_scaled

def main(config_path):
    conf = sup.load_config(config_path)
    features, y, df_y, class_labels = sup.load_features(conf)

    #source_filename = os.path.join(conf['Preparation'].get("source_in"))
    #source = sup.load_data_source(source_filename)

    image_save_directory = conf['Paths'].get('results_directory') + "/data_preparation"

    #analyze_timegraph(source, features, y, conf, image_save_directory)
    print("WARNING: If a singular matrix occurs in a calculation, probably the outcome is "
          "only one value.")
    analyse_features(features, y, class_labels, conf, image_save_directory)




if __name__ == "__main__":
    main(args.config_path)


    print("=== Program end ===")