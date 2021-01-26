#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Feature Selection
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

# Libs
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib as m
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFE

# Own modules
import data_visualization_functions as vis
import data_handling_support_functions as sup

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

parser = argparse.ArgumentParser(description='Step 3 - Perform feature selection')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()


def predict_features_simple(X, y):
    '''


    '''

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    return clf.score(X, y)

def execute_lasso_feature_selection(X_scaled, y, conf, image_save_directory):
    '''


    '''

    print("Feature selection with lasso regression")
    reg = LassoCV(cv=10, max_iter=100000)
    reg.fit(X_scaled, y)
    coef = pd.Series(reg.coef_, index=X_scaled.columns)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X_scaled, y))
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()
    coefList = list(imp_coef[imp_coef != 0].index)
    print("Lasso coefficient list\n:", coefList)

    # plt.figure()
    m.rcParams['figure.figsize'] = (8.0, 20.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    plt.tight_layout()

    if image_save_directory:
        if not os.path.isdir(image_save_directory):
            os.makedirs(image_save_directory)
        plt.savefig(os.path.join(image_save_directory, conf['Common'].get('dataset_name') + '_Lasso_Model_Weights'), dpi=300)

    plt.show(block = False)

    return coefList

def execute_treebased_feature_selection(X_scaled, y, conf, image_save_directory):
    '''


    '''

    print("Tree based feature selection")
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_scaled, y)
    print(clf.feature_importances_)
    print("Best score: %f" % clf.score(X_scaled, y))
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_scaled)
    X_new.shape

    threshold = 0.010
    tree_coef = pd.Series(clf.feature_importances_, index=X_scaled.columns)

    print("Tree search picked " + str(sum(tree_coef >= threshold)) + " variables and eliminated the other " + str(
        sum(tree_coef < threshold)) + " variables")
    imp_treecoef = tree_coef.sort_values()
    treecoefList = list(imp_treecoef[imp_treecoef > threshold].index)

    print("Tree based coefficent list:\n", treecoefList)

    plt.figure()
    m.rcParams['figure.figsize'] = (8.0, 20.0)
    imp_treecoef.plot(kind="barh")
    plt.title("Feature importance using Tree Search Model")
    plt.vlines(threshold, 0, len(X_scaled.columns), color='red')
    plt.tight_layout()

    if image_save_directory:
        if not os.path.isdir(image_save_directory):
            os.makedirs(image_save_directory)
        plt.savefig(os.path.join(image_save_directory, conf['Common'].get('dataset_name') + '_Tree_Based_Importance'), dpi=300)

    plt.show(block = False)

    return treecoefList

def execute_backwardelimination_feature_selection(X_scaled, y):
    '''


    '''

    print("Backward elimination")
    cols = list(X_scaled.columns)
    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X_scaled[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols

    print("Selected features:")
    print(selected_features_BE)
    print("\nNumber of features={}. Original number of features={}\n".format(len(selected_features_BE),
                                                                             len(X_scaled.columns)))
    [print("column {} removed".format(x)) for x in X_scaled.columns if x not in selected_features_BE]
    print("Finished")

    return selected_features_BE

def execute_recursive_elimination_feature_selection(X_scaled, y):
    '''


    '''

    print("Recursive elimination")
    model = LogisticRegressionCV(solver='liblinear', cv=3)
    print("Start Recursive Elimination. Fit model with {} examples.".format(X_scaled.shape[0]))
    # Initializing RFE model, 3 features selected
    rfe = RFE(model, 1)  # It has to be one to get a unique index
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X_scaled, y)
    # Fitting the data to model
    model.fit(X_rfe, y)

    print("Best accuracy score using built-in Logistic Regression: ", model.score(X_rfe, y))
    print("Ranking")
    rfe_coef = pd.Series(X_scaled.columns, index=rfe.ranking_ - 1).sort_index()
    print(rfe_coef)

    print("Selected columns")
    print(X_scaled.columns[rfe.support_].values)

    return X_scaled.columns[rfe.support_].values, rfe_coef

def create_feature_list_from_top_features(relevantFeatureList):
    '''
    Use the list of relevant features


    '''
    ### Weighted values
    # Weights
    values, counts = np.unique(relevantFeatureList, return_counts=True)
    s = pd.Series(index=values, data=counts).sort_values(ascending=False)
    print(s)
    ### Add Manually Selected Subset
    # print subset
    newval = [x for x, c in zip(values, counts) if c > 1]
    subsetColumns = newval  # X.columns[rfe.support_].values #list(values)

    print("Subset columns:\n", subsetColumns)

    return subsetColumns


def perform_feature_selection_algorithms(features, y, conf, image_save_directory):
    '''
    Perform feature selection


    '''

    # Scale
    # Use this scaler also for the test data at the end
    X_scaled = pd.DataFrame(data=StandardScaler().fit(features).transform(features),
                            index=features.index, columns=features.columns)

    # Reduce the training set with the number of samples randomly chosen
    X_train_index_subset = sup.get_random_data_subset_index(1000, X_scaled)
    print("Scaled data with standard scaler.")
    print("Get a subset of 1000 samples.")

    relevantFeatureList = []
    selected_feature_list = pd.DataFrame()

    # Predict with logistic regression


    ### Lasso Feature Selection
    m.rc_file_defaults()  # Reset sns
    coefList = execute_lasso_feature_selection(X_scaled, y, conf, image_save_directory)
    selected_feature_list = selected_feature_list.append(pd.Series(name='Lasso', data=coefList))
    relevantFeatureList.extend(coefList)

    print("Prediction of training data with logistic regression: {0:.2f}".format(
        predict_features_simple(X_scaled[coefList], y)))

    ### Tree based feature selection
    treecoefList = execute_treebased_feature_selection(X_scaled, y, conf, image_save_directory)
    selected_feature_list = selected_feature_list.append(pd.Series(name='Tree', data=treecoefList))
    relevantFeatureList.extend(treecoefList)

    print("Prediction of training data with logistic regression: {0:.2f}".format(
        predict_features_simple(X_scaled[treecoefList], y)))

    ### Backward Elimination
    # Backward Elimination - Wrapper method
    selected_features_BE = execute_backwardelimination_feature_selection(X_scaled, y)
    relevantFeatureList.extend(selected_features_BE)
    selected_feature_list = selected_feature_list.append(
        pd.Series(name='Backward_Elimination', data=selected_features_BE))

    print("Prediction of training data with logistic regression: {0:.2f}".format(
        predict_features_simple(X_scaled[selected_features_BE], y)))

    ### Recursive Elimination with Logistic Regression
    # Recursive Elimination - Wrapper method, Feature ranking with recursive feature elimination
    relevant_features, rfe_coef = execute_recursive_elimination_feature_selection(X_scaled.iloc[X_train_index_subset],
                                                                                  y[X_train_index_subset])
    relevantFeatureList.extend(relevant_features)

    step_size = np.round(len(X_scaled.columns) / 4, 0).astype("int")
    for i in range(step_size, len(X_scaled.columns), step_size):
        selected_feature_list = selected_feature_list.append(
            pd.Series(name='RecursiveTop' + str(i), data=rfe_coef.loc[0:i - 1]))
        print('Created RecursiveTop{}'.format(str(i)))

    ### Add the top coloums from all methods
    top_feature_cols = create_feature_list_from_top_features(relevantFeatureList)
    selected_feature_list = selected_feature_list.append(pd.Series(name='Manual', data=top_feature_cols))

    ### Add all columns
    selected_feature_list = selected_feature_list.append(pd.Series(name='All', data=X_scaled.columns))

    return selected_feature_list


def main(config_path):
    conf = sup.load_config(config_path)
    features, y, df_y, class_labels = sup.load_features(conf)

    image_save_directory = conf['Paths'].get('result_directory') + "/data_preparation"

    selected_feature_columns_filename = os.path.join(conf['Paths'].get('prepared_data_directory'),
                                                     conf['Preparation'].get("selected_feature_columns_out"))

    selected_feature_list = perform_feature_selection_algorithms(features, y, conf, image_save_directory)

    print("List of selected features")
    print(selected_feature_list.transpose())

    selected_feature_list.transpose().to_csv(selected_feature_columns_filename, sep=';', index=False, header=True)
    print("Saved selected feature columns to " + selected_feature_columns_filename)


if __name__ == "__main__":
    main(args.config_path)

    print("=== Program end ===")