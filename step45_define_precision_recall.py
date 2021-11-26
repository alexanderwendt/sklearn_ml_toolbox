#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Define Precision/Recall coefficent for binary classes
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
import json
import os
import time

# Libs
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

import utils.sklearn_utils as model_util

import argparse
from pandas.plotting import register_matplotlib_converters
import pickle

import numpy as np

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup
import utils.execution_utils as exe

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

parser = argparse.ArgumentParser(description='Step 4 - Define precision/recall')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()

def get_optimal_precision_recall_threshold(X_train_full, y_train_full, y_classes, model_pipe, figure_path_prefix):
    '''


    args:
        X_train_full: Trainingdaten X
        y_train_full: Ground truth y
        y_classes: Class dict key number, value label
        model_pipe: Pipe of the model
        figure_path_prefix: Prefix path for saving images of the graphs

    return:
        optimal_threshold: Threshold calculated to be the optimum

    '''

    # Split the training set in training and cross validation set
    ### WARNING: Data is not shuffled for this example ####
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, random_state=0, test_size=0.2,
                                                                shuffle=True, stratify=y_train_full)  # cross validation size 20
    print("Total number of samples: {}. X_trainsub: {}, X_cross: {}, y_trainsub: {}, y_cross: {}".format(
        X_train.shape[0], X_train.shape, X_val.shape, y_train.shape, y_val.shape))

    print("")
    print("Original final pipe: ", model_pipe)
    number_of_samples = 1000000

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    optclf = model_pipe.fit(X_train, y_train)

    print("Predict training data")
    y_trainsub_pred = optclf.predict(X_train.values)
    #y_trainsub_pred_scores = optclf.decision_function(X_train.values)
    y_trainsub_pred_proba = optclf.predict_proba(X_train.values)

    print("Predict y_val")
    y_val_pred = optclf.predict(X_val.values)

    print("Predict probabilities and scores of validation data")
    y_val_pred_proba = optclf.predict_proba(X_val.values)
    #y_val_pred_scores = optclf.decision_function(X_val.values)
    #y_val_pred_scores = y_val_pred_proba[:,1]
    print('Model properties: ', optclf)

    reduced_class_dict = model_util.reduce_classes(y_classes, y_val, y_val_pred)

    vis.plot_precision_recall_evaluation(y_train, y_trainsub_pred, y_trainsub_pred_proba, reduced_class_dict,
                                         figure_path_prefix, title_prefix="training_data")
    vis.plot_precision_recall_evaluation(y_val, y_val_pred, y_val_pred_proba, reduced_class_dict,
                                         figure_path_prefix, title_prefix="validation_data")

    #y_val_pred_proba[:,1]: probability estimations of the positive class
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_proba[:,1])
    # custom_threshold = 0.25

    # Get the optimal threshold
    closest_zero_index = np.argmin(np.abs(precision - recall))
    optimal_threshold = thresholds[closest_zero_index]
    closest_zero_p = precision[closest_zero_index]
    closest_zero_r = recall[closest_zero_index]

    print("Optimal threshold value = {0:.2f}".format(optimal_threshold))
    y_val_pred_roc_adjusted = model_util.adjusted_classes(y_val_pred_proba[:,1], optimal_threshold)

    vis.precision_recall_threshold(y_val_pred_roc_adjusted, y_val, precision, recall, thresholds, optimal_threshold,
                                   save_fig_prefix=figure_path_prefix, title_prefix="opt_pr")
    vis.plot_precision_recall_vs_threshold(precision, recall, thresholds, optimal_threshold,
                                           save_fig_prefix=figure_path_prefix, title_prefix="opt_pr")
    print("Optimal threshold value = {0:.2f}".format(optimal_threshold))

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, auc_thresholds = roc_curve(y_val, y_val_pred_proba[:,1])
    print("AUC without P/R adjustments: ", auc(fpr, tpr))  # AUC of ROC
    vis.plot_roc_curve(fpr, tpr, label='ROC', title_prefix="Unadjusted_", save_fig_prefix=figure_path_prefix)

    fpr, tpr, auc_thresholds = roc_curve(y_val, y_val_pred_roc_adjusted)
    print("AUC with P/R adjustments: ", auc(fpr, tpr))  # AUC of ROC
    vis.plot_roc_curve(fpr, tpr, label='ROC', title_prefix="Adjusted_", save_fig_prefix=figure_path_prefix)

    print("Classification report without threshold adjustment.")
    print(classification_report(y_val, y_val_pred, target_names=list(reduced_class_dict.values())))
    print("=========================================================")
    print("Classification report with threshold adjustment of {0:.4f}".format(optimal_threshold))
    print(classification_report(y_val, y_val_pred_roc_adjusted, target_names=list(reduced_class_dict.values())))

    # Summarize optimal results
    print("Optimal score threshold: {0:.2f}".format(optimal_threshold))

    return optimal_threshold

def define_precision_recall_threshold(config_path):
    '''
    Load model data and training data. Check if the problem is a multiclass or single class,
    the precision/recall threshold and save it to a file.

    args:
        data_input_path: Path for pickle file with data. Optional

    return:
        optimal_threshold: Optimal precision/recall threshold
    '''

    # Get data
    # Load file paths
    #paths, model, train, test = step40.load_training_files(paths_path)
    config = sup.load_config(config_path)
    # Load file paths
    # paths, model, train, test = step40.load_training_files(paths_path)
    # Load complete training input
    X_train, y_train, X_val, y_val, y_classes, selected_features, \
    feature_dict, paths, scorers, refit_scorer_name = exe.load_training_input_input(config)

    pipe_final_selection = config['Training'].get('pipeline_out')
    #Use config parameters
    svm_external_parameters_filename = config['Training'].get('ext_param_out')
    result_directory = paths['results_directory']

    figure_path_prefix = result_directory + '/model_images/'
    #if not os.path.isdir(result_directory + '/model_images'):
    os.makedirs(result_directory + '/model_images', exist_ok=True)
    #    print("Created folder: ", result_directory + '/model_images')

    # Check if precision recall can be applied, i.e. it is a binary problem
    if len(y_classes) > 2:
        print("The problem is a multi class problem. No precision/recall optimization will be done.")
        optimal_threshold = 0
    else:
        print("The problem is a binary class problem. Perform precision/recall analysis.")

        # Load model
        # Load saved results
        r = open(pipe_final_selection, "rb")
        model_pipe = pickle.load(r)
        model_pipe['model'].probability = True

        optimal_threshold = get_optimal_precision_recall_threshold(X_train, y_train, y_classes, model_pipe, figure_path_prefix)

    #Store optimal threshold
    # save the optimal precision/recall value to disk
    print("Save external parameters, precision recall threshold to disk")
    extern_param = {}
    extern_param['pr_threshold'] = float(optimal_threshold)
    with open(svm_external_parameters_filename, 'w') as fp:
        json.dump(extern_param, fp)

    return optimal_threshold


if __name__ == "__main__":

    # Define precision/recall
    define_precision_recall_threshold(args.config_path)

    print("=== Program end ===")