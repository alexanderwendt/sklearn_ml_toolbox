#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Evaluate training model
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
import json
import joblib
from sklearn.metrics import precision_recall_curve
import sklearn_utils as model_util
import argparse
from pandas.plotting import register_matplotlib_converters
import pickle
import numpy as np

# Own modules
import data_visualization_functions as vis
import data_handling_support_functions as sup
import execution_utils as exe
import evaluation_utils as eval

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

parser = argparse.ArgumentParser(description='Step 6.0 - Evaluation model')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()

# def load_evaluation_data(conf):
#     '''
#
#
#     '''
#
#     X_path = conf['Evaluation'].get('features_in')
#     y_path = conf['Evaluation'].get('outcomes_in')
#     labels_path = conf['Evaluation'].get('labels_in')
#     model_in = conf['Evaluation'].get('model_in')
#     ext_param_in = conf['Evaluation'].get('ext_param_in')
#
#     # Load X and y
#     X_val, _, y_val = exe.load_data(X_path, y_path)
#
#     # Labels
#     labels = exe.load_labels(labels_path)
#
#     # Load model
#     model = joblib.load(model_in)
#     print("Loaded trained evaluation model from ", model_in)
#     print("Model", model)
#
#     # Load external parameters
#     with open(ext_param_in, 'r') as fp:
#         external_params = json.load(fp)
#
#     return X_val, y_val, labels, model, external_params


def evaluate_model(config_path):
    '''


    '''
    # Get data
    config = sup.load_config(config_path)

    X_val, y_val, labels, model, external_params = eval.load_evaluation_data(config)

    y_classes = labels #train['label_map']

    result_directory = config['Paths'].get('result_directory')
    model_name = config['Common'].get('dataset_name')

    figure_path_prefix = result_directory + '/model_images/' + model_name
    if not os.path.isdir(result_directory + '/model_images'):
        os.makedirs(result_directory + '/model_images')
        print("Created folder: ", result_directory + '/model_images')

    # Load model external parameters
    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: ", pr_threshold)

    # Load model

    #print("Predict training data")
    #y_train_pred = clf.predict(X_train.values)
    #y_train_pred_scores = clf.decision_function(X_train.values)
    #y_train_pred_proba = clf.predict_proba(X_train.values)

    print("Predict validation data")
    y_test_pred = model.predict(X_val.values)
    #If there is an error here, set model_pipe['svm'].probability = True
    y_test_pred_proba = model.predict_proba(X_val.values)
    y_test_pred_scores = model.decision_function(X_val.values)

    #Reduce the number of classes only to classes that can be found in the data
    #reduced_class_dict_train = model_util.reduce_classes(y_classes, y_train, y_train_pred)
    reduced_class_dict_test = model_util.reduce_classes(y_classes, y_val, y_test_pred)

    if len(y_classes) == 2:
        #y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)  # (y_train_pred_scores>=pr_threshold).astype('int')
        y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)  # (y_test_pred_scores>=pr_threshold).astype('int')
        print("This is a binarized problem. Apply optimal threshold to precision/recall. Threshold=", pr_threshold)
    else:
        #y_train_pred_adjust = y_train_pred
        y_test_pred_adjust = y_test_pred
        print("This is a multi class problem. No precision/recall adjustment of scores are made.")

    #print("Model training finished")

    #Plot graphs
    #If binary class plot precision/recall
    # Plot the precision and the recall together with the selected value for the test set
    if len(y_classes) == 2:
        print("Plot precision recall graphs")
        precision, recall, thresholds = precision_recall_curve(y_val, y_test_pred_scores)
        vis.plot_precision_recall_vs_threshold(precision, recall, thresholds, pr_threshold, save_fig_prefix=figure_path_prefix + "_step46_")

    #Plot evaluation
    #vis.plot_precision_recall_evaluation(y_train, y_train_pred_adjust, y_train_pred_proba, reduced_class_dict_train,
    #                                 save_fig_prefix=figure_path_prefix + "_step46_train_")
    vis.plot_precision_recall_evaluation(y_val, y_test_pred_adjust, y_test_pred_proba, reduced_class_dict_test,
                                     save_fig_prefix=figure_path_prefix + "_step46_test_")
    #Plot decision boundary plot
    X_decision = X_val.values[0:1000, :]
    y_decision = y_val[0:1000]
    vis.plot_decision_boundary(X_decision, y_decision, model, save_fig_prefix=figure_path_prefix + "_step60_test_")

    print("Visualization complete")


if __name__ == "__main__":
    evaluate_model(args.config_path)


    print("=== Program end ===")