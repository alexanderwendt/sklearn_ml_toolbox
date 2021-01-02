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
import execution_utils as step40

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

parser = argparse.ArgumentParser(description='Step 4.6 - Train evaluation model for final testing')
parser.add_argument("-d", '--data_path', default="config/paths.pickle",
                    help='Prepared data', required=False)

args = parser.parse_args()


def train_model_for_evaluation(paths_path = "config/paths.pickle"):
    # Get data
    paths, model, train, test = step40.load_training_files(paths_path)

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']
    y_classes = train['label_map']

    svm_pipe_final_selection = paths['svm_pipe_final_selection']
    svm_evaluation_model_filepath = paths['svm_evaluated_model_filename']
    svm_external_parameters_filename = paths['svm_external_parameters_filename']
    model_directory = paths['model_directory']
    result_directory = paths['result_directory']
    model_name = paths['dataset_name']

    figure_path_prefix = result_directory + '/model_images/' + model_name
    if not os.path.isdir(result_directory + '/model_images'):
        os.makedirs(result_directory + '/model_images')
        print("Created folder: ", result_directory + '/model_images')

    # Load model external parameters
    with open(svm_external_parameters_filename, 'r') as fp:
        external_params = json.load(fp)

    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: ", pr_threshold)

    # Load model
    r = open(svm_pipe_final_selection, "rb")
    model_pipe = pickle.load(r)
    model_pipe['svm'].probability = True
    print("")
    print("Original final pipe: ", model_pipe)
    #number_of_samples = 1000000

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    clf = model_pipe.fit(X_train, y_train)
    t_end = time.time() - t
    print("Training took {0:.2f}s".format(t_end))

    print("Predict training data")
    y_train_pred = clf.predict(X_train.values)
    y_train_pred_scores = clf.decision_function(X_train.values)
    y_train_pred_proba = clf.predict_proba(X_train.values)

    print("Predict test data")
    y_test_pred = clf.predict(X_test.values)
    y_test_pred_proba = clf.predict_proba(X_test.values)
    y_test_pred_scores = clf.decision_function(X_test.values)

    #Reduce the number of classes only to classes that can be found in the data
    reduced_class_dict_train = model_util.reduce_classes(y_classes, y_train, y_train_pred)
    reduced_class_dict_test = model_util.reduce_classes(y_classes, y_test, y_test_pred)

    if len(y_classes) == 2:
        y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)  # (y_train_pred_scores>=pr_threshold).astype('int')
        y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)  # (y_test_pred_scores>=pr_threshold).astype('int')
        print("This is a binarized problem. Apply optimal threshold to precision/recall. Threshold=", pr_threshold)
    else:
        y_train_pred_adjust = y_train_pred
        y_test_pred_adjust = y_test_pred
        print("This is a multi class problem. No precision/recall adjustment of scores are made.")

    print("Model training finished")

    #Plot graphs
    #If binary class plot precision/recall
    # Plot the precision and the recall together with the selected value for the test set
    if len(y_classes) == 2:
        print("Plot precision recall graphs")
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_scores)
        vis.plot_precision_recall_vs_threshold(precision, recall, thresholds, pr_threshold, save_fig_prefix=figure_path_prefix + "_step46_")

    #Plot evaluation
    vis.plot_precision_recall_evaluation(y_train, y_train_pred_adjust, y_train_pred_proba, reduced_class_dict_train,
                                     save_fig_prefix=figure_path_prefix + "_step46_train_")
    vis.plot_precision_recall_evaluation(y_test, y_test_pred_adjust, y_test_pred_proba, reduced_class_dict_test,
                                     save_fig_prefix=figure_path_prefix + "_step46_test_")
    #Plot decision boundary plot
    X_decision = X_train.values[0:1000, :]
    y_decision = y_train[0:1000]
    vis.plot_decision_boundary(X_decision, y_decision, clf, save_fig_prefix=figure_path_prefix + "_step46_test_")

    print("Visualization complete")

    print("Store model")
    print("Model to save: ", clf)

    joblib.dump(clf, svm_evaluation_model_filepath)
    print("Saved model at location ", svm_evaluation_model_filepath)




def main():
    train_model_for_evaluation(paths_path=args.data_path)


if __name__ == "__main__":
    main()


    print("=== Program end ===")