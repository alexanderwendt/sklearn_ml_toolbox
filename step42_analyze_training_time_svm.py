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
#import os

# Libs
import argparse
import os

import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Own modules
#import data_visualization_functions as vis
#import data_handling_support_functions as sup
import execution_utils as exe
from evaluation_utils import Metrics
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

parser = argparse.ArgumentParser(description='Step 4 - Calculate predictions for training like estimated time')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()


# def load_input(conf):
#     '''
#     Load input model and data from a prepared pickle file
#
#     :args:
#         input_path: Input path of pickle file with prepared data
#     :return:
#         X_train: Training data
#         y_train: Training labels as numbers
#         X_test: Test data
#         y_test: Test labels as numbers
#         y_classes: Class names assigned to numbers
#         scorer: Scorer for the evaluation, default f1
#
#     '''
#     # Load input
#     X_train_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('features_train_in'))
#     y_train_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('outcomes_train_in'))
#     X_val_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('features_val_in'))
#     y_val_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('outcomes_val_in'))
#
#     #paths, model, train, test = step40.load_training_files(paths_path)
#     X_train, _, y_train = step40.load_data(X_train_path, y_train_path)
#     X_val, _, y_val = step40.load_data(X_val_path, y_val_path)
#
#     labels = step40.load_labels(conf['Paths'].get('labels_path'))
#
#
#     #X_train = train['X']
#     #y_train = train['y']
#     #X_test = test['X']
#     #y_test = test['y']
#     y_classes = labels #train['label_map']
#     metrics = Metrics(conf)
#     scorers = metrics.scorers #model['scorers']
#     refit_scorer_name = metrics.refit_scorer_name #model['refit_scorer_name']
#     scorer = scorers[refit_scorer_name]
#
#     return X_train, y_train, X_val, y_val, y_classes, scorer


def run_training_estimation(X_train, y_train, X_test, y_test, scorer, image_save_directory=None):
    '''
    Run estimation of scorer (default f1) and duration dependent of subset size of input data

    :args:
        X_train: Training data
        y_train: Training labels as numbers
        X_test: Test data
        y_test: Test labels as numbers
        scorer: Scorer for the evaluation, default f1
    :return:
        Nothing

    '''
    # Estimate training duration
    # run_training_estimation = True

    #if run_training_estimation==True:
        #Set test range
    test_range = list(range(100, 6500+1, 500))
    #test_range = list(range(100, 1000, 200))
    print("Test range", test_range)

    # SVM model
    # Define the model
    model_clf = SVC()
    xaxis, durations, scores = exe.estimate_training_duration(model_clf, X_train, y_train, X_test, y_test, test_range, scorer)

    # Paint figure
    plt.figure()
    plt.plot(xaxis, durations)
    plt.xlabel('Number of training examples')
    plt.ylabel('Duration [s]')
    plt.title("Training Duration")

    if image_save_directory:
        if not os.path.isdir(image_save_directory):
            os.makedirs(image_save_directory)
        plt.savefig(os.path.join(image_save_directory, 'SVM_Duration_Samples'), dpi=300)

    plt.show(block = False)
    plt.pause(0.1)
    plt.close()

    plt.figure()
    plt.plot(xaxis, scores)
    plt.xlabel('Number of training examples')
    plt.ylabel('F1-Score on cross validation set (=the rest). Size={}'.format(X_test.shape[0]))
    plt.title("F1 Score Improvement With More Data")

    if image_save_directory:
        if not os.path.isdir(image_save_directory):
            os.makedirs(image_save_directory)
        plt.savefig(os.path.join(image_save_directory, 'SVM_F1_Samples'), dpi=300)

    plt.show(block = False)
    plt.pause(0.1)
    plt.close()

def run_training_predictors(data_input_path):
    '''


    '''

    conf = sup.load_config(data_input_path)
    metrics = Metrics(conf)

    X_train, y_train, X_val, y_val, y_classes, selected_features, \
    feature_dict, paths, scorers, refit_scorer_name = exe.load_training_input_input(conf)
    scorer = scorers[refit_scorer_name]

    result_directory = conf['Paths'].get('result_directory')
    save_fig_prefix = result_directory + '/model_images'

    #Baseline test
    baseline_results = exe.execute_baseline_classifier(X_train, y_train, X_val, y_val, y_classes, scorer)
    print("Baseline results=", baseline_results)

    run_training_estimation(X_train, y_train, X_val, y_val, scorer, save_fig_prefix)


if __name__ == "__main__":

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    run_training_predictors(data_input_path=args.config_path)

    print("=== Program end ===")