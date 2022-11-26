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
import sys

# Libs
import argparse
import os
import logging
from pydoc import locate

import numpy as np
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

# Own modules
import utils.data_visualization_functions as vis
import utils.execution_utils as exe
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

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y%m%d %H:%M:%S',
                    handlers=[logging.FileHandler("logs/" + "toolbox" + ".log"), logging.StreamHandler()])
log = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Step 4 - Calculate predictions for training like estimated time')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)
#parser.add_argument("-algo", '--algorithm', default="svm",
#                    help='Select algorithm to test: SVM (svm) or XGBoost (xgboost). Deafult: SVM.', required=False)

args = parser.parse_args()


def run_training_estimation(X_train, y_train, X_test, y_test, scorer, model_clf, image_save_directory=None):
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

    #Set test range
    test_range = list(range(100, 6500+1, 500))
    print("Test range", test_range)

    # SVM model
    # Define the mode
    xaxis, durations, scores = exe.estimate_training_duration(model_clf, X_train, y_train, X_test, y_test, test_range, scorer)

    # Paint figure
    plt.figure()
    plt.plot(xaxis, durations)
    plt.xlabel('Number of training examples')
    plt.ylabel('Duration [s]')
    plt.title("Training Duration")

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='Duration_Samples')

    plt.figure()
    plt.plot(xaxis, scores)
    plt.xlabel('Number of training examples')
    plt.ylabel('F1-Score on cross validation set (=the rest). Size={}'.format(X_test.shape[0]))
    plt.title("F1 Score Improvement With More Data")

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename='F1_Samples')

def run_training_predictors(data_input_path):
    '''


    '''

    config = sup.load_config(data_input_path)

    pipeline_class_name = config.get('Training', 'pipeline_class', fallback=None)
    PipelineClass = locate('models.' + pipeline_class_name + '.ModelParam')
    model_param = PipelineClass()
    if model_param is None:
        raise Exception("Model pipeline could not be found: {}".format('models.' + pipeline_class_name + '.ModelParam'))


    X_train, y_train, X_val, y_val, y_classes, selected_features, \
    feature_dict, paths, scorers, refit_scorer_name = exe.load_training_input_input(config)
    scorer = scorers[refit_scorer_name]

    results_directory = paths['results_directory']
    save_fig_prefix = results_directory + '/model_images'

    #Baseline test
    baseline_results = exe.execute_baseline_classifier(X_train, y_train, X_val, y_val, y_classes, scorer)
    print("Baseline results=", baseline_results)

    #Set classifier and estimate performance
    model_clf = model_param.create_pipeline()['model']
    log.info("{} selected.".format(model_clf))

    run_training_estimation(X_train, y_train, X_val, y_val, scorer, model_clf, save_fig_prefix)


if __name__ == "__main__":
    # Execute analysis
    run_training_predictors(args.config_path)

    print("=== Program end ===")