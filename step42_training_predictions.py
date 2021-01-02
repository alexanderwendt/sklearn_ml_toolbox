#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Feature Selection
License_info: TBD
"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
#import os

# Libs
import argparse
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Own modules
#import data_visualization_functions as vis
#import data_handling_support_functions as sup
import execution_utils as step40

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)
#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 4.2 - Calculate predictions for training like estimated time')
parser.add_argument("-d", '--data_path', default="config/paths.pickle",
                    help='Prepared data', required=False)

args = parser.parse_args()


def load_input(paths_path = "config/paths.pickle"):
    '''
    Load input model and data from a prepared pickle file

    :args:
        input_path: Input path of pickle file with prepared data
    :return:
        X_train: Training data
        y_train: Training labels as numbers
        X_test: Test data
        y_test: Test labels as numbers
        y_classes: Class names assigned to numbers
        scorer: Scorer for the evaluation, default f1

    '''
    # Load input
    paths, model, train, test = step40.load_training_files(paths_path)

    #f = open(input_path,"rb")
    #prepared_data = pickle.load(f)
    #print("Loaded data: ", prepared_data)

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']
    y_classes = train['label_map']
    scorers = model['scorers']
    refit_scorer_name = model['refit_scorer_name']
    scorer = scorers[refit_scorer_name]

    return X_train, y_train, X_test, y_test, y_classes, scorer


def run_training_estimation(X_train, y_train, X_test, y_test, scorer):
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
    xaxis, durations, scores = step40.estimate_training_duration(model_clf, X_train, y_train, X_test, y_test, test_range, scorer)

    # Paint figure
    plt.figure()
    plt.plot(xaxis, durations)
    plt.xlabel('Number of training examples')
    plt.ylabel('Duration [s]')
    plt.title("Training Duration")
    plt.show(block = False)

    plt.figure()
    plt.plot(xaxis, scores)
    plt.xlabel('Number of training examples')
    plt.ylabel('F1-Score on cross validation set (=the rest). Size={}'.format(X_test.shape[0]))
    plt.title("F1 Score Improvement With More Data")
    plt.show(block = False)

def run_training_predictors(data_input_path):
    '''


    '''
    X_train, y_train, X_test, y_test, y_classes, scorer = load_input(data_input_path)

    #Baseline test
    baseline_results = step40.execute_baseline_classifier(X_train, y_train, X_test, y_test, y_classes, scorer)
    print("Baseline results=", baseline_results)

    run_training_estimation(X_train, y_train, X_test, y_test, scorer)


if __name__ == "__main__":

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    run_training_predictors(data_input_path=args.data_path)

    print("=== Program end ===")