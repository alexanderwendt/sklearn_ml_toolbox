#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Define Precision/Recall coefficent for binary classes
License_info: TBD
"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
import json
import os
import time

# Libs
import joblib

import argparse
from pandas.plotting import register_matplotlib_converters
import pickle

import numpy as np

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

parser = argparse.ArgumentParser(description='Step 4.7 - Train evaluation model for final testing')
parser.add_argument("-d", '--data_path', default="config/paths.pickle",
                    help='Prepared data', required=False)

args = parser.parse_args()

def train_final_model(paths_path = "04_Model/paths.pickle"):
    # Get data
    paths, model, train, test = step40.load_training_files(paths_path)

    #print("load inputs: ", data_input_path)
    #f = open(data_input_path, "rb")
    #prepared_data = pickle.load(f)
    #print("Loaded data: ", prepared_data)

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']
    #y_classes = train['label_map']

    svm_pipe_final_selection = paths['svm_pipe_final_selection']
    svm_final_model_filepath = paths['svm_final_model_filename']
    model_directory = paths['model_directory']
    model_name = paths['dataset_name']

    figure_path_prefix = model_directory + '/images/' + model_name

    # Load model external parameters
    #with open(svm_external_parameters_filename, 'r') as fp:
    #    external_params = json.load(fp)

    #pr_threshold = external_params['pr_threshold']
    #print("Loaded precision/recall threshold: ", pr_threshold)

    # Load model
    r = open(svm_pipe_final_selection, "rb")
    model_pipe = pickle.load(r)
    #model_pipe['svm'].probability = True
    print("")
    print("Original final pipe: ", model_pipe)

    #Merge training and test data
    X = X_train.append(X_test)
    y = np.append(y_train, y_test)
    print("Merge training and test data from sizes train {} and test {} to all data {}".format(
        X_train.shape, X_train.shape, X.shape
    ))

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    clf = model_pipe.fit(X_train, y_train)
    t_end = time.time() - t
    print("Training took {0:.2f}s".format(t_end))

    print("Store model")
    print("Model to save: ", clf)

    joblib.dump(clf, svm_final_model_filepath)
    print("Saved model at location ", svm_final_model_filepath)


def main():
    train_final_model(paths_path=args.data_path)


if __name__ == "__main__":

    main()


    print("=== Program end ===")