#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Train final model
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
import joblib

import argparse
from pandas.plotting import register_matplotlib_converters
import pickle

import numpy as np

# Own modules
#import data_vsualization_functions as vis
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

parser = argparse.ArgumentParser(description='Step 4.7 - Train evaluation model for final testing')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)
parser.add_argument("-sec", '--config_section', default="Model",
                    help='Configuration section in config file', required=False)

args = parser.parse_args()

def load_data(conf, config_section="Model"):
    '''


    '''

    X_train_path = conf[config_section].get('features_in')
    y_train_path = conf[config_section].get('outcomes_in')
    labels_path = conf[config_section].get('labels_in')
    pipeline_in = conf[config_section].get('pipeline_in')
    ext_param_in = conf[config_section].get('ext_param_in')

    # Load X and y
    X_train, _, y_train = exe.load_data(X_train_path, y_train_path)

    # Labels
    labels = exe.load_labels(labels_path)

    # Load pipe
    r = open(pipeline_in, "rb")
    pipe = pickle.load(r)

    return X_train, y_train, pipe

def train_final_model(config_path, config_section="Evaluation"):
    # Get data
    config = sup.load_config(config_path)
    #paths, model, train, test = step40.load_training_files(paths_path)
    X_train, y_train, pipe = load_data(config)

    #print("load inputs: ", data_input_path)
    #f = open(data_input_path, "rb")
    #prepared_data = pickle.load(f)
    #print("Loaded data: ", prepared_data)

    #X_train = train['X']
    #y_train = train['y']
    #X_test = test['X']
    #y_test = test['y']
    #y_classes = train['label_map']

    #svm_pipe_final_selection = paths['svm_pipe_final_selection']
    svm_final_model_filepath = config[config_section].get('model_out') #paths['svm_final_model_filename']
    #model_directory = paths['model_directory']
    #model_name = paths['dataset_name']

    #figure_path_prefix = model_directory + '/images/' + model_name

    # Load model external parameters
    #with open(svm_external_parameters_filename, 'r') as fp:
    #    external_params = json.load(fp)

    #pr_threshold = external_params['pr_threshold']
    #print("Loaded precision/recall threshold: ", pr_threshold)

    # Load model
    #r = open(svm_pipe_final_selection, "rb")
    #model_pipe = pickle.load(r)
    #model_pipe['svm'].probability = True
    print("")

    print("Set probability measurements in the model to True")
    pipe['model'].probability = True
    print("Original final pipe: ", pipe)

    #Merge training and test data
    #X = X_train.append(X_test)
    #y = np.append(y_train, y_test)
    #print("Merge training and test data from sizes train {} and test {} to all data {}".format(
    #    X_train.shape, X_train.shape, X.shape
    #))

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    clf = pipe.fit(X_train, y_train)
    t_end = time.time() - t
    print("Training took {0:.2f}s".format(t_end))

    print("Store model")
    print("Model to save: ", clf)

    joblib.dump(clf, svm_final_model_filepath)
    print("Saved model at location ", svm_final_model_filepath)


if __name__ == "__main__":

    train_final_model(args.config_path, args.config_section)


    print("=== Program end ===")