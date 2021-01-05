#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File paths class, which is used to create the training files.
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
from pandas.plotting import register_matplotlib_converters
from pickle import dump #Save data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, f1_score, confusion_matrix

# Own modules
#import data_visualization_functions as vis
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

class Config:

    def __init__(self, config_file_path):
        self.conf = sup.load_config(config_file_path)

    def load_labels(self):
        return pd.read_csv(self.conf["Paths"].get("labels_path"), sep=';', header=None).set_index(0).to_dict()[1]

class Paths:

    def __init__(self, config):
        # Instance Variable
        #conf = sup.load_config(config_file_path)
        self.path = self.generate_paths(config)

    def generate_paths(self, conf):
        '''
        Generate paths to save the files that we generate during the machine learning cycle. It adapt paths according to
        the configuration file, e.g. considering the dataset name in the model path

        :args:
            conf: Configuration file
        :return:
            paths: dict with paths adapted to the configuration

        '''


        # Define constants from config input
        # Constants for all notebooks in the Machine Learning Toolbox
        paths = dict()

        # Model name
        dataset_name = conf['Common'].get('dataset_name')
        class_name = conf['Common'].get('class_name')
        dataset_class_prefix = dataset_name + "_" + class_name
        paths['dataset_name'] = conf['Common'].get('dataset_name')

        # Generating directories
        print("Directories")
        #paths['annotations_directory'] = conf['Paths'].get('annotations_directory')
        paths['prepared_data_directory'] = conf['Paths'].get('prepared_data_directory')
        #paths['inference_data_directory'] = conf['Paths'].get('inference_data_directory')
        paths['model_directory'] = conf['Paths'].get('model_directory')
        paths['result_directory'] = conf['Paths'].get('result_directory')
        paths['config_directory'] = "config"

        if os.path.isdir(paths['model_directory'])==False:
            os.makedirs(paths['model_directory'])
            print("Created directory ", paths['model_directory'])

        print("Prepared data directory: ", paths['prepared_data_directory'])
        print("Model directory: ", paths['model_directory'])
        print("Results directory: ", paths['result_directory'])

        # Dump file name
        paths['model_input'] = paths['model_directory'] + "/" + "model.pickle"
        paths['paths'] = paths['config_directory'] + "/" + "paths.pickle"
        paths['train_record'] = paths['prepared_data_directory'] + "/" + "train.record"
        paths['test_record'] = paths['prepared_data_directory'] + "/" + "test.record"
        paths['inference_record'] = paths['prepared_data_directory'] + "/" + "inference.record"

        # Generating filenames for loading the files
        paths['model_features_filename'] = paths['prepared_data_directory'] + "/" + dataset_class_prefix + "_features_for_model" + ".csv"
        paths['model_outcomes_filename'] = paths['prepared_data_directory'] + "/" + dataset_class_prefix + "_outcomes_for_model" + ".csv"
        paths['source_filename'] = paths['prepared_data_directory'] + "/" + dataset_name + "_source" + ".csv"
        #paths['inference_features_filename'] = paths['inference_data_directory'] + "/" + dataset_class_prefix + "_inference_features" + ".csv"

        #Modified labels
        paths['model_labels_filename'] = paths['prepared_data_directory'] + "/" + dataset_class_prefix + "_labels_for_model" + ".csv"
        #Columns for feature selection
        paths['selected_feature_columns_filename'] = paths['prepared_data_directory'] + "/" + dataset_class_prefix + "_selected_feature_columns.csv"

        #Model specifics
        paths['svm_evaluated_model_filename'] = paths['model_directory'] + "/" + dataset_class_prefix + "_svm_evaluated_model" + ".sav"
        paths['svm_final_model_filename'] = paths['model_directory'] + "/" + dataset_class_prefix + "_svm_final_model" + ".sav"
        paths['svm_external_parameters_filename'] = paths['model_directory'] + "/" + dataset_class_prefix + "_svm_model_ext_parameters" + ".json"
        #paths['svm_default_hyper_parameters_filename'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model_hyper_parameters" + ".json"

        paths['svm_pipe_first_selection'] = paths['model_directory'] + "/" + dataset_class_prefix + '_pipe_run1_selection.pkl'
        paths['svm_pipe_final_selection'] = paths['model_directory'] + "/" + dataset_class_prefix + '_pipe_run2_final.pkl'

        #Results
        paths['svm_run1_result_filename'] = paths['result_directory'] + "/" + dataset_class_prefix + '_results_run1.pkl'
        paths['svm_run2_result_filename'] = paths['result_directory'] + "/" + dataset_class_prefix + '_results_run2.pkl'

        # Source data files folder paths
        paths['source_path'] = paths['prepared_data_directory'] + "/" + dataset_name + "_source" + ".csv"
        #paths['source_path_inference'] = paths['inference_data_directory'] + "/" + dataset_name + "_source" + ".csv"

        print("=== Paths ===")
        print("Used file paths: ", paths)

        return paths