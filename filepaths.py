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

class Config:

    def __init__(self, config_file_path):
        self.conf = sup.load_config(config_file_path)

    def load_labels(self):
        return pd.read_csv(self.conf["Paths"].get("labels_path"), sep=';', header=None).set_index(0).to_dict()[1]

class Paths:

    def __init__(self, config):
        # Instance Variable
        #conf = sup.load_config(config_file_path)
        self.paths = self.generate_paths(config)

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

        # Custom settings - Model name #
        dataset_name = conf['Common'].get('dataset_name')
        class_name = conf['Common'].get('class_name')
        model_type = conf['Common'].get('model_type')

        dataset_class_prefix = dataset_name # + "_" + class_name
        paths['dataset_name'] = conf['Common'].get('dataset_name')
        paths['source_path'] = conf["Paths"].get("source_path")
        paths['labels_path'] = conf["Paths"].get("labels_path")

        ## Generating directories
        subdir = dataset_name + "/" + model_type

        #paths['annotations_directory'] = conf['Paths'].get('annotations_directory')
        ## Data path
        paths['prepared_data_directory'] = os.path.join("prepared-data", dataset_name) #conf['Paths'].get('prepared_data_directory')
        paths['config_directory'] = os.path.join("config")
        #paths['inference_data_directory'] = conf['Paths'].get('inference_data_directory')
        #Model specific directory
        paths['models_directory'] = os.path.join("models", subdir) #conf['Paths'].get('model_directory')
        #paths['result_data_directory'] = os.path.join("results", dataset_name) #conf['Paths'].get('result_directory')
        paths['results_directory'] = os.path.join("results", subdir)

        ## Create directories
        #if os.path.isdir(paths['model_directory'])==False:
        os.makedirs(paths['models_directory'], exist_ok=True)
        #os.makedirs(paths['result_data_directory'], exist_ok=True)
        os.makedirs(paths['results_directory'], exist_ok=True)
        os.makedirs(paths['prepared_data_directory'], exist_ok=True)
        os.makedirs(paths['config_directory'], exist_ok=True)
        #print("Created directory ", paths['model_directory'])

        print("Directories")
        print("Prepared data directory: ", paths['prepared_data_directory'])
        print("Model directory: ", paths['models_directory'])
        print("Results directory: ", paths['results_directory'])
        print("Configs directory: ", paths['config_directory'])

        ## Dump file name
        paths['paths'] = os.path.join(paths['config_directory'], "paths.pickle")

        # Prepared data #
        paths['train_record'] = os.path.join(paths['prepared_data_directory'], "train.record")
        paths['test_record'] = os.path.join(paths['prepared_data_directory'], "test.record")
        paths['inference_record'] = os.path.join(paths['prepared_data_directory'], "inference.record")
        paths['copied_source_path'] = os.path.join(paths['prepared_data_directory'], "source.csv")

        ## Generating filenames for loading the files ##
        paths['model_features_filename'] = os.path.join(paths['prepared_data_directory'], "features_for_model.csv")
        paths['model_outcomes_filename'] = os.path.join(paths['prepared_data_directory'], "outcomes_for_model.csv")
        #paths['source_filename'] = paths['source_path']
        #paths['inference_features_filename'] = paths['inference_data_directory'] + "/" + dataset_class_prefix + "_inference_features" + ".csv"

        ## Modified labels ##
        paths['model_labels_filename'] = os.path.join(paths['prepared_data_directory'], "labels_for_model.csv")
        #Columns for feature selection
        paths['selected_feature_columns_filename'] = os.path.join(paths['prepared_data_directory'], "selected_feature_columns.csv")

        # Model specifics #
        paths['model_input'] = os.path.join(paths['models_directory'], "model.pickle")
        paths['evaluated_model_filename'] = os.path.join(paths['models_directory'], "evaluated_model.sav")
        paths['final_model_filename'] = os.path.join(paths['models_directory'], "final_model.sav")
        paths['external_parameters_filename'] = os.path.join(paths['models_directory'], "model_ext_parameters.json")
        #paths['svm_default_hyper_parameters_filename'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model_hyper_parameters" + ".json"

        paths['pipe_first_selection'] = os.path.join(paths['models_directory'], 'pipe_run1_selection.pkl')
        paths['pipe_final_selection'] = os.path.join(paths['models_directory'], 'pipe_run2_final.pkl')

        #Results
        paths['run1_result_filename'] = os.path.join(paths['results_directory'], 'results_run1.pkl')
        paths['run2_result_filename'] = os.path.join(paths['results_directory'], 'results_run2.pkl')

        # Source data files folder paths
        #paths['source_path_inference'] = paths['inference_data_directory'] + "/" + dataset_name + "_source" + ".csv"

        print("=== Paths ===")
        print("Used file paths: ", paths)

        return paths