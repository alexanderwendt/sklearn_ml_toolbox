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

register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)
#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 4.1 - Prepare data for machine learning algorithms')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
parser.add_argument("-i", "--do_inference", action='store_true',
                    help="Set inference if only inference and no training")

args = parser.parse_args()

def generate_paths(conf):
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

def load_files(paths, do_inference):
    '''
    Load files

    :args:
        paths: Dictionary of paths
    :return:
        df_X: Training data with features as dataframe X
        df_y: Labels y as dataframe
        y_classes: dict of class labels and assigned integers
        df_feature_columns: columns of column names for feature selection e.g. through lasso. The type is dataframe
    '''
    if do_inference==True:  #If inference is selected, only X file shall be loaded
        #=== Load features from X ===#
        df_X = pd.read_csv(paths['inference_features_filename'], delimiter=';').set_index('id') #Read inference data
        # === Load y values ===#
        y = np.zeros(df_X.shape[0]) #For inference, just create an empty y as there is no y
    else:
        # === Load features from X ===#
        df_X = pd.read_csv(paths['model_features_filename'], delimiter=';').set_index('id')   #Read training data

        # === Load y values ===#
        df_y = pd.read_csv(paths['model_outcomes_filename'], delimiter=';').set_index('id')
        y = df_y.values.flatten()

    print("Loaded feature names for X={}".format(df_X.columns))
    print("X. Shape={}".format(df_X.shape))
    print("Indexes of X={}".format(df_X.index.shape))
    print("y. Shape={}".format(y.shape))

    #=== Load classes ===#
    df_y_classes = pd.read_csv(paths['model_labels_filename'], delimiter=';', header=None)
    y_classes_source = sup.inverse_dict(df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1])
    print("Loaded classes into dictionary: {}".format(y_classes_source))

    #=== Load list of feature columns ===#
    df_feature_columns = pd.read_csv(paths['selected_feature_columns_filename'], delimiter=';')
    print("Selected features: {}".format(paths['selected_feature_columns_filename']))
    print(df_feature_columns)

    if do_inference==False:
        print("Inference is done. Therefore no handling of classes.")
        y_classes = get_class_information(y, y_classes_source)
    else:
        y_classes = y_classes_source

    return df_X, y, y_classes, df_feature_columns


def get_class_information(y, y_classes_source):
    '''
    Get class information and remove unused classes.

    :args:
        y: y values for each feature x
        y_classes_source: class labels
    :return:
        y_classes: class to number assignment as dictionary

    '''

    a, b = np.unique(y, return_counts=True)
    vfunc = np.vectorize(lambda x: y_classes_source[x])  # Vectorize a function to process a whole array
    print("For the classes with int {} and names {} are the counts {}".format(a, vfunc(a), b))
    y_classes = {}
    for i in a:
        y_classes[i] = y_classes_source[i]
    print("The following classes remain", y_classes)

    # Check if y is binarized
    if len(y_classes) == 2 and np.max(list(y_classes.keys())) == 1:
        is_multiclass = False
        print("Classes are binarized, 2 classes.")
    else:
        is_multiclass = True
        print(
            "Classes are not binarized, {} classes with values {}. For a binarized class, the values should be [0, 1].".
            format(len(y_classes), list(y_classes.keys())))

    return y_classes


def create_feature_dict(df_feature_columns, df_X):
    '''
    Create a dictionary of feature selection names and the column numbers.

    :args:
        df_feature_columns: column numbers for a feature selection method as dataframe
        df_X: features as dataframe
    :return:
        feature_dict: feature selection method name assigned to a list of column numbers as dictionary

    '''

    # Create a list of column indices for the selection of features
    selected_features = [sup.getListFromColumn(df_feature_columns, df_X, i) for i in range(0, df_feature_columns.shape[1])]
    feature_dict = dict(zip(df_feature_columns.columns, selected_features))

    return feature_dict

def create_training_validation_data(df_X, y, y_classes, df_feature_columns, test_size=0.2, shuffle_data=False):
    '''
    Create a dict with prepared data like models, training data and paths. This object can then be put into a pickle structure

    :args:
        df_X: matrix with features, X as dataframe
        y: label numbers for X
        y_classes: label to class number as dictionary
        df_feature_columns: column number for each feature selection method as dataframe
        paths:
    :return:
        prepared_data: input data for a machine learning algorithm as dataframe

    '''

    #Split test and training sets
    ### WARNING: Data is not shuffled for this example ####
    #X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state=0, test_size=0.2, shuffle=True, stratify = y) #cross validation size 20
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state=0, test_size=test_size, shuffle=shuffle_data) #cross validation size 20
    print("Total number of samples: {}. X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(df_X.shape[0], X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    #Check if training and test data have all classes
    if len(np.unique(y_train))==1:
        raise Exception("y_train only consists one class after train/test split. Please adjust the data.")
    if len(np.unique(y_test)) == 1:
        raise Exception("y_test only consists one class after train/test split. Please adjust the data.")
    #
    #Create scorers to be used in GridSearchCV and RandomizedSearchCV

    #average_method = 'micro' #Add all FP1..FPn together and create Precision or recall from them. It is like weighted classes
    average_method = 'macro' #Calculate Precision1...Precisionn and Recall1...Recalln separately and average. It is good to increase
    #the weight of smaller classes

    scorers = {
        'precision_score': make_scorer(precision_score, average=average_method),
        'recall_score': make_scorer(recall_score, average=average_method),
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average=average_method)
    }

    refit_scorer_name = list(scorers.keys())[3]
    print("Refit scorer name: ", refit_scorer_name)

    # Create feature dictionary
    feature_dict = create_feature_dict(df_feature_columns, df_X)

    # Dump in train and test sets
    train_record = dict()
    train_record['X'] = X_train
    train_record['y'] = y_train
    train_record['label_map'] = y_classes

    test_record = dict()
    test_record['X'] = X_test
    test_record['y'] = y_test
    test_record['label_map'] = y_classes


    # Create prepared data for model
    model_data = dict()
    model_data['scorers'] = scorers
    model_data['refit_scorer_name'] = refit_scorer_name
    model_data['selected_features'] = list(feature_dict.values()) #selected_features
    model_data['feature_dict'] = feature_dict

    return model_data, train_record, test_record

def create_inference_data(df_X, y_classes):
    '''



    '''
    # Dump in train and test sets
    inference_record = dict()
    inference_record['X'] = df_X
    inference_record['label_map'] = y_classes

    return inference_record

def prepare_data(config_file_path, do_inference):
    '''
    Load config file, load training and other files, create a pickle

    :args:
        config_file_path: Filepath for configuration with dataset name and file paths
    :return:
        Nothing

    '''
    conf = sup.load_config(config_file_path)
    paths = generate_paths(conf)
    df_X, y, y_classes, df_feature_columns = load_files(paths, do_inference)

    if do_inference==False:
        print("=== Prepare training ===")
        # Save results
        model_data, train_record, test_record = create_training_validation_data(df_X, y, y_classes, df_feature_columns)

        #Dump path data
        dump(paths, open(paths['paths'], 'wb'))
        print("Stored paths to: ", paths['paths'])

        #Dump model prepararations
        dump(model_data, open(paths['model_input'], 'wb'))
        print("Stored model input to: ", paths['model_input'])

        #Dump training data
        dump(train_record, open(paths['train_record'], 'wb'))
        print("Stored train record to: ", paths['train_record'])

        #Dump test data
        dump(test_record, open(paths['test_record'], 'wb'))
        print("Stored test record to: ", paths['test_record'])

    else:
        print("=== Prepare inference ===")
        inference_record = create_inference_data(df_X, y_classes)

        #Dump inference set
        dump(inference_record, open(paths['inference_record'], 'wb'))
        print("Stored inference record to: ", paths['inference_record'])

if __name__ == "__main__":

    prepare_data(args.config_path, args.do_inference)

    print("=== Program end ===")