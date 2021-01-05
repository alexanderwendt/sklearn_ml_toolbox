#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 Evaluation Temporal Data: Evaluate model for temporal data
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

# Libs
import argparse
import json
import joblib
import pandas as pd
import matplotlib.dates as mdates
import sklearn_utils as model_util
from pandas.plotting import register_matplotlib_converters
import numpy as np

# Own modules
import data_visualization_functions as vis
import data_handling_support_functions as sup
import execution_utils as step40
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

parser = argparse.ArgumentParser(description='Step 5.0 - Evaluate Model for Temporal Data')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()


def visualize_temporal_data(config_path):
    # Load intermediate model, which has only been trained on training data
    # Get data
    # Load file paths
    #paths, model, train, test = step40.load_training_files(paths_path)
    config = sup.load_config(config_path)
    #paths, model, train, test = step40.load_training_files(paths_path)

    X_val, y_val, labels, model, external_params = eval.load_evaluation_data(config)

    #X_train = train['X']
    #y_train = train['y']
    #X_test = test['X']
    #y_test = test['y']
    y_classes = labels #train['label_map']

    #svm_pipe_final_selection = prepared_data['paths']['svm_pipe_final_selection']
    #svm_evaluation_model_filepath = paths['svm_evaluated_model_filename']
    #svm_external_parameters_filename = paths['svm_external_parameters_filename']
    #model_directory = paths['model_directory']
    model_name = config['Common'].get('dataset_name')
    source_path = config['Evaluation'].get('source_in') #paths['source_path']
    result_directory = config['Paths'].get('result_directory')

    figure_path_prefix = result_directory + '/evaluation/' + model_name
    if not os.path.isdir(result_directory + '/evaluation'):
        os.makedirs(result_directory + '/evaluation')
        print("Created folder: ", result_directory + '/evaluation')

    # Load model external parameters
    #with open(svm_external_parameters_filename, 'r') as fp:
    #    external_params = json.load(fp)
    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: {0:.2f}".format(pr_threshold))

    # Open evaluation model
    #evalclf = joblib.load(svm_evaluation_model_filepath)
    #print("Loaded trained evaluation model from ", svm_evaluation_model_filepath)
    #print("Model", evalclf)

    # Make predictions
    #y_train_pred_scores = evalclf.decision_function(X_train.values)
    #y_train_pred_proba = evalclf.predict_proba(X_train.values)
    #y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)
    y_test_pred_scores = model.decision_function(X_val.values)
    #y_test_pred_proba = evalclf.predict_proba(X_test.values)
    y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)

    # Load original data for visualization
    df_time_graph = pd.read_csv(source_path, delimiter=';').set_index('id')
    df_time_graph['Date'] = pd.to_datetime(df_time_graph['Date'])
    df_time_graph['Date'].apply(mdates.date2num)
    print("Loaded feature names for time graph={}".format(df_time_graph.columns))
    print("X. Shape={}".format(df_time_graph.shape))

    # Create a df from the y array for the visualization functions
    #y_order_train = pd.DataFrame(index=X_train.index,
    #                             data=pd.Series(data=y_train, index=X_train.index, name="y")).sort_index()

    #y_order_train_pred = pd.DataFrame(index=X_train.index,
    #                                  data=pd.Series(data=y_train_pred_adjust, index=X_train.index, name="y")).sort_index()

    y_order_test = pd.DataFrame(index=X_val.index,
                                data=pd.Series(data=y_val, index=X_val.index, name="y")).sort_index()

    y_order_test_pred = pd.DataFrame(index=X_val.index,
                                     data=pd.Series(data=y_test_pred_adjust, index=X_val.index, name="y")).sort_index()


    #Visualize the results
    print("Plot fpr training data")
    #vis.plot_three_class_graph(y_order_train_pred['y'].values,
    #                           df_time_graph['Close'][y_order_train.index],
    #                           df_time_graph['Date'][y_order_train.index], 0, 0, 0,
    #                           ('close', 'neutral', 'positive', 'negative'),
    #                           save_fig_prefix=figure_path_prefix + "_train_")
    #vis.plot_two_class_graph(y_order_train, y_order_train_pred,
    #                         save_fig_prefix=figure_path_prefix + "_train_")


    print("Plot for test data")
    vis.plot_three_class_graph(y_order_test['y'].values,
                               df_time_graph['Close'][y_order_test.index],
                               df_time_graph['Date'][y_order_test.index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               save_fig_prefix=figure_path_prefix + "_test_gt")

    vis.plot_three_class_graph(y_order_test_pred['y'].values,
                               df_time_graph['Close'][y_order_test.index],
                               df_time_graph['Date'][y_order_test.index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               save_fig_prefix=figure_path_prefix + "_test_")
    #vis.plot_two_class_graph(y_order_test, y_order_test_pred,
    #                         save_fig_prefix=figure_path_prefix + "_test_")


if __name__ == "__main__":
    visualize_temporal_data(args.config_path)

    print("=== Program end ===")