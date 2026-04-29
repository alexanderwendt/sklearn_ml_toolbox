#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 6.0: Evaluate the trained model.

This script evaluates the performance of the trained machine learning model on the
validation data. It generates various evaluation plots, such as confusion
matrices, precision-recall curves, and ROC curves, to assess the model's
performance.

Inputs:
    - Validation data.
    - The trained model.
    - External parameters, such as the precision/recall threshold.

Outputs:
    - Various evaluation plots saved in the `model_images` subdirectory of the
      results directory.

Main Functions:
    - `evaluate_model`: Loads the data and the model, performs the evaluation,
      and generates the evaluation plots.

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
import utils.sklearn_utils as model_util
import argparse
from pandas.plotting import register_matplotlib_converters
import pickle
import numpy as np

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup
import utils.evaluation_utils as evalutil
from filepaths import Paths

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

parser = argparse.ArgumentParser(description='Step 6.0 - Evaluation model')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
parser.add_argument("-sec", '--config_section', default="EvaluationTraining",
                    help='Configuration section in config file', required=False)

args = parser.parse_args()


def evaluate_model(config_path, config_section="EvaluationTraining"):
    '''


    '''
    # Get data
    config = sup.load_config(config_path)
    print("Load paths")
    paths = Paths(config).paths

    X_val, y_val, labels, model, external_params = evalutil.load_evaluation_data(config, config_section)
    y_classes = labels

    result_directory = paths['results_directory']
    #model_name = config['Common'].get('dataset_name')

    title = config.get(config_section, 'title')

    figure_path_prefix = result_directory + '/model_images/' + title
    os.makedirs(result_directory + '/model_images', exist_ok=True)

    problem_type = config['Common'].get('problem_type', fallback='classification')

    # Load model
    print("Predict validation data")
    y_test_pred = model.predict(X_val.values)

    if problem_type == 'classification':
        # Load model external parameters
        pr_threshold = external_params['pr_threshold']
        print("Loaded precision/recall threshold: ", pr_threshold)

        #If there is an error here, set model_pipe['svm'].probability = True
        y_test_pred_proba = model.predict_proba(X_val.values)
        y_test_pred_scores = y_test_pred_proba[:,1] #model.decision_function(X_val.values)

        #Reduce the number of classes only to classes that can be found in the data
        #reduced_class_dict_train = model_util.reduce_classes(y_classes, y_train, y_train_pred)
        reduced_class_dict_test = model_util.reduce_classes(y_classes, y_val, y_test_pred)

        if len(y_classes) == 2:
            #y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)  # (y_train_pred_scores>=pr_threshold).astype('int')
            y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)  # (y_test_pred_scores>=pr_threshold).astype('int')
            print("This is a binarized problem. Apply optimal threshold to precision/recall. Threshold=", pr_threshold)
        else:
            #y_train_pred_adjust = y_train_pred
            y_test_pred_adjust = y_test_pred
            print("This is a multi class problem. No precision/recall adjustment of scores are made.")

        #Plot graphs
        #If binary class plot precision/recall
        # Plot the precision and the recall together with the selected value for the test set
        if len(y_classes) == 2:
            print("Plot precision recall graphs")
            precision, recall, thresholds = precision_recall_curve(y_val, y_test_pred_scores)
            vis.plot_precision_recall_vs_threshold(precision, recall, thresholds, pr_threshold,
                                                   save_fig_prefix=figure_path_prefix, title_prefix="pr_adjusted")

            vis.plot_precision_recall_evaluation(y_val, y_test_pred_adjust, y_test_pred_proba, reduced_class_dict_test,
                                                 save_fig_prefix_dir=figure_path_prefix, title_prefix="pr_adjusted")

        #Plot evaluation for unadjusted values
        vis.plot_precision_recall_evaluation(y_val, y_test_pred, y_test_pred_proba, reduced_class_dict_test,
                                             save_fig_prefix_dir=figure_path_prefix, title_prefix="")
        #Plot decision boundary plot
        X_decision = X_val.values[0:1000, :]
        y_decision = y_val[0:1000]
        vis.plot_decision_boundary(X_decision, y_decision, model, title_prefix=title + "_", save_fig_prefix=figure_path_prefix)

    elif problem_type == 'regression':
        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_val, y_test_pred)
        mae = mean_absolute_error(y_val, y_test_pred)
        r2 = r2_score(y_val, y_test_pred)

        print(f"Regression Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        with open(os.path.join(result_directory, title + "_regression_metrics.txt"), 'w') as f:
            f.write(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\n")

        # Regression plots
        vis.plot_regression_results(y_val, y_test_pred, title=title + ' Regression Results',
                                     save_fig_prefix=figure_path_prefix)
        vis.plot_residuals(y_val, y_test_pred, title=title + ' Residuals',
                            save_fig_prefix=figure_path_prefix)

    print("Visualization complete")


if __name__ == "__main__":
    evaluate_model(args.config_path, args.config_section)


    print("=== Program end ===")