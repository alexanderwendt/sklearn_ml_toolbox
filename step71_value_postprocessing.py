#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 71: Postprocess and generate values from the validation data to check how well the system works


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
import utils.sklearn_utils as model_util
from pandas.plotting import register_matplotlib_converters
import numpy as np

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup
import utils.execution_utils as step40
import utils.evaluation_utils as eval
import utils.stock_market_utils as stock

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

from filepaths import Paths

register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)
#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 7 - Predict Temporal Data')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)
parser.add_argument("-sec", '--config_section', default="EvaluationValidation",
                    help='Configuration section in config file', required=False)

args = parser.parse_args()
print(args)

def generate_reference_results(signal_indicator):
    """
    If price>MA200 buy, if not, sell

    """

    return (signal_indicator>0)*1

def generate_values_for_backtesting(config_path, config_section):
    """
    Generate values for backtesting to find out how well the system performs. First, an MA200 reference value is used.
    Second, the predictions are created and then smoothed

    """

    conf = sup.load_config(config_path)
    print("Load paths")
    #paths = Paths(conf).paths
    #title = conf.get(config_section, 'title')

    X_val, y_val, labels, model, external_params = eval.load_evaluation_data(conf, config_section)

    #model_name = conf['Common'].get('dataset_name')
    source_path = conf[config_section].get('source_in')
    #result_directory = paths['results_directory']

    # Load model external parameters
    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: {0:.2f}".format(pr_threshold))

    df_time_graph = stock.load_ohlc_graph(source_path)

    # Y_val
    y_val_data = pd.DataFrame(index=X_val.index, data=y_val, columns=['val'])

    # Reference system
    print("Create reference values")
    y_ref = generate_reference_results(X_val['SMA200'])
    y_red_data = pd.DataFrame(y_ref)

    #Visualize the results
    figure_path_prefix = os.path.join(conf['Paths'].get('results_directory'), "evaluation")
    print("Plot for inference data to ", figure_path_prefix)
    vis.plot_three_class_graph(y_ref,
                               df_time_graph['Close'][y_red_data.index],
                               df_time_graph['Date'][y_red_data.index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               title="Reference_System_SMA200" + "_Validation_",
                               save_fig_prefix=figure_path_prefix)


    # Predictionsystem
    print("Get prediction values")
    y_test_pred = model.predict(X_val.values)
    y_test_pred_data = pd.DataFrame(index=X_val.index, data=y_test_pred, columns=['model'])
    pred_outcomes_filename = os.path.join(conf['Paths'].get('results_directory'), "evaluation", "outcomes_pred.csv")
    #y_test_pred_data.to_csv(pred_outcomes_filename, sep=';', index=True, header=True)

    # Post processing for prediction
    # Use a moving average
    N=3
    smoothed_values_unfixed = np.convolve(y_test_pred_data['model'], np.ones(N) / N, mode='valid')
    smoothed_data_raw = np.concatenate([np.zeros(2), smoothed_values_unfixed])
    smoothed_data = (smoothed_data_raw>0.5)*1

    y_test_pred_data_smoothed = pd.DataFrame(index=X_val.index, data=smoothed_data, columns=['model_pp'])

    print("Plot for post processed inference data to ", figure_path_prefix)
    vis.plot_three_class_graph(smoothed_data,
                               df_time_graph['Close'][y_test_pred_data_smoothed.index],
                               df_time_graph['Date'][y_test_pred_data_smoothed.index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               title="Smoothed_Prediction_Model" + "_Validation_",
                               save_fig_prefix=figure_path_prefix)

    # Merge all values
    all_data = y_test_pred_data.join(y_red_data).join(y_val_data).join(y_test_pred_data_smoothed)
    pred_outcomes_filename = os.path.join(conf['Paths'].get('results_directory'), "evaluation", "outcomes_backtest.csv")
    all_data.to_csv(pred_outcomes_filename, sep=';', index=True, header=True)
    print("Completed")


if __name__ == "__main__":
    generate_values_for_backtesting(args.config_path, args.config_section)

    print("=== Program end ===")