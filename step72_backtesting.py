#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 7 Predict Temporal Data: Predict model for temporal data
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

from backtesting import Backtest, Strategy

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

def get_y(data):
    """Return dependent variable y"""
    y = data['y_val']
    return y

class ModelTrade(Strategy):
    step = 0

    def init(self, y_values=None):
        # Plot y for inspection
        self.I(get_y, self.data.df, name='y_val')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

    def next(self):

        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        forecast = self.data.df.loc[current_time].y_val

        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast

        print("Start Position Long: {}, Size: {}".format(self.position.is_long, self.position.size))

        if forecast == 1 and not self.position.is_long:
            self.buy(size=1.0)
            print("Buy: time {}, price {}".format(current_time, self.data.df.loc[current_time].Close))
        elif forecast == 0 and self.position.is_long:
            self.sell(size=0.1)
            print("Sell: time {}, price {}".format(current_time, self.data.df.loc[current_time].Close))

        print("End Position Long: {}, Size: {}. Close: {}. Trades: {}".format(self.position.is_long, self.position.size, self.data.df.loc[current_time].Close, self.trades))

def backTestModel(data, save_folder, title):
    """
    Perform the backtest for a model

    """

    bt = Backtest(data, ModelTrade, commission=.000, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    reference_csv_path = os.path.join(save_folder, title + "_backtest.csv")
    stats.to_csv(reference_csv_path, sep=';')
    reference_graph_path = os.path.join(save_folder, title + "_backtest")
    bt.plot(filename=reference_graph_path)

def backtest(config_path, config_section):
    """
    Backtesting

    """

    conf = sup.load_config(config_path)
    print("Load paths")
    #paths = Paths(conf).paths
    #title = conf.get(config_section, 'title')

    X_val, y_val, labels, model, external_params = eval.load_evaluation_data(conf, config_section)

    #model_name = conf['Common'].get('dataset_name')
    source_path = conf[config_section].get('source_in')
    #result_directory = paths['results_directory']
    save_folder = os.path.join(conf['Paths'].get('results_directory'), "evaluation")

    pred_outcomes_path = os.path.join(conf['Paths'].get('results_directory'), "evaluation", "outcomes_backtest.csv")
    y_values = pd.read_csv(pred_outcomes_path, sep=';').set_index('id')

    df_time_graph = stock.load_ohlc_graph(source_path)
    df_time_graph_cut = df_time_graph.loc[X_val.index]


    # Backtest validation data
    data = df_time_graph_cut.join(y_values['val'])
    data = data.rename(columns={"val": "y_val"})
    data.set_index(['Date'], inplace=True)

    backTestModel(data, save_folder, 'Validation')

    # Backtest reference data MA200
    data = df_time_graph_cut.join(y_values['SMA200'])
    data = data.rename(columns={"SMA200": "y_val"})
    data.set_index(['Date'], inplace=True)

    backTestModel(data, save_folder, 'Reference_SMA200')

    # Backtest prediction data
    data = df_time_graph_cut.join(y_values['model'])
    data = data.rename(columns={"model": "y_val"})
    data.set_index(['Date'], inplace=True)

    backTestModel(data, save_folder, 'Predicted_Model')

    # Backtest prediction data with post processing
    data = df_time_graph_cut.join(y_values['model_pp'])
    data = data.rename(columns={"model_pp": "y_val"})
    data.set_index(['Date'], inplace=True)

    backTestModel(data, save_folder, 'Predicted_Smoothed_Model')

    print("Complete")

if __name__ == "__main__":
    backtest(args.config_path, args.config_section)

    print("=== Program end ===")