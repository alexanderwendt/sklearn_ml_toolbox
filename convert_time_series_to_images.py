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
# from __future__ import print_function

# Built-in/Generic Imports
import json
import os

# Libs
import argparse
import json
import numpy as np
import mplfinance as mpf

from backtesting import Backtest, Strategy

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup
import utils.execution_utils as exe
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

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 7 - Predict Temporal Data')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30_datapreparation_train.ini",
                    help='Configuration file path', required=False)
parser.add_argument("-sec", '--config_section', default="EvaluationValidation",
                    help='Configuration section in config file', required=False)
parser.add_argument('--feature_dir', default="./prepared-data/debug_omxs30_LongTrend",
                    help='Featuredir', required=False)

args = parser.parse_args()
print(args)


def convert_time_series(config_path, feature_dir):
    """


    """

    # Load time series
    config = sup.load_config(config_path)

    X_train_path = os.path.join(feature_dir, "features_val.csv")
    y_train_path = os.path.join(feature_dir, "outcomes_val.csv")
    source_path = os.path.join(feature_dir, "source.csv")

    os.makedirs("./prepared-data/val_rolling_csv", exist_ok=True)
    os.makedirs("./prepared-data/val_rolling_images/pos", exist_ok=True)
    os.makedirs("./prepared-data/val_rolling_images/neg", exist_ok=True)

    # Load X and y
    X_train, y_train_df, y_train = exe.load_data(X_train_path, y_train_path)
    # Load source data
    df_time_graph = stock.load_ohlc_graph(source_path)
    df_time_graph_red = df_time_graph.loc[X_train.index]
    # Merge all values
    # np.log(df_time_graph['Close'] - df_time_graph['Close'].iloc[0] + 1)

    all_df = X_train.join(df_time_graph).join(y_train_df)
    all_df.reset_index(inplace=True)
    all_df.set_index('Date', inplace=True)

    all_columns = ['id', 'SMA200', 'RSI_14', 'Open', 'High', 'Low', 'Close', 'LongTrend']
    all_df_red = all_df[all_columns]

    # Create rolling values in a df for multivariates
    for df_subset in all_df_red.rolling(250):
        #print(type(df_subset), '\n', df_subset)
        if df_subset.shape[0] >= 250:
            print("Processing id {}".format(df_subset.iloc[-1].id))
            # Save subset as file
            csv_columns = ['SMA200', 'RSI_14', 'LongTrend']

            csv_path = "./prepared-data/val_rolling_csv/roll_" + str(int(df_subset.iloc[-1].id)) + ".csv"
            df_subset[csv_columns].to_csv(csv_path, sep=';',
                             index=True, header=True)

            #Create graph of subset

            if df_subset.iloc[-1].LongTrend==1:
                image_path = "./prepared-data/val_rolling_images/pos/roll_" + str(int(df_subset.iloc[-1].id)) + ".png"
            elif df_subset.iloc[-1].LongTrend==0:
                image_path = "./prepared-data/val_rolling_images/neg/roll_" + str(int(df_subset.iloc[-1].id)) + ".png"

            apd = mpf.make_addplot(df_subset['RSI_14'], panel=1, color='black', ylim=(10, 90), secondary_y=True)
            mpf.plot(df_subset,
                     type='candle', volume=False, mav=(20, 100, 200),
                     figscale=1.5, addplot=apd, panel_ratios=(1, 0.3),
                     savefig=image_path)



    # Save images of ohlc graph

    # Save rolling charts in files

    print("End")


if __name__ == "__main__":
    convert_time_series(args.config_path, args.feature_dir)

    print("=== Program end ===")
