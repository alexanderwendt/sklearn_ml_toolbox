#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Data analysis for temporal data.

This script performs a temporal analysis of the data, focusing on autocorrelation
and partial autocorrelation. It helps in understanding the time-dependent
structures within the data, which is crucial for time series forecasting.

Inputs:
    - Features and outcomes from the previous steps (loaded via `sup.load_features`).
    - Source data file (specified in the config).

Outputs:
    - Autocorrelation and partial autocorrelation plots for the source data and
      selected features, saved in the `data_preparation` subdirectory of the
      results directory.

Main Functions:
    - `analyze_timegraph`: Performs the temporal analysis, generating the
      autocorrelation and partial autocorrelation plots.
    - `main`: Loads the data and calls `analyze_timegraph`.

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
import os

# Libs
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib as m
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup

__author__ = "Alexander Wendt"
__copyright__ = (
    "Copyright 2020, Christian Doppler Laboratory for " "Embedded Machine Learning"
)
__credits__ = [""]
__license__ = "ISC"
__version__ = "0.2.0"
__maintainer__ = "Alexander Wendt"
__email__ = "alexander.wendt@tuwien.ac.at"
__status__ = "Experiental"

register_matplotlib_converters()

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description="Step 3 - Analyze Data Temporal Analysis")
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)

args = parser.parse_args()


def rescale(conf, features, y):
    """
    Rescale features and outcomes using StandardScaler.

    Parameters
    ----------
    conf : configparser.ConfigParser
        The configuration object.
    features : pd.DataFrame
        The features DataFrame.
    y : np.ndarray
        The outcomes array.

    Returns
    -------
    tuple
        A tuple containing the scaled features and outcomes.
    """

    scaler = StandardScaler()
    scaler.fit(features)
    X_scaled = pd.DataFrame(
        data=scaler.transform(features), index=features.index, columns=features.columns
    )
    print("Unscaled values")
    print(features.iloc[0:2, :])
    print("Scaled values")
    print(X_scaled.iloc[0:2, :])
    scaler.fit(y.reshape(-1, 1))
    y_scaled = pd.DataFrame(
        data=scaler.transform(y.reshape(-1, 1)),
        index=features.index,
        columns=[conf["Common"].get("class_name")],
    )
    print("Unscaled values")
    print(y[0:10])
    print("Scaled values")
    print(y_scaled.iloc[0:10, :])

    return X_scaled, y_scaled


def analyze_timegraph(source, features, y, conf, image_save_directory):
    """
    Perform temporal analysis of the data.

    Parameters
    ----------
    source : pd.DataFrame
        The source data.
    features : pd.DataFrame
        The features DataFrame.
    y : np.ndarray
        The outcomes array.
    conf : configparser.ConfigParser
        The configuration object.
    image_save_directory : str
        The directory where the plots will be saved.
    """

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox

    m.rc_file_defaults()

    print(
        "Plot the total autocorrelation of the price.The dark blue values are the correlation of the price with "
        "the lag. The light blue cone is the confidence interval. If the correlation is > cone, the value is "
        "significant."
    )

    vis.plot_autocorrelation(
        np.log(source["Close"]),
        "OMXS30",
        mode="acf",
        lags=None,
        xlim=None,
        ylim=None,
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        np.log(source["Close"]),
        "OMXS30_700_first",
        mode="acf",
        lags=None,
        xlim=[0, 700],
        ylim=None,
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        np.log(source["Close"]),
        "OMXS30",
        mode="pacf",
        lags=200,
        xlim=None,
        ylim=None,
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        np.log(source["Close"]),
        "OMXS30_first_10",
        mode="pacf",
        lags=50,
        xlim=[0, 10],
        ylim=None,
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        features.MA200Norm,
        "OMXS30_MA200",
        mode="acf",
        lags=None,
        xlim=None,
        ylim=None,
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        features.MA200Norm,
        "OMXS30_MA200_first_200",
        mode="acf",
        lags=None,
        xlim=[0, 200],
        ylim=None,
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        features.MA200Norm,
        "OMXS30_MA200",
        mode="pacf",
        lags=200,
        xlim=None,
        ylim=None,
        image_save_directory=image_save_directory,
    )

    diff = pd.DataFrame(
        data=np.divide(source["Close"] - source["Close"].shift(1), source["Close"])
    ).set_index(source["Date"])
    diff = diff.iloc[1:, :]
    fig = plt.figure(figsize=(15, 4))
    plt.plot(source["Date"].iloc[1:], diff)
    plt.grid()

    print(
        "Plot the total autocorrelation of the price. The dark blue values are the correlation of the price with the lag. "
        + "The light blue cone is the confidence interval. If the correlation is > cone, the value is significant."
    )

    vis.plot_autocorrelation(
        diff,
        "OMXS30_difference",
        mode="acf",
        lags=None,
        xlim=[0, 50],
        ylim=[-0.2, 0.2],
        image_save_directory=image_save_directory,
    )

    vis.plot_autocorrelation(
        diff,
        "OMXS30_difference",
        mode="pacf",
        lags=100,
        xlim=[0, 50],
        ylim=[-0.2, 0.2],
        image_save_directory=image_save_directory,
    )

    X_scaled, y_scaled = rescale(conf, features, y)
    vis.plot_temporal_correlation_feature(
        X_scaled,
        conf["Common"].get("dataset_name"),
        image_save_directory,
        source,
        y_scaled,
    )


def main(config_path):
    """
    Main function to execute the script.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    conf = sup.load_config(config_path)
    features, y, df_y, class_labels = sup.load_features(conf)

    source_filename = os.path.join(conf["Preparation"].get("source_in"))
    source = sup.load_data_source(source_filename)

    image_save_directory = conf["Paths"].get("results_directory") + "/data_preparation"

    analyze_timegraph(source, features, y, conf, image_save_directory)


if __name__ == "__main__":
    main(args.config_path)

    print("=== Program end ===")
