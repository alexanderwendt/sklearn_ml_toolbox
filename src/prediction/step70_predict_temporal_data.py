#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 7: Predict on temporal data.

This script uses the trained model to make predictions on new, unseen temporal
data. It visualizes the predictions against the source data to provide a
qualitative assessment of the model's performance on the inference data.

Inputs:
    - Inference data.
    - The trained model.
    - The source data for the inference period.

Outputs:
    - Plots showing the model's predictions on the inference data, saved in the
      `evaluation` subdirectory of the results directory.

Main Functions:
    - `visualize_temporal_data`: Loads the data and the model, generates the
      predictions, and creates the temporal prediction plots.

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
import pandas as pd
import matplotlib.dates as mdates
import utils.sklearn_utils as model_util
from pandas.plotting import register_matplotlib_converters
import numpy as np

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup
import utils.evaluation_utils as eval
from filepaths import Paths

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

parser = argparse.ArgumentParser(description="Step 7 - Predict Temporal Data")
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)
parser.add_argument(
    "-sec",
    "--config_section",
    default="EvaluationValidation",
    help="Configuration section in config file",
    required=False,
)

args = parser.parse_args()
print(args)


def visualize_temporal_data(config_path, config_section):
    """
    Visualize the model's predictions on temporal data.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    config_section : str
        The section in the configuration file to use.
    """
    config = sup.load_config(config_path)
    print("Load paths")
    paths = Paths(config).paths
    title = config.get(config_section, "title")

    X_val, y_val, labels, model, external_params = eval.load_evaluation_data(
        config, config_section
    )

    y_classes = labels

    model_name = config["Common"].get("dataset_name")
    source_path = config[config_section].get("source_in")
    result_directory = paths["results_directory"]

    figure_path_prefix = result_directory + "/evaluation"
    os.makedirs(result_directory + "/evaluation", exist_ok=True)

    pr_threshold = external_params["pr_threshold"]
    print("Loaded precision/recall threshold: {0:.2f}".format(pr_threshold))

    y_test_pred_scores = model.predict_proba(X_val.values)[:, 1]
    y_test_pred = model.predict(X_val.values)
    y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)

    df_time_graph = pd.read_csv(source_path, delimiter=";").set_index("id")
    df_time_graph["Date"] = pd.to_datetime(df_time_graph["Date"])
    df_time_graph["Date"].apply(mdates.date2num)
    print("Loaded feature names for time graph={}".format(df_time_graph.columns))
    print("X. Shape={}".format(df_time_graph.shape))

    y_order_test_pred = pd.DataFrame(
        index=X_val.index,
        data=pd.Series(data=y_test_pred, index=X_val.index, name="y"),
    ).sort_index()

    y_order_test_pred_adjust = pd.DataFrame(
        index=X_val.index,
        data=pd.Series(data=y_test_pred_adjust, index=X_val.index, name="y"),
    ).sort_index()

    print("Plot for inference data to ", figure_path_prefix)
    vis.plot_three_class_graph(
        y_order_test_pred["y"].values,
        df_time_graph["Close"][y_order_test_pred.index],
        df_time_graph["Date"][y_order_test_pred.index],
        0,
        0,
        0,
        ("close", "neutral", "positive", "negative"),
        title=title + "_Inference_" + model_name,
        save_fig_prefix=figure_path_prefix,
    )

    vis.plot_three_class_graph(
        y_order_test_pred_adjust["y"].values,
        df_time_graph["Close"][y_order_test_pred.index],
        df_time_graph["Date"][y_order_test_pred.index],
        0,
        0,
        0,
        ("close", "neutral", "positive", "negative"),
        title=title + "_Inference_Adjusted" + model_name,
        save_fig_prefix=figure_path_prefix,
    )
    if np.unique(y_order_test_pred["y"].values) == 2:
        vis.plot_two_class_graph(
            y_order_test_pred["y"].values,
            df_time_graph["Close"][y_order_test_pred.index],
            df_time_graph["Date"][y_order_test_pred.index],
            0,
            ("close", "Positive Trend"),
            title=title + "_Inference_2_Class" + model_name,
            save_fig_prefix=figure_path_prefix,
        )
        vis.plot_two_class_graph(
            y_order_test_pred_adjust["y"].values,
            df_time_graph["Close"][y_order_test_pred.index],
            df_time_graph["Date"][y_order_test_pred.index],
            0,
            ("close", "Positive Trend"),
            title=title + "_Inference_Adjusted_2_Class" + model_name,
            save_fig_prefix=figure_path_prefix,
        )
    else:
        print("Data is not binarized.")


if __name__ == "__main__":
    visualize_temporal_data(args.config_path, args.config_section)

    print("=== Program end ===")
