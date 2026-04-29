#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2X Data generation: Generate ground truth for stock markets based on annotations.

This script generates ground truth signals for stock market data by loading them
from an external annotation file. Unlike `step20_generate_groundtruth_stockmarket.py`,
which calculates trends and signals from raw OHLC data, this script uses
pre-existing, manually or externally created annotations as the ground truth.

Inputs:
    - Configuration file (specified by --config_path argument): Contains paths
      for raw data, prepared data, results directories, and the annotation file.
    - Raw stock market OHLC data: Loaded from the path specified in the config file.
    - Annotation file: A CSV file containing the ground truth signals, specified
      by the `outcomes_source` path in the config file.

Outputs:
    - `outcomes_cut.csv`: A CSV file containing the ground truth signals loaded
      from the annotation file.
    - Various PNG plots: Visualizations of the raw data and the loaded ground
      truth signals. These are saved in a 'data_generation' subdirectory within
      the configured results directory.

Main Functions:
    - `generate_features_outcomes`: Loads the ground truth signals from the
      annotation file and merges them with the source data.
    - `main`: Parses arguments, loads configuration, loads raw data and annotations,
      calls `generate_features_outcomes`, and saves the final outcomes and plots.

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

# Built-in/Generic Imports

# Libs
import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Own modules
import utils.data_handling_support_functions as sup
import utils.custom_methods as custom
import utils.data_visualization_functions as vis

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

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(
    description="Step 2.0 - Generate features and outcomes from raw data"
)
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


def generate_features_outcomes(
    outcomes_source, outcome_col, source, rename_outcome_col
):
    """
    Load ground truth signals from an annotation file and merge them with the source data.

    Parameters
    ----------
    outcomes_source : str
        Path to the annotation file.
    outcome_col : str
        Name of the outcome column in the annotation file.
    source : pd.DataFrame
        The source OHLC data.
    rename_outcome_col : str
        The new name for the outcome column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the ground truth signals.
    """

    # Outcome and Feature Construction
    # Generate the class values, i.e.the y for the data.Construct features. The following dataframes are
    # generated:
    # - source
    # - features
    # - outcomes

    # Load outcome file
    outcome_raw = pd.read_csv(outcomes_source, sep=";")
    outcome_raw.index.name = "id"
    outcome_raw.columns = ["Date", outcome_col]
    outcome_raw["Date"] = pd.to_datetime(outcome_raw["Date"])
    outcome_raw["Date"].apply(mdates.date2num)
    outcome_raw.rename(columns={outcome_col: rename_outcome_col}, inplace=True)

    # Merge all y values to the series start
    outcomes = pd.DataFrame(index=source.index).join(outcome_raw[rename_outcome_col])

    return outcomes


def main(config_path):
    """
    Main function to execute the script.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    conf = sup.load_config(config_path)
    # Load annotations file
    y_labels = (
        pd.read_csv(conf["Paths"].get("labels_path"), sep=";", header=None)
        .set_index(0)
        .to_dict()[1]
    )

    # Generating filenames for saving the files
    image_save_directory = os.path.join(
        conf["Paths"].get("results_directory"), "data_generation"
    )
    outcomes_filename_raw = os.path.join(
        conf["Paths"].get("prepared_data_directory"),
        "temp",
        "temp_outcomes_uncut" + ".csv",
    )
    os.makedirs(os.path.dirname(outcomes_filename_raw), exist_ok=True)

    # Load only a subset of the whole raw data to create a debug dataset
    source = custom.load_source(conf["Paths"].get("source_path"))  # .iloc[0:1000, :]
    outcomes_source = conf["Paths"].get("outcomes_source")
    outcome_col = conf["Generation"].get("outcome_col")
    rename_outcome_col = conf["Common"].get("class_name")

    # Plot source
    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(source["Date"], source["Close"])
    plt.title(conf["Paths"].get("source_path"))
    plt.show(block=False)

    # y_labels = annotations #generate_custom_class_labels()
    outcomes = generate_features_outcomes(
        outcomes_source, outcome_col, source, rename_outcome_col
    )

    # Drop the 50 last values as they cannot be used for prediction as +50 days ahead is predicted
    # No drop as the annotations were loaded
    source_cut = source  # source.drop(source.tail(50).index, inplace=False)
    outcomes_cut = outcomes  # outcomes.drop(outcomes.tail(50).index, inplace=False)

    vis.plot_three_class_graph(
        outcomes_cut[rename_outcome_col].values,
        source_cut["Close"],
        source_cut["Date"],
        0,
        0,
        0,
        ("close", "neutral", "positive", "negative"),
        title=conf["Common"].get("dataset_name") + "_Groud_Truth_LongTrend",
        save_fig_prefix=image_save_directory,
    )

    def binarize(outcomes, class_number):
        return (outcomes == class_number).astype(int)

    vis.plot_two_class_graph(
        binarize(outcomes_cut[rename_outcome_col], conf["Common"].getint("class_number")),
        source_cut["Close"],
        source_cut["Date"],
        0,
        ("close", "Positive Trend"),
        title=conf["Common"].get("dataset_name") + "_Groud_Truth_LongTrend",
        save_fig_prefix=image_save_directory,
    )

    # Save file
    # Save outcomes to a csv file
    print("Outcomes shape {}".format(outcomes_cut.shape))
    outcomes_cut.to_csv(outcomes_filename_raw, sep=";", index=True, header=True)
    print("Saved outcomes to " + outcomes_filename_raw)


if __name__ == "__main__":
    # if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    main(args.config_path)

    print("=== Program end ===")
