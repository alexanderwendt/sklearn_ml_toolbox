#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2X Data Generation: Adapt dimensions of generated features and outcomes.

This script takes the generated features and outcomes and adapts their dimensions
to ensure they are aligned and ready for the next steps in the machine learning
pipeline. This includes removing unusable samples (e.g., from the beginning
of a moving average calculation), cleaning NaN values, and ensuring that the
features, outcomes, and source data all have the same length and index.

Inputs:
    - `temp_features_uncut.csv`: CSV file with the generated features.
    - `temp_outcomes_uncut.csv`: CSV file with the generated outcomes.
    - Raw source data file (specified in the config).

Outputs:
    - `features_out.csv`: CSV file with the dimensionally-adapted features.
    - `outcomes_out.csv`: CSV file with the dimensionally-adapted outcomes.
    - `source_out.csv`: CSV file with the dimensionally-adapted source data.

Main Functions:
    - `cut_unusable_parts_of_dataframe`: Removes a specified number of rows from
      the beginning or end of a DataFrame.
    - `clean_nan`: Removes all rows containing NaN values from a DataFrame.
    - `main`: Loads the data, calls the cleaning and cutting functions, and saves
      the adapted DataFrames to new CSV files.

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

# Libs
import os
import pandas as pd
import numpy as np
import argparse

# Own modules
import utils.data_handling_support_functions as sup
import utils.custom_methods as custom

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


def cut_unusable_parts_of_dataframe(df, head_index=-1, tail_index=-1):
    """
    Remove samples that are not useful, e.g. the first samples from a moving average.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cut.
    head_index : int, optional
        The number of rows to remove from the beginning, by default -1.
    tail_index : int, optional
        The number of rows to remove from the end, by default -1.

    Returns
    -------
    pd.DataFrame
        The cut DataFrame.
    """

    if tail_index > 0:
        dfr = df.drop(df.tail(tail_index).index, inplace=False)
    if head_index > 0:
        dfr = df.drop(df.head(head_index).index, inplace=False)
    # Drop from the timerows too
    # source_cut = source.drop(source.tail(50).index, inplace=False)

    return dfr


def cut_dataframe_subset(df, start_loc, stop_loc):
    """
    Remove all but a specified subset of the dataset, e.g. for debug training.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cut.
    start_loc : int
        The starting location of the subset.
    stop_loc : int
        The stopping location of the subset.

    Returns
    -------
    pd.DataFrame
        The subset of the DataFrame.
    """
    return df.iloc[start_loc:stop_loc, :]


def clean_nan(df):
    """
    Remove all NaN values from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """

    nan_index = pd.isnull(df).any(1).to_numpy().nonzero()[0]
    print("Found {} rows with NaN".format(len(nan_index)))
    df_nonan = df.drop(nan_index)

    print("Got df shape={} from original shape={}".format(df_nonan.shape, df.shape))
    print(df_nonan.head(5))
    print(df_nonan.tail(5))

    return df_nonan


def main(config_path):
    """
    Main function to execute the script.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    conf = sup.load_config(config_path)

    prepared_data_directory = conf["Paths"].get("prepared_data_directory")
    outcomes_filename_uncut = os.path.join(
        prepared_data_directory, "temp", "temp_outcomes_uncut" + ".csv"
    )
    features_filename_uncut = os.path.join(
        prepared_data_directory, "temp", "temp_features_uncut" + ".csv"
    )

    source_uncut = custom.load_source(conf["Paths"].get("source_path"))
    features_uncut = pd.read_csv(features_filename_uncut, sep=";").set_index("id")
    if os.path.isfile(outcomes_filename_uncut):
        outcomes_uncut = pd.read_csv(outcomes_filename_uncut, sep=";").set_index("id")
        print("Outcomes file found. Adapting dimensions for training data.")
        print("Outcomes shape: ", outcomes_uncut.shape)
    else:
        outcomes_uncut = None
        print("Outcomes file not found. Adapting dimensions for inference data.")

    print("Source shape: ", source_uncut.shape)
    print("Features shape: ", features_uncut.shape)

    features_reduced1 = clean_nan(features_uncut)

    if not outcomes_uncut is None:
        intersection_index = outcomes_uncut.index.intersection(features_reduced1.index)

        outcomes = outcomes_uncut.loc[intersection_index]
        print("Cut outcomes shape: ", outcomes.shape)
    else:
        outcomes = None
        intersection_index = features_reduced1.index
        print("Nothing will be cut. Size of features will be used.")

    features = features_reduced1.loc[intersection_index]
    source = source_uncut.loc[intersection_index]

    print("Cut source shape: ", source.shape)
    print("Cut features shape: ", features.shape)

    subset_start = 0
    subset_stop = features.shape[0]

    features_subset = cut_dataframe_subset(features, subset_start, subset_stop)
    source_subset = cut_dataframe_subset(source, subset_start, subset_stop)

    print("Subset source shape: ", source_subset.shape)
    print("Subset features shape: ", features_subset.shape)

    if "outcomes_out" in conf["Generation"]:
        outcomes_out_filename = os.path.join(conf["Generation"].get("outcomes_out"))
    else:
        outcomes_out_filename = None
        outcomes = None
        print("Only preparing features for inference. No outcomes file used.")

    features_out_filename = os.path.join(conf["Generation"].get("features_out"))

    source_out_filename = os.path.join(conf["Generation"].get("source_out"))

    print("=== Paths ===")
    print("Features in: ", features_out_filename)
    print("Outcomes in: ", outcomes_out_filename)
    print("Source out: ", source_out_filename)

    print("Feature shape {}".format(features_subset.shape))
    features_subset.to_csv(features_out_filename, sep=";", index=True, header=True)
    print("Saved features graph to " + features_out_filename)

    print("source shape {}".format(source_subset.shape))
    source_subset.to_csv(source_out_filename, sep=";", index=True, header=True)
    print("Saved source graph to " + source_out_filename)

    if not outcomes is None:
        outcomes_subset = cut_dataframe_subset(outcomes, subset_start, subset_stop)
        print("Subset outcomes shape: ", outcomes_subset.shape)

        print("Outcomes shape {}".format(outcomes_subset.shape))
        outcomes_subset.to_csv(outcomes_out_filename, sep=";", index=True, header=True)
        print("Saved source graph to " + outcomes_out_filename)


if __name__ == "__main__":
    main(args.config_path)

    print("=== Program end ===")
