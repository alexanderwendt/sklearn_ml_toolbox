#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Clean raw data.

This script performs the initial cleaning of the raw feature and outcome data.
It handles missing values, renames columns, and performs a basic analysis of the
data to identify potential issues. The cleaned data is then saved for further
processing in the next steps.

Inputs:
    - Raw features CSV file (specified in the config).
    - Raw outcomes CSV file (specified in the in the config).
    - Raw source data file (specified in the config).
    - Labels file (specified in the config).

Outputs:
    - `step31out.pickle`: A pickle file containing the cleaned features, outcomes,
      class labels, and other relevant data for the next step.
    - Various plots in the results directory, showing the distribution of each
      feature and the missing data matrix.

Main Functions:
    - `clean_features_first_pass`: Performs initial cleaning of the feature
      DataFrame, including renaming columns and handling missing values.
    - `load_files`: Loads all the necessary input files.
    - `analyze_raw_data`: Performs a basic analysis of the raw data, including
      plotting feature distributions and checking for unique columns.
    - `main`: Orchestrates the loading, cleaning, and analysis of the data, and
      saves the cleaned data to a pickle file.

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
import argparse
import os
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from pandas.plotting import register_matplotlib_converters

import utils.data_handling_support_functions as sup
# Own modules
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

register_matplotlib_converters()

parser = argparse.ArgumentParser(description="Step 3 - Clean raw data")
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)
parser.add_argument(
    "-n",
    "--no_images",
    action="store_true",
    default=False,
    help="Set true if the generation of feature images shall be disabled. "
    "It is usually done for inference data.",
)
parser.add_argument(
    "-i",
    "--on_inference_data",
    action="store_true",
    help="Set inference if only inference and no training",
)
parser.add_argument(
    "-nds",
    "--no_source_data",
    action="store_true",
    help="Load no source data if the data does not need to be visualized in time charts.",
)

args = parser.parse_args()


def clean_features_first_pass(features_raw, class_name):
    """
    Perform a first pass of cleaning on the raw feature data.

    This function takes a raw features DataFrame and performs the following cleaning steps:
    1.  Creates a copy of the DataFrame to avoid modifying the original.
    2.  Renames columns by replacing spaces with underscores and slashes with hyphens.
    3.  Strips leading/trailing whitespace from all string columns.
    4.  Replaces specified missing value placeholders (e.g., '?') with `np.nan`.
    5.  Prints information about the DataFrame, including its size, head, missing values, and column types.

    Parameters
    ----------
    features_raw : pd.DataFrame
        The raw features DataFrame to be cleaned.
    class_name : str
        The name of the class column. Although this parameter is passed, it is not
        currently used in the function. It is likely a remnant of a previous
        implementation or intended for future use.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the initial cleaning steps applied.
    """

    features = features_raw.copy()

    # === Define index name ===#
    # Define name if there is no index name

    # df.index.name = 'id'

    # === rename colums ===#
    # df.rename(columns={'model.year':'year'}, inplace=True)

    # Rename columns with " "
    features.columns = [x.replace(" ", "_") for x in features.columns]
    features.columns = [x.replace("/", "-") for x in features.columns]

    print("Features size : ", features.shape)
    print(features.head(5))
    # print("Outcomes size : ", outcomes.shape)
    # display(outcomes.head(5))

    ## Data Cleanup of Features and Outcomes before Features are Modified

    # Strip all string values to find the missing data
    from pandas.api.types import is_string_dtype

    for col in features.columns:
        if is_string_dtype(features[col]):
            print("Strip column {}".format(col))
            features[col] = features[col].str.strip()

    # Replace values for missing data

    # === Replace all missing values with np.nan
    for col in features.columns[0:-1]:
        features[col] = features[col].replace("?", np.nan)
        # df[col] = df[col].replace('unknown', np.nan)

    print("Missing data in the data frame")
    print(sum(features.isna().sum()))

    # Get column types
    print("Column types:")
    print(features.dtypes)
    print("\n")

    print("feature columns: {}\n".format(features.columns))
    # print("Outcome column: {}".format(outcomes[class_name].name))

    return features


def load_files(features_path, outcomes_path, source_path, labels_path, no_source_data=False):
    """
    Load all necessary input files.

    Parameters
    ----------
    features_path : str
        Path to the features file.
    outcomes_path : str
        Path to the outcomes file.
    source_path : str
        Path to the source data file.
    labels_path : str
        Path to the labels file.
    no_source_data : bool, optional
        If True, do not load the source data, by default False.

    Returns
    -------
    tuple
        A tuple containing the loaded DataFrames for features, outcomes, source data, and class labels.
    """
    # Generating filenames for loading the files
    input_features_filename = features_path
    input_outcomes_filename = outcomes_path

    source_filename = source_path

    print("=== Paths ===")
    print("Input Features: ", input_features_filename)
    print("Input Outcomes: ", input_outcomes_filename)
    print("Original source: ", source_filename)

    # === Load Features ===#
    features_raw = pd.read_csv(input_features_filename, sep=";").set_index("id")
    print(features_raw.head(1))

    # === Load Outcomes ===#
    if input_outcomes_filename and os.path.isfile(input_outcomes_filename):
        outcomes_raw = pd.read_csv(input_outcomes_filename, sep=";").set_index("id")
        print(outcomes_raw.head(1))
    else:
        outcomes_raw = None
        print("No outcomes available for inference data")

    # === Load Source ===#
    if source_filename and os.path.isfile(source_filename) and not no_source_data:
        data_source_raw = sup.load_data_source(source_filename)
        print("Loading data source as time graph")
    else:
        data_source_raw = None
        print(
            "No raw data source found or no source data should be loaded as it would be for temporal processing."
        )

    # === Load class labels or modify ===#
    if labels_path and os.path.isfile(labels_path):
        class_labels = load_class_labels(labels_path)
        print("Class labels found")
    else:
        class_labels = None
        print("No class labels found")

    return features_raw, outcomes_raw, data_source_raw, class_labels


def load_class_labels(labels_filename):
    """
    Load class labels from a file.

    Parameters
    ----------
    labels_filename : str
        Path to the labels file.

    Returns
    -------
    dict
        A dictionary mapping class labels to integer values.
    """

    df_y_classes = pd.read_csv(labels_filename, delimiter=";", header=None)
    class_labels = sup.inverse_dict(
        df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1]
    )
    print("Loaded  classes from file", class_labels)
    print(class_labels)

    return class_labels


def print_characteristics(
    features_raw, image_save_directory, dataset_name, save_graphs=False
):
    """
    Print and plot the characteristics of each feature.

    Parameters
    ----------
    features_raw : pd.DataFrame
        The DataFrame containing the features.
    image_save_directory : str
        The directory where the plots will be saved.
    dataset_name : str
        The name of the dataset.
    save_graphs : bool, optional
        If True, save the generated plots, by default False.
    """
    for i, d in enumerate(features_raw.dtypes):
        if is_string_dtype(d):
            print("Column {} is a categorical string".format(features_raw.columns[i]))
            s = (
                features_raw[features_raw.columns[i]].value_counts()
                / features_raw.shape[0]
            )
            fig = vis.paintBarChartForCategorical(s.index, s)
        else:
            print("Column {} is a numerical value".format(features_raw.columns[i]))
            fig = vis.paintHistogram(features_raw, features_raw.columns[i])

        plt.figure(fig.number)

        vis.save_figure(
            plt.gcf(),
            image_save_directory=image_save_directory,
            filename="feature_{}-{}".format(i, features_raw.columns[i]),
        )


def analyze_raw_data(
    features,
    outcomes,
    result_directory,
    dataset_name,
    class_name,
    no_images=False,
    on_inference_data=False,
):
    """
    Perform a basic analysis of the raw data.

    Parameters
    ----------
    features : pd.DataFrame
        The features DataFrame.
    outcomes : pd.DataFrame
        The outcomes DataFrame.
    result_directory : str
        The directory where the results will be saved.
    dataset_name : str
        The name of the dataset.
    class_name : str
        The name of the class column.
    no_images : bool, optional
        If True, do not generate images, by default False.
    on_inference_data : bool, optional
        If True, perform analysis on inference data, by default False.
    """
    print("Results target: {}".format(result_directory))

    numSamples = features.shape[0]
    print("Number of samples={}".format(numSamples))

    numFeatures = features.shape[1]
    print("Number of features={}".format(numFeatures))

    save_graphs = True

    if not outcomes is None:
        numClasses = outcomes[class_name].value_counts().shape[0]
        print("Number of classes={}".format(numClasses))

        if not unique_cols(outcomes):
            raise Exception(
                "Data processing error. At least one column has all the same values."
            )

        print_characteristics(
            outcomes, result_directory, dataset_name, save_graphs=save_graphs
        )
    else:
        numClasses = -1

    if not no_images:
        print_characteristics(
            features, result_directory, dataset_name, save_graphs=save_graphs
        )

    if (not on_inference_data) and (not unique_cols(features)):
        raise Exception(
            "Data processing error. At least one column has all the same values."
        )


def unique_cols(df):
    """
    Check if all values of a column of a dataframe are the same.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked.

    Returns
    -------
    bool
        True if all columns are unique, False otherwise.
    """
    a = df.to_numpy()
    return sum((a[0] == a).all(0)) == 0


def main(config_path, on_inference_data, no_images, no_source_data):
    """
    Main function to execute the script.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    on_inference_data : bool
        If True, perform analysis on inference data.
    no_images : bool
        If True, do not generate images.
    no_source_data : bool
        If True, do not load the source data.
    """
    conf = sup.load_config(config_path)

    data_directory = conf["Paths"].get("prepared_data_directory")
    result_directory = os.path.join(
        conf["Paths"].get("results_directory"), "data_preparation"
    )

    data_preparation_dump_file_path = os.path.join(
        conf["Paths"].get("prepared_data_directory"), "temp", "step31out.pickle"
    )
    os.makedirs(os.path.dirname(data_preparation_dump_file_path), exist_ok=True)

    features_path = os.path.join(conf["Preparation"].get("features_in"))
    if "outcomes_in" in conf["Preparation"]:
        outcomes_path = os.path.join(conf["Preparation"].get("outcomes_in"))
    else:
        outcomes_path = None
        print("No outcomes in, do inference")
    labels_path = conf["Paths"].get("labels_path")
    source_path = os.path.join(conf["Preparation"].get("source_in"))

    features_raw, outcomes_cleaned1, data_source_raw, class_labels = load_files(
        features_path, outcomes_path, source_path, labels_path, no_source_data
    )

    features_cleaned1 = clean_features_first_pass(features_raw, class_labels)

    analyze_raw_data(
        features_cleaned1,
        outcomes_cleaned1,
        result_directory,
        conf["Common"].get("dataset_name"),
        conf["Common"].get("class_name"),
        no_images,
        on_inference_data,
    )

    dump(
        (
            features_cleaned1,
            outcomes_cleaned1,
            class_labels,
            data_source_raw,
            data_directory,
            result_directory,
        ),
        open(data_preparation_dump_file_path, "wb"),
    )
    print("Stored paths to: ", data_preparation_dump_file_path)


if __name__ == "__main__":
    main(args.config_path, args.on_inference_data, args.no_images, args.no_source_data)

    print("=== Program end ===")
