#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Clean raw data
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

# Libs
import argparse
import os
from pickle import dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display
#from matplotlib.ticker import FuncFormatter, MaxNLocator
from pandas.core.dtypes.common import is_string_dtype
from pandas.plotting import register_matplotlib_converters

# Own modules
import data_visualization_functions as vis
import data_handling_support_functions as sup

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

register_matplotlib_converters()

parser = argparse.ArgumentParser(description='Step 3.1 - Clean raw data')
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
#parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


def clean_features_first_pass(features_raw, class_name):
    '''


    '''

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
    display(features.head(5))
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
        features[col] = features[col].replace('?', np.nan)
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


def load_files(data_directory, dataset_name, annotations_filename):
    # Constants for all notebooks in the Machine Learning Toolbox
    print("Data source: {}".format(data_directory))

    # Generating filenames for loading the files
    input_features_filename = data_directory + "/" + dataset_name + "_features" + ".csv"
    input_outcomes_filename = data_directory + "/" + dataset_name + "_outcomes" + ".csv"

    source_filename = data_directory + "/" + dataset_name + "_source" + ".csv"
    #labels_filename = data_directory + "/" + dataset_name + "_labels" + ".csv"
    # Columns for feature selection
    # selected_feature_columns_filename = data_directory + "/" + dataset_name + "_" + class_name + "_selected_feature_columns.csv"

    print("=== Paths ===")
    print("Input Features: ", input_features_filename)
    # print("Output Features: ", model_features_filename)
    print("Input Outcomes: ", input_outcomes_filename)
    # print("Output Outcomes: ", model_outcomes_filename)
    #print("Labels: ", labels_filename)
    print("Original source: ", source_filename)
    # print("Labels for the model: ", model_labels_filename)
    # print("Selected feature columns: ", selected_feature_columns_filename)

    ### Load Features and Outcomes

    # === Load Features ===#
    features_raw = pd.read_csv(input_features_filename, sep=';').set_index('id')  # Set ID to be the data id
    print(features_raw.head(1))

    # === Load Outcomes ===#
    if os.path.isfile(input_outcomes_filename):
        #if not on_inference_data:
        outcomes_raw = pd.read_csv(input_outcomes_filename, sep=';').set_index('id')  # Set ID to be the data id
        print(outcomes_raw.head(1))
    else:
        outcomes_raw =None
        print("No outcomes available for inference data")

    # === Load Source ===#
    # Load original data for visualization
    data_source_raw = sup.load_data_source(source_filename)

    # === Load class labels or modify ===#
    #Load annotations
    #annotations = pd.read_csv(annotations_filename, sep=';', header=None).set_index(0).to_dict()[1]
    class_labels = load_class_labels(annotations_filename)

    return features_raw, outcomes_raw, data_source_raw, class_labels

def load_class_labels(labels_filename):
    '''


    '''

    # === Load Class Labels ===#
    # Get classes into a dict from outcomes
    # class_labels = dict(zip(outcomes_raw[class_name].unique(), list(range(1,len(outcomes_raw[class_name].unique())+1, 1))))
    # print(class_labels)
    # Load class labels file
    df_y_classes = pd.read_csv(labels_filename, delimiter=';', header=None)
    class_labels = sup.inverse_dict(df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1])
    print("Loaded  classes from file", class_labels)
    # === Define classes manually ===#
    # class_labels = {
    #    0 : 'class1',
    #    1 : 'class2'
    # }
    print(class_labels)

    return class_labels


def print_characteristics(features_raw, image_save_directory, dataset_name, save_graphs=False):
    for i, d in enumerate(features_raw.dtypes):
        if is_string_dtype(d):
            print("Column {} is a categorical string".format(features_raw.columns[i]))
            s = features_raw[features_raw.columns[i]].value_counts() / features_raw.shape[0]
            fig = vis.paintBarChartForCategorical(s.index, s)
        else:
            print("Column {} is a numerical value".format(features_raw.columns[i]))
            fig = vis.paintHistogram(features_raw, features_raw.columns[i])

        plt.figure(fig.number)
        if save_graphs == True:
            plt.savefig(
                image_save_directory + '/' + dataset_name + '_{}-{}'.format(i, features_raw.columns[i]),
                dpi=300)
        plt.show(block = False)


def analyze_raw_data(features, outcomes, result_directory, dataset_name, class_name):
    # Define file names
    print("Results target: {}".format(result_directory))

    ## Analyse the Features Individually

    # Print graphs for all features

    # Get number of samples
    numSamples = features.shape[0]
    print("Number of samples={}".format(numSamples))

    # Get number of features
    numFeatures = features.shape[1]
    print("Number of features={}".format(numFeatures))

    save_graphs = True  # If set true, then all images are saved into the image save directory.

    # Get the number of classes for the supervised learning
    if not outcomes is None:
        numClasses = outcomes[class_name].value_counts().shape[0]
        print("Number of classes={}".format(numClasses))

        if not unique_cols(outcomes):
            raise Exception("Data processing error. At least one column has all the same values.")

        # Print graphs for all features
        print_characteristics(outcomes, result_directory, dataset_name, save_graphs=save_graphs)
    else:
        numClasses = -1

    # Print graphs for all features
    print_characteristics(features, result_directory, dataset_name, save_graphs=save_graphs)

    # Check if raw data has useless values, i.e. all values of one column are the same
    if not unique_cols(features):
        raise Exception("Data processing error. At least one column has all the same values.")


def unique_cols(df):
    '''
    Check if all values of a column of a dataframe are the same. If yes, then columns are unique. If False,
    the columns have to be processed.

    '''
    a = df.to_numpy() # df.values (pandas<0.24)
    return sum((a[0] == a).all(0))==0

def main(config_path):
    conf = sup.load_config(config_path)

    #if not on_inference_data:
    data_directory = conf['Paths'].get('prepared_data_directory')
    result_directory = os.path.join(conf['Paths'].get('result_directory'), "data_preparation")
    annotations_filename = conf["Paths"].get("annotations_file")
    #else:
    #    data_directory = conf['Paths'].get('inference_data_directory')
    #    result_directory = os.path.join(conf['Paths'].get('result_directory', "data_preparation_inference"))

    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)
        print("Created directory: ", result_directory)

    data_preparation_dump_file_path = os.path.join("tmp", "step31out.pickle")
    if not os.path.isdir("tmp"):
        os.makedirs("tmp")
        print("Created directory: ", "tmp")

    features_raw, outcomes_cleaned1, data_source_raw, class_labels = load_files(data_directory, conf['Common'].get('dataset_name'),
                                                                                annotations_filename)
    ## Data Cleanup of Features and Outcomes before Features are Modified
    features_cleaned1 = clean_features_first_pass(features_raw, class_labels)

    analyze_raw_data(features_cleaned1, outcomes_cleaned1, result_directory, conf['Common'].get('dataset_name'), conf['Common'].get('class_name'))

    # Save structures for further processing
    # Dump path data
    dump((features_cleaned1, outcomes_cleaned1, class_labels, data_source_raw, data_directory, result_directory),
         open(data_preparation_dump_file_path, 'wb'))
    print("Stored paths to: ", data_preparation_dump_file_path)


if __name__ == "__main__":

    main(args.config_path)

    print("=== Program end ===")
