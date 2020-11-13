import argparse
import json
import os
from pickle import dump

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
import scipy.stats
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as m
from IPython.core.display import display
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pandas.core.dtypes.common import is_string_dtype

import data_visualization_functions as vis
import data_handling_support_functions as sup

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)

#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

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
    #print("Outcomes size : ", outcomes.shape)
    #display(outcomes.head(5))

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
    #print("Outcome column: {}".format(outcomes[class_name].name))

    return features

def load_files(data_directory, dataset_name, class_name):
    # Constants for all notebooks in the Machine Learning Toolbox
    print("Data source: {}".format(data_directory))

    # Generating filenames for loading the files
    input_features_filename = data_directory + "/" + dataset_name + "_features" + ".csv"
    input_outcomes_filename = data_directory + "/" + dataset_name + "_outcomes" + ".csv"

    source_filename = data_directory + "/" + dataset_name + "_source" + ".csv"
    labels_filename = data_directory + "/" + dataset_name + "_labels" + ".csv"
    # Columns for feature selection
    #selected_feature_columns_filename = data_directory + "/" + dataset_name + "_" + class_name + "_selected_feature_columns.csv"

    print("=== Paths ===")
    print("Input Features: ", input_features_filename)
    #print("Output Features: ", model_features_filename)
    print("Input Outcomes: ", input_outcomes_filename)
    #print("Output Outcomes: ", model_outcomes_filename)
    print("Labels: ", labels_filename)
    print("Original source: ", source_filename)
    #print("Labels for the model: ", model_labels_filename)
    #print("Selected feature columns: ", selected_feature_columns_filename)

    ### Load Features and Outcomes

    # === Load Features ===#
    features_raw = pd.read_csv(input_features_filename, sep=';').set_index('id')  # Set ID to be the data id
    display(features_raw.head(1))

    # === Load Outcomes ===#
    outcomes_raw = pd.read_csv(input_outcomes_filename, sep=';').set_index('id')  # Set ID to be the data id
    display(outcomes_raw.head(1))

    # === Load Source ===#
    # Load original data for visualization
    data_source_raw = load_data_source(source_filename)

    # === Load class labels or modify ===#
    class_labels = load_class_labels(labels_filename)

    return features_raw, outcomes_raw, data_source_raw, class_labels


def load_data_source(source_filename):
    '''


    '''
    source = pd.read_csv(source_filename, sep=';').set_index('id')  # Set ID to be the data id
    display(source.head(1))

    source = pd.read_csv(source_filename, delimiter=';').set_index('id')
    source['Date'] = pd.to_datetime(source['Date'])
    source['Date'].apply(mdates.date2num)
    print("Loaded source time graph={}".format(source.columns))
    print("X. Shape={}".format(source.shape))
    display(source.head())

    return source


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

    return  class_labels

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
        plt.show()

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

    # Get the number of classes for the supervised learning
    numClasses = outcomes[class_name].value_counts().shape[0]
    print("Number of classes={}".format(numClasses))

    save_graphs = True  # If set true, then all images are saved into the image save directory.

    # Print graphs for all features
    print_characteristics(features, result_directory, dataset_name, save_graphs=save_graphs)

    save_graphs = True  # If set true, then all images are saved into the image save directory.

    # Print graphs for all features
    print_characteristics(outcomes, result_directory, dataset_name, save_graphs=save_graphs)

def main():
    conf = sup.load_config(args.config_path)

    annotations_directory = conf['annotations_directory']

    if not args.on_inference_data:
        data_directory = conf['training_data_directory']
        result_directory = conf['result_directory'] + "/analysis_training"
    else:
        data_directory = conf['inference_data_directory']
        result_directory = conf['result_directory'] + "/analysis_inference"

    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)
        print("Created directory: ", result_directory)

    data_preparation_dump_file_path = os.path.join("tmp", "step31out.pickle")
    if not os.path.isdir("tmp"):
        os.makedirs("tmp")
        print("Created directory: ", "tmp")

    features_raw, outcomes_cleaned1, data_source_raw, class_labels = load_files(data_directory, conf['dataset_name'], conf['class_name'])
    ## Data Cleanup of Features and Outcomes before Features are Modified
    features_cleaned1 = clean_features_first_pass(features_raw, class_labels)

    analyze_raw_data(features_cleaned1, outcomes_cleaned1, result_directory, conf['dataset_name'], conf['class_name'])

    #Save structures for further processing
    # Dump path data
    dump((features_cleaned1, outcomes_cleaned1, class_labels, data_source_raw, data_directory, result_directory), open(data_preparation_dump_file_path, 'wb'))
    print("Stored paths to: ", data_preparation_dump_file_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 3.1 - Analyze Data')
    #parser.add_argument("-r", '--retrain_all_data', action='store_true',
    #                    help='Set flag if retraining with all available data shall be performed after ev')
    parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.json",
                        help='Configuration file path', required=False)
    parser.add_argument("-i", "--on_inference_data", action='store_true',
                        help="Set inference if only inference and no training")

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    main()


    print("=== Program end ===")