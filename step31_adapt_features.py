#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Adapt features for Machine Learning
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
import pickle
import missingno as msno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter, MaxNLocator
#from pandas.core.dtypes.common import is_string_dtype
from pandas.plotting import register_matplotlib_converters

# Own modules
import utils.data_handling_support_functions as sup
from utils import data_visualization_functions as vis

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

parser = argparse.ArgumentParser(description='Step 3 - Adapt features')
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


def adapt_features_for_model(features_cleaned1, outcomes_cleaned1, result_dir, class_labels, conf):


    ## Prepare the Feature Columns

    # === Replace signs for missing values or other values with ===#
    features = features_cleaned1.copy()

    # Custom replacements, replace only if there is something to replace, else it makes NAN of it
    # value_replacements = {
    #    'n': 0,
    #    'y': 1,
    #    'unknown': np.NAN
    # }

    # === Replace all custom values and missing values with content from the value_replacement
    for col in features.columns:
        # df_dig[col] = df[col].map(value_replacements)
        # df_dig[col] = df[col].replace('?', np.nan)

        # Everything to numeric
        features[col] = pd.to_numeric(features[col])
        # df_dig[col] = np.int64(df_dig[col])

    print(features.head(5))

    # Create one-hot-encoding for certain classes and replace the original class
    # onehotlabels = pd.get_dummies(df_dig.iloc[:,1])

    # Add one-hot-encondig columns to the dataset
    # for i, name in enumerate(onehotlabels.columns):
    #    df_dig.insert(i+1, column='Cylinder' + str(name), value=onehotlabels.loc[:,name])

    # Remove the original columns
    # df_dig.drop(columns=['cylinders'], inplace=True)

    ## Prepare the Outcomes if they exist
    if outcomes_cleaned1 is not None:
        # Replace classes with digital values
        outcomes = outcomes_cleaned1.copy()
        outcomes = outcomes.astype(int)
        print("Outcome types")
        print(outcomes.dtypes)

        ### Binarize Multiclass Dataset
        # If the binarize setting is used, then binarize the class of the outcome.
        if conf['Common'].getboolean('binarize_labels') == True:
            binarized_outcome = (outcomes[conf['Common'].get('class_name')] == conf['Common'].getint('class_number')).astype(int)
            y = binarized_outcome.values.flatten()
            print("y was binarized. Classes before: {}. Classes after: {}".format(np.unique(outcomes[conf['Common'].get('class_name')]),
                                                                                  np.unique(y)))

            # Redefine class labels
            class_labels = {
                0: conf['Common'].get('binary_0_label'),
                1: conf['Common'].get('binary_1_label')
            }

            print("Class labels redefined to: {}".format(class_labels))
            print("y labels: {}".format(class_labels))
        else:
            y = outcomes[conf['Common'].get('class_name')].values.flatten()
            print("No binarization was made. Classes: {}".format(np.unique(y)))


        print("y shape: {}".format(y.shape))
        print("y unique classes: {}".format(np.unique(y, axis=0)))
    else:
        y = None
        class_labels = None

    ## Determine Missing Data
    #Missing data is only visualized here as it is handled in the training algorithm in S40.

    # Check if there are any nulls in the data
    print("Missing data in the features: ", features.isnull().values.sum())
    features[features.isna().any(axis=1)]

    # Missing data part
    print("Number of missing values per feature")
    missingValueShare = []
    for col in features.columns:
        # if is_string_dtype(df_dig[col]):
        missingValueShare.append(sum(features[col].isna()) / features.shape[0])

    # Print missing value graph
    vis.paintBarChartForMissingValues(features.columns, missingValueShare)
    barplot = plt.gcf()
    vis.save_figure(plt.gcf(), image_save_directory=result_dir, filename=str(barplot.axes[0].get_title()).replace(' ', '_'))

    # Visualize missing data with missingno
    #fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    msno.matrix(features)
    fig_matrix = plt.gcf()
    vis.save_figure(fig_matrix, image_save_directory=result_dir, filename='missing_numbers_matrix')

    #plt.savefig(os.path.join(result_dir,'_missing_numbers_matrix'))
    #plt.show(block = False)

    if features.isnull().values.sum() > 0:
        plt.gcf()
        msno.heatmap(features)
        vis.save_figure(plt.gcf(), image_save_directory=result_dir, filename='missing_numbers_heatmap')
        #plt.savefig(os.path.join(result_dir, '_missing_numbers_heatmap'))
        #plt.show(block = False)

    #### View Prepared Binary Features
    # We need some more plots for the binary data types.

    # vis.plotBinaryValues(df_dig, df_dig.columns) #0:-1
    # plt.savefig(image_save_directory + "/BinaryFeatures.png", dpi=70)

    return features, y, class_labels

def main(config_path):
    conf = sup.load_config(config_path)

    data_directory = conf['Paths'].get('prepared_data_directory')

    data_preparation_dump_file_path = os.path.join(data_directory, "temp", "step31out.pickle")
    (features_cleaned1, outcomes_cleaned1, class_labels,
     data_source_raw, data_directory, result_directory) = pickle.load(open(data_preparation_dump_file_path, "rb" ))

    #dataset_name = conf['Common'].get('dataset_name')
    class_name = conf['Common'].get('class_name')

    #model_features_filename = os.path.join(data_directory, dataset_name + "_" + class_name + "_features_model.csv")
    #model_outcomes_filename = os.path.join(data_directory, dataset_name + "_" + class_name + "_outcomes_model.csv")
    #model_labels_filename = os.path.join(data_directory, dataset_name + "_" + class_name + "_labels_model.csv")

    model_features_filename = os.path.join(conf['Preparation'].get('features_out'))
    model_outcomes_filename = os.path.join(conf['Preparation'].get('outcomes_out'))
    model_labels_filename = os.path.join(conf['Preparation'].get('labels_out'))

    features, y, class_labels = adapt_features_for_model(features_cleaned1, outcomes_cleaned1, result_directory,
                                                         class_labels, conf)

    # === Save features to a csv file ===#
    print("Features shape {}".format(features.shape))
    features.to_csv(model_features_filename, sep=';', index=True)
    # np.savetxt(filenameprefix + "_X.csv", X, delimiter=";", fmt='%s')
    print("Saved features to " + model_features_filename)

    # === Save the selected outcome to a csv file ===#
    if y is not None:
        print("outcome shape {}".format(y.shape))
        y_true = pd.DataFrame(y, columns=[class_name], index=outcomes_cleaned1.index)
        y_true.to_csv(model_outcomes_filename, sep=';', index=True, header=True)
        print("Saved features to " + model_outcomes_filename)
    else:
        print("y values not saved as no ourcome was provided.")

    # === Save new y labels to a csv file ===#
    if class_labels is not None:
        print("Class labels length {}".format(len(class_labels)))
        with open(model_labels_filename, 'w') as f:
            for key in class_labels.keys():
                f.write("%s;%s\n" % (class_labels[key],
                                     key))  # Classes are saved inverse to the labelling in the file, i.e. first value, then key
        print("Saved class names and id to " + model_labels_filename)
    else:
        print("Class labels were not saved as no outcome was available.")


if __name__ == "__main__":
    main(args.config_path)

    print("=== Program end ===")