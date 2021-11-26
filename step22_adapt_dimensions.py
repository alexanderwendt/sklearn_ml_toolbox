#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2X Data Generation: Adapt dimensions of generated features and outcomes
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
import os
import pandas as pd
import numpy as np
import argparse

# Own modules
import utils.data_handling_support_functions as sup
import utils.custom_methods as custom

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

parser = argparse.ArgumentParser(description='Step 2.0 - Generate features and outcomes from raw data')
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


def cut_unusable_parts_of_dataframe(df, head_index=-1, tail_index=-1):
    '''
    Remove samples that are not useful, e.g. the first samples from a moving average

    '''

    if tail_index > 0:
        dfr = df.drop(df.tail(tail_index).index, inplace=False)
    if head_index > 0:
        dfr = df.drop(df.head(head_index).index, inplace=False)
    #Drop from the timerows too
    #source_cut = source.drop(source.tail(50).index, inplace=False)

    return dfr

def cut_dataframe_subset(df, start_loc, stop_loc):
    '''
    Remove all but a specified subset of the dataset, e.g. for debug training

    '''
    return df.iloc[start_loc:stop_loc, :]



def clean_nan(df):
    '''
    Remove all NaN
    # Assign the columns to X and y
    # Clean the featurestable and make an array or it
    # Drop all NaN rows

    '''

    nan_index = pd.isnull(df).any(1).to_numpy().nonzero()[0]
    print("Found {} rows with NaN".format(len(nan_index)))
    df_nonan = df.drop(nan_index)

    print("Got df shape={} from original shape={}".format(df_nonan.shape, df.shape))
    print(df_nonan.head(5))
    print(df_nonan.tail(5))

    return df_nonan

    #nan_index = pd.isnull(features).any(1).nonzero()[0]
    # Drop for features
    #features_nonan = features.drop(nan_index)
    # Drop for outcomes
    #outcomes_nonan = outcomes.drop(nan_index)
    # Drop for source
    #source_nonan = source.drop(nan_index)

    # print("New index range=", featues_nonan.index)
    #print("Got features features shape={}, outcomes shape={}, source shape={}".format(features_nonan.shape,
    #                                                                                  outcomes_nonan.shape,
    #                                                                                  source_nonan.shape))

    #print(features_nonan.head(5))
    #print(features_nonan.tail(5))
    #print(source_nonan.head(5))
    #print(source_nonan.tail(5))

def main(config_path):
    conf = sup.load_config(config_path)

    #image_save_directory = conf['result_directory'] + "/data_preparation_images"
    prepared_data_directory = conf['Paths'].get('prepared_data_directory')
    outcomes_filename_uncut = os.path.join(prepared_data_directory, "temp", "temp_outcomes_uncut" + ".csv")
    features_filename_uncut = os.path.join(prepared_data_directory, "temp", "temp_features_uncut" + ".csv")

    # Load only a subset of the whole raw data to create a debug dataset
    source_uncut = custom.load_source(conf['Paths'].get('source_path'))
    features_uncut = pd.read_csv(features_filename_uncut, sep=';').set_index('id')
    if os.path.isfile(outcomes_filename_uncut):
        outcomes_uncut = pd.read_csv(outcomes_filename_uncut, sep=';').set_index('id')
        print("Outcomes file found. Adapting dimensions for training data.")
        print("Outcomes shape: ", outcomes_uncut.shape)
    else:
        outcomes_uncut = None
        print("Outcomes file not found. Adapting dimensions for inference data.")

    print("Source shape: ", source_uncut.shape)
    print("Features shape: ", features_uncut.shape)


    # Cut outcomes and by last 50 as smoothing was used
    #outcomes_reduced1 = cut_unusable_parts_of_dataframe(outcomes_uncut, tail_index=50)

    #Clean features # Cut NaNs
    features_reduced1 = clean_nan(features_uncut)

    if not outcomes_uncut is None:
        intersection_index = outcomes_uncut.index.intersection(features_reduced1.index)

        # Cut all dataframes to have the same index
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


    # Cut for subsets
    subset_start = 0
    subset_stop = features.shape[0]
    #subset_stop = 1000

    features_subset = cut_dataframe_subset(features, subset_start, subset_stop)
    source_subset = cut_dataframe_subset(source, subset_start, subset_stop)

    print("Subset source shape: ", source_subset.shape)
    print("Subset features shape: ", features_subset.shape)

    outcomes_out_filename = os.path.join(conf['Generation'].get('outcomes_out')) #conf['Common'].get('dataset_name') + "_outcomes" + ".csv")
    features_out_filename = os.path.join(conf['Generation'].get('features_out'))

    source_out_filename = os.path.join(conf['Generation'].get('source_out'))

    print("=== Paths ===")
    print("Features in: ", features_out_filename)
    print("Outcomes in: ", outcomes_out_filename)
    print("Source out: ", source_out_filename)

    # Save the graph data for visualization of the results
    print("Feature shape {}".format(features_subset.shape))
    features_subset.to_csv(features_out_filename, sep=';', index=True, header=True)
    print("Saved features graph to " + features_out_filename)

    # Save the graph data for visualization of the results
    print("source shape {}".format(source_subset.shape))
    source_subset.to_csv(source_out_filename, sep=';', index=True, header=True)
    print("Saved source graph to " + source_out_filename)

    if not outcomes is None:
        outcomes_subset = cut_dataframe_subset(outcomes, subset_start, subset_stop)
        print("Subset outcomes shape: ", outcomes_subset.shape)

        # Save the graph data for visualization of the results
        print("Outcomes shape {}".format(outcomes_subset.shape))
        outcomes_subset.to_csv(outcomes_out_filename, sep=';', index=True, header=True)
        print("Saved source graph to " + outcomes_out_filename)

if __name__ == "__main__":

    main(args.config_path)

    print("=== Program end ===")