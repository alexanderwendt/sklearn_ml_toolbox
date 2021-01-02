#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Hyperparameter search for T-SNE
License_info: TBD
"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
import os

# Libs
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display
#from matplotlib.ticker import FuncFormatter, MaxNLocator
#from pandas.core.dtypes.common import is_string_dtype
from pandas.plotting import register_matplotlib_converters
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Own modules
import data_handling_support_functions as sup

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

#Global settings
np.set_printoptions(precision=3)
#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 3.3 - Search Hyperparameters for unsupervised methods')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()

def find_tsne_parmeters(X_scaled_subset, y_subset, class_labels, conf, image_save_directory):
    # Optimize t-sne plot
    #tne_gridsearch = False
    # Create a TSNE grid search with two variables
    perplex = [5, 10, 30, 50, 100]
    exaggregation = [5, 12, 20, 50, 100]
    # learning_rate = [10, 50, 200]
    fig, axarr = plt.subplots(len(perplex), len(exaggregation), figsize=(15, 15))
    #if tne_gridsearch == True:
    # for m,l in enumerate(learning_rate):
    for k, p in enumerate(perplex):
        # print("i {}, p {}".format(i, p))
        for j, e in enumerate(exaggregation):
            # print("j {}, e {}".format(j, e))
            X_embedded = TSNE(n_components=2, perplexity=p, early_exaggeration=e, n_iter=5000,
                              n_iter_without_progress=1000, learning_rate=10).fit_transform(
                X_scaled_subset)

            for i, t in enumerate(set(y_subset)):
                idx = y_subset == t
                axarr[k, j].scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=class_labels[t])

            axarr[k, j].set_title("p={}, e={}".format(p, e))

            # clear_output(wait=True)
            print('perplex paramater k={}/{}, exaggregation parameter perplexity={}/{}'.format(k, len(perplex), j,
                                                                                   len(exaggregation)))
    fig.subplots_adjust(hspace=0.3)

    plt.gcf()

    if image_save_directory:
        if not os.path.isdir(image_save_directory):
            os.makedirs(image_save_directory)
        plt.savefig(os.path.join(image_save_directory, conf['Common'].get('dataset_name') + '_TSNE_Calibration_Plot'), dpi=300)


    plt.show(block = False)


def main():
    conf = sup.load_config(args.config_path)
    features, y, df_y, class_labels = sup.load_features(conf)

    source_filename = conf['Paths'].get("training_data_directory") + "/" + conf['Common'].get('dataset_name') + "_source" + ".csv"
    source = sup.load_data_source(source_filename)

    image_save_directory = conf['Paths'].get('result_directory') + "/analysis_data_analysis"

    scaler = StandardScaler()  # Because normal distribution. Don't use minmax scaler for PCA or unsupervised learning
    # as the axis shall be centered and not shifted.
    scaler.fit(features)
    # Use this scaler also for the test data at the end
    X_scaled = pd.DataFrame(data=scaler.transform(features), index=features.index, columns=features.columns)
    print("Unscaled values")
    print(features.iloc[0:2, :])
    print("Scaled values")
    print(X_scaled.iloc[0:2, :])
    scaler.fit(df_y)
    y_scaled = pd.DataFrame(data=scaler.transform(df_y), index=df_y.index, columns=df_y.columns)

    # Reduce the training set with the number of samples randomly chosen
    X_train_index_subset = sup.get_random_data_subset_index(1000, features)
    X_train_scaled_subset = X_scaled.iloc[X_train_index_subset, :]
    y_train_subset = np.array(y[X_train_index_subset]).flatten()

    find_tsne_parmeters(X_train_scaled_subset, y_train_subset, class_labels, conf, image_save_directory)

    #analyze_timegraph(source, features, y, conf, image_save_directory)
    #analyse_features(features, y, class_labels, source, conf, image_save_directory)


if __name__ == "__main__":
    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    main()


    print("=== Program end ===")