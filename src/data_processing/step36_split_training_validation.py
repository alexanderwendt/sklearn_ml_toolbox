#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3X Preprocessing: Split data into training and validation sets.

This script splits the prepared data into training and validation sets. This is
a crucial step before training a machine learning model, as it allows for an
unbiased evaluation of the model's performance.

Inputs:
    - Features and outcomes from the previous steps (loaded via `sup.load_features`).

Outputs:
    - `features_out_train.csv`: CSV file with the training features.
    - `features_out_val.csv`: CSV file with the validation features.
    - `outcomes_out_train.csv`: CSV file with the training outcomes.
    - `outcomes_out_val.csv`: CSV file with the validation outcomes.

Main Functions:
    - `split_train_validation_data`: Splits the data into training and validation
      sets using `train_test_split` from scikit-learn and saves the results to
      CSV files.

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
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

# Own modules
import utils.data_handling_support_functions as sup
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

parser = argparse.ArgumentParser(
    description="Step 3 - Prepare data for machine learning algorithms"
)
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)

args = parser.parse_args()


def split_train_validation_data(config_path):
    """
    Split the data into training and validation sets.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    print("=== Split data into training and validation data ===")
    conf = sup.load_config(config_path)
    features, y, df_y, class_labels = sup.load_features(conf)

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        df_y,
        random_state=0,
        test_size=float(conf["Preparation"].get("test_size")),
        shuffle=conf["Preparation"].get("shuffle_data") == "True",
    )

    print(
        "Total number of samples: {}. X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(
            features.shape[0],
            X_train.shape,
            X_val.shape,
            y_train.shape,
            y_val.shape,
        )
    )

    if len(np.unique(y_train)) == 1:
        raise Exception(
            "y_train only consists one class after train/test split. Please adjust the data."
        )
    if len(np.unique(y_val)) == 1:
        raise Exception(
            "y_test only consists one class after train/test split. Please adjust the data."
        )

    X_train.to_csv(
        os.path.join(conf["Preparation"].get("features_out_train")),
        sep=";",
        index=True,
        header=True,
    )
    X_val.to_csv(
        os.path.join(conf["Preparation"].get("features_out_val")),
        sep=";",
        index=True,
        header=True,
    )
    y_train.to_csv(
        os.path.join(conf["Preparation"].get("outcomes_out_train")),
        sep=";",
        index=True,
        header=True,
    )
    y_val.to_csv(
        os.path.join(conf["Preparation"].get("outcomes_out_val")),
        sep=";",
        index=True,
        header=True,
    )

    print("Saved training and validation files.")


if __name__ == "__main__":
    split_train_validation_data(args.config_path)

    print("=== Program end ===")
