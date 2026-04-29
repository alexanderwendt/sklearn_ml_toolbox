#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Analyze training time and performance.

This script analyzes the training time and performance of a machine learning model
as a function of the training set size. It helps in understanding how the model
scales with more data and provides insights into the learning process.

Inputs:
    - Training and validation data from the previous steps.
    - A machine learning pipeline (specified in the config).

Outputs:
    - `Duration_Samples.png`: A plot showing the training duration as a function
      of the number of training samples.
    - `F1_Samples.png`: A plot showing the F1 score on the validation set as a
      function of the number of training samples.

Main Functions:
    - `run_training_estimation`: Executes the training and evaluation for different
      subset sizes of the training data and generates the plots.
    - `run_training_predictors`: Loads the data and the model pipeline, and calls
      `run_training_estimation`.

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
import sys
import argparse
import os
import logging
from pydoc import locate

# Libs
import numpy as np
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

# Own modules
import utils.data_visualization_functions as vis
import utils.execution_utils as exe
import utils.data_handling_support_functions as sup

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

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/" + "toolbox" + ".log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Step 4 - Calculate predictions for training like estimated time"
)
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)

args = parser.parse_args()


def run_training_estimation(
    X_train, y_train, X_test, y_test, scorer, model_clf, image_save_directory=None
):
    """
    Run estimation of scorer and duration dependent of subset size of input data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : pd.DataFrame
        Test data.
    y_test : np.ndarray
        Test labels.
    scorer : callable
        Scorer for the evaluation.
    model_clf : object
        The machine learning model.
    image_save_directory : str, optional
        The directory where the plots will be saved, by default None.
    """
    test_range = list(range(100, 6500 + 1, 500))
    print("Test range", test_range)

    xaxis, durations, scores = exe.estimate_training_duration(
        model_clf, X_train, y_train, X_test, y_test, test_range, scorer
    )

    plt.figure()
    plt.plot(xaxis, durations)
    plt.xlabel("Number of training examples")
    plt.ylabel("Duration [s]")
    plt.title("Training Duration")

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="Duration_Samples"
    )

    plt.figure()
    plt.plot(xaxis, scores)
    plt.xlabel("Number of training examples")
    plt.ylabel(
        "F1-Score on cross validation set (=the rest). Size={}".format(X_test.shape[0])
    )
    plt.title("F1 Score Improvement With More Data")

    vis.save_figure(
        plt.gcf(), image_save_directory=image_save_directory, filename="F1_Samples"
    )


def run_training_predictors(data_input_path):
    """
    Run training predictors.

    Parameters
    ----------
    data_input_path : str
        Path to the data input.
    """

    config = sup.load_config(data_input_path)

    pipeline_class_name = config.get("Training", "pipeline_class", fallback=None)
    PipelineClass = locate("models." + pipeline_class_name + ".ModelParam")
    model_param = PipelineClass()
    if model_param is None:
        raise Exception(
            "Model pipeline could not be found: {}".format(
                "models." + pipeline_class_name + ".ModelParam"
            )
        )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        y_classes,
        selected_features,
        feature_dict,
        paths,
        scorers,
        refit_scorer_name,
    ) = exe.load_training_input_input(config)
    scorer = scorers[refit_scorer_name]

    results_directory = paths["results_directory"]
    save_fig_prefix = results_directory + "/model_images"

    baseline_results = exe.execute_baseline_classifier(
        X_train, y_train, X_val, y_val, y_classes, scorer
    )
    print("Baseline results=", baseline_results)

    model_clf = model_param.create_pipeline()["model"]
    log.info("{} selected.".format(model_clf))

    run_training_estimation(
        X_train, y_train, X_val, y_val, scorer, model_clf, save_fig_prefix
    )


if __name__ == "__main__":
    run_training_predictors(args.config_path)

    print("=== Program end ===")
