#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Perform a narrow, incremental hyperparameter search.

This script executes a narrow, iterative search to fine-tune the hyperparameters
of the machine learning model. It starts with the best parameters from the wide
search and explores the surrounding parameter space in more detail.

Inputs:
    - Training data.
    - The best pipeline from the wide search (`pipe_first_selection.pickle`).

Outputs:
    - `run2_result.pickle`: A pickle file containing the results of the narrow search.
    - `pipeline_out.pickle`: A pickle file with the final, fine-tuned pipeline.
    - Visualization of the narrow search results, saved as PNG files in the
      `model_images` subdirectory of the results directory.

Main Functions:
    - `execute_search_iterations_random_search_SVM`: Performs the iterative random
      search for SVM hyperparameters.
    - `perform_run2_svm`: Orchestrates the narrow search for SVM models.
    - `perform_run2_xgboost`: Orchestrates the narrow search for XGBoost models.
    - `execute_narrow_search`: Loads the data and the best pipeline from the wide
      search, and then calls the appropriate narrow search function based on the
      model type.

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
import json
import os

# Libs
import argparse
import warnings
from pydoc import locate
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
from pickle import dump
import numpy as np

# Own modules
import utils.data_visualization_functions as vis
import utils.data_handling_support_functions as sup
import utils.execution_utils as exe
import utils.data_visualization_functions_for_SVM as svmvis

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
    description="Step 4 - Execute narrow incremental search"
)
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)

args = parser.parse_args()


def execute_search_iterations_random_search_SVM(
    X_train,
    y_train,
    init_parameter_svm,
    pipe_run_random,
    scorers,
    refit_scorer_name,
    iter_setup,
    save_fig_prefix=None,
    problem_type="classification",
):
    """
    Iterated search for parameters for SVM.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : np.ndarray
        Training labels.
    init_parameter_svm : dict
        Initial SVM parameters.
    pipe_run_random : object
        The machine learning pipeline.
    scorers : dict
        Scorers for the evaluation.
    refit_scorer_name : str
        The name of the refit scorer.
    iter_setup : dict
        The setup for the iterations.
    save_fig_prefix : str, optional
        The prefix for the saved figures, by default None.
    problem_type : str, optional
        The type of problem, either 'classification' or 'regression', by default 'classification'.

    Returns
    -------
    tuple
        A tuple containing the final parameters and the results of the random search.
    """

    sample_size = list((np.array(iter_setup["samples"]) * X_train.shape[0]).astype(int))
    kfolds = iter_setup["kfolds"]
    number_of_interations = iter_setup["iter"]
    select_from_best = iter_setup["selection"]

    combined_parameters = zip(
        sample_size, kfolds, number_of_interations, select_from_best
    )

    new_parameter_rand = init_parameter_svm

    for i, combination in enumerate(combined_parameters):
        sample_size, folds, iterations, selection = combination
        print(
            "Start random optimization run {} with the following parameters: ".format(i)
        )
        print("Sample size: ", sample_size)
        print("Number of folds: ", folds)
        print("Number of tries: ", iterations)
        print("Number of best results to select from: ", selection)

        new_parameter_rand, results_random_search, clf = exe.run_random_cv_for_SVM(
            X_train,
            y_train,
            new_parameter_rand,
            pipe_run_random,
            scorers,
            refit_scorer_name,
            number_of_samples=sample_size,
            kfolds=folds,
            n_iter_search=iterations,
            plot_best=selection,
            problem_type=problem_type,
        )
        print("Got best parameters: ")
        print(new_parameter_rand)

        if problem_type == "classification":
            ax = svmvis.visualize_random_search_results(
                clf,
                refit_scorer_name,
                param_x="param_model__C",
                param_y="param_model__gamma",
            )
            ax_enhanced = svmvis.add_best_results_to_random_search_visualization(
                ax, results_random_search, selection
            )

            plt.gca()
            plt.tight_layout()

            vis.save_figure(
                plt.gcf(),
                image_save_directory=save_fig_prefix,
                filename="run2_subrun_"
                + str(i)
                + "_samples"
                + str(sample_size)
                + "_fold"
                + str(folds)
                + "_iter"
                + str(iterations)
                + "_sel"
                + str(selection),
            )

        print("===============================================================")

    print("Best parameter limits: ")
    print(new_parameter_rand)

    print("Best results: ")
    print(results_random_search.round(3).head(10))

    param_final = {}
    param_final["C"] = results_random_search.iloc[0]["param_model__C"]
    param_final["gamma"] = results_random_search.iloc[0]["param_model__gamma"]

    print("Hyper parameters found")
    print(param_final)

    return param_final, results_random_search


def execute_narrow_search(config_path):
    """
    Execute a narrow hyperparameter search.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """

    config = sup.load_config(config_path)
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
    problem_type = config["Common"].get("problem_type", fallback="classification")

    pipeline_class_name = config.get("Training", "pipeline_class", fallback=None)
    PipelineClass = locate("models." + pipeline_class_name + ".ModelParam")
    model_param = PipelineClass()
    if model_param is None:
        raise Exception(
            "Model pipeline could not be found: {}".format(
                "models." + pipeline_class_name + ".ModelParam"
            )
        )

    samples = json.loads(config.get("Training", "narrow_samples"))
    kfolds = json.loads(config["Training"].get("narrow_kfolds"))
    iterations = json.loads(config["Training"].get("narrow_iterations"))
    selection = json.loads(config["Training"].get("narrow_selection"))

    iter_setup = dict()
    iter_setup["samples"] = samples
    iter_setup["kfolds"] = kfolds
    iter_setup["iter"] = iterations
    iter_setup["selection"] = selection

    results_run2_file_path = paths["run2_result_filename"]
    pipe_first_selection = paths["pipe_first_selection"]
    pipe_final_selection = config.get("Training", "pipeline_out")
    result_directory = paths["results_directory"]
    save_fig_prefix = result_directory + "/model_images"
    os.makedirs(save_fig_prefix, exist_ok=True)

    r = open(pipe_first_selection, "rb")
    pipe_run_best_first_selection = pickle.load(r)

    if model_param.get_model_type() == "svm":
        pipe_run_second_selection, results_run2 = perform_run2_svm(
            X_train,
            iter_setup,
            pipe_run_best_first_selection,
            refit_scorer_name,
            save_fig_prefix,
            scorers,
            y_train,
            problem_type=problem_type,
        )
    else:
        warnings.warn(
            "No 2nd search will be performed for {}".format(model_param.get_model_type())
        )
        pipe_run_second_selection, results_run2 = perform_run2_xgboost(
            X_train,
            iter_setup,
            pipe_run_best_first_selection,
            refit_scorer_name,
            save_fig_prefix,
            scorers,
            y_train,
        )

    print("Model parameters defined", pipe_run_second_selection)

    print("Save model")
    dump(pipe_run_second_selection, open(pipe_final_selection, "wb"))
    print("Stored pipe_run_best_first_selection at ", pipe_final_selection)

    if results_run2 is not None:
        dump(results_run2, open(results_run2_file_path, "wb"))
        print("Stored results ", results_run2_file_path)

        results_run2.round(4).to_csv(results_run2_file_path + "_results.csv", sep=";")

    with open(results_run2_file_path + "_pipe.txt", "w") as f:
        print(pipe_run_second_selection, file=f)

    print("Method end")


def perform_run2_xgboost(
    X_train,
    iter_setup,
    pipe_run_best_first_selection,
    refit_scorer_name,
    save_fig_prefix,
    scorers,
    y_train,
):
    """
    Perform the second run with fine tuning for XGBoost.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    iter_setup : dict
        The setup for the iterations.
    pipe_run_best_first_selection : object
        The best pipeline from the first run.
    refit_scorer_name : str
        The name of the refit scorer.
    save_fig_prefix : str
        The prefix for the saved figures.
    scorers : dict
        Scorers for the evaluation.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    tuple
        A tuple containing the final pipeline and the results of the second run.
    """

    pipe_run_second_selection = deepcopy(pipe_run_best_first_selection)
    param_final, results_run2 = (None, None)
    warnings.warn(
        "For XGBoost, the same parameters are used as in pipe 1, i.e. no fine tuning. TODO: make fine tuning."
    )

    return pipe_run_second_selection, results_run2


def perform_run2_svm(
    X_train,
    iter_setup,
    pipe_run_best_first_selection,
    refit_scorer_name,
    save_fig_prefix,
    scorers,
    y_train,
    problem_type="classification",
):
    """
    Perform the second run with fine tuning for SVM.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    iter_setup : dict
        The setup for the iterations.
    pipe_run_best_first_selection : object
        The best pipeline from the first run.
    refit_scorer_name : str
        The name of the refit scorer.
    save_fig_prefix : str
        The prefix for the saved figures.
    scorers : dict
        Scorers for the evaluation.
    y_train : np.ndarray
        Training labels.
    problem_type : str, optional
        The type of problem, either 'classification' or 'regression', by default 'classification'.

    Returns
    -------
    tuple
        A tuple containing the final pipeline and the results of the second run.
    """

    pipe_run_second_selection = deepcopy(pipe_run_best_first_selection)

    parameter_svm = exe.get_continuous_parameter_range_for_SVM_based_on_kernel(
        pipe_run_best_first_selection
    )
    param_final, results_run2 = execute_search_iterations_random_search_SVM(
        X_train,
        y_train,
        parameter_svm,
        pipe_run_best_first_selection,
        scorers,
        refit_scorer_name,
        iter_setup,
        save_fig_prefix=save_fig_prefix + "/",
        problem_type=problem_type,
    )
    pipe_run_second_selection["model"].C = param_final["C"]
    pipe_run_second_selection["model"].gamma = param_final["gamma"]

    return pipe_run_second_selection, results_run2


if __name__ == "__main__":
    execute_narrow_search(args.config_path)

    print("=== Program end ===")
