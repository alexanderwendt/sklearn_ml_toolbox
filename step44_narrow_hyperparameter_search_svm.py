#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Train narrow search
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
import os

# Libs
import argparse
from pandas.plotting import register_matplotlib_converters
import pickle
from pickle import dump

from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

import pickle

#from IPython.core.display import display

import execution_utils as exe
import data_visualization_functions_for_SVM as svmvis
import matplotlib.pyplot as plt

## %% First run with a wide grid search
# Minimal set of parameter to test different grid searches
from pickle import dump

#import sklearn_utils as modelutil
import numpy as np
#import copy

#from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import ADASYN
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.combine import SMOTEENN
#from imblearn.combine import SMOTETomek

# Own modules
#import data_visualization_functions as vis
import data_handling_support_functions as sup
import execution_utils as exe

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)
#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 4 - Execute narrow incremental search for SVM')
#parser.add_argument("-exe", '--execute_narrow', default=False, action='store_true', help='Execute narrow training')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()

def execute_search_iterations_random_search_SVM(X_train, y_train, init_parameter_svm, pipe_run_random, scorers,
                                                refit_scorer_name, save_fig_prefix):
    '''
    Iterated search for parameters. Set sample size, kfolds, number of iterations and top result selection. Execute
    random search cv for the number of entries and extract the best parameters from that search. As a result the
    best C and gamma are extracted.

    :args:
        X_train: Training data, featrues X
        y_train: Training labels, ground truth y
        init_parameter_svm: Initial SVM parameters C and gamma
        pipe_run_random: ML Pipe
        scorers: scorers to use
        refit_scorer_name: Refit scrorer
        save_fig_prefix: Prefix for images from the analysis

    :return:
        param_final: Final parameters C and gamma
    '''

    # Iterated pipeline with increasing number of tries
    sample_size = [200, 400, 600]
    kfolds = [2, 3, 3]
    number_of_interations = [100, 100, 20]
    select_from_best = [10, 10, 10]

    combined_parameters = zip(sample_size, kfolds, number_of_interations, select_from_best)

    new_parameter_rand = init_parameter_svm  # Initialize the system with the parameter borders

    for i, combination in enumerate(combined_parameters):
        sample_size, folds, iterations, selection = combination
        print("Start random optimization run {} with the following parameters: ".format(i))
        print("Sample size: ", sample_size)
        print("Number of folds: ", folds)
        print("Number of tries: ", iterations)
        print("Number of best results to select from: ", selection)

        # Run random search
        new_parameter_rand, results_random_search, clf = exe.run_random_cv_for_SVM(X_train, y_train, new_parameter_rand,
                                                                   pipe_run_random, scorers,
                                                                   refit_scorer_name, number_of_samples=sample_size,
                                                                   kfolds=folds,
                                                                   n_iter_search=iterations, plot_best=selection)
        print("Got best parameters: ")
        print(new_parameter_rand)

        # Display random search results
        ax = svmvis.visualize_random_search_results(clf, refit_scorer_name)
        ax_enhanced = svmvis.add_best_results_to_random_search_visualization(ax, results_random_search, selection)

        plt.gca()
        plt.tight_layout()
        plt.savefig(save_fig_prefix + '_' + 'run2_subrun_' + str(i) + '_samples' + str(sample_size) + '_fold'
                    + str(folds) + '_iter' + str(iterations) + '_sel' + str(selection), dpi=300)
        plt.show(block = False)
        plt.pause(0.01)
        plt.close()

        print("===============================================================")

    ##
    print("Best parameter limits: ")
    print(new_parameter_rand)

    print("Best results: ")
    print(results_random_search.round(3).head(10))

    param_final = {}
    param_final['C'] = results_random_search.iloc[0]['param_svm__C']
    param_final['gamma'] = results_random_search.iloc[0]['param_svm__gamma']

    #param_final = new_parameter_rand[0]
    print("Hyper parameters found")
    print(param_final)

    return param_final, results_random_search


def execute_narrow_search(config_path):
    '''
    Execute a narrow search on the subset of data


    '''

    config = sup.load_config(config_path)
    # Load file paths
    #paths, model, train, test = step40.load_training_files(paths_path)
    # Load complete training input
    X_train, y_train, X_val, y_val, y_classes, selected_features, \
    feature_dict, paths, scorers, refit_scorer_name = exe.load_training_input_input(config)
    #f = open(data_input_path, "rb")
    #prepared_data = pickle.load(f)
    #print("Loaded data: ", prepared_data)

    #results_run1_file_path = prepared_data['paths']['svm_run1_result_filename']

    #X_train = train['X']
    #y_train = train['y']
    #scorers = model['scorers']
    #refit_scorer_name = model['refit_scorer_name']
    results_run2_file_path = paths['svm_run2_result_filename']
    svm_pipe_first_selection = paths['svm_pipe_first_selection']
    #svm_pipe_final_selection = paths['svm_pipe_final_selection']
    #Use predefined export location for the pipe
    svm_pipe_final_selection = os.path.join(config['Paths'].get('model_directory'), config['Training'].get('pipeline_out'))
    model_directory = paths['model_directory']
    result_directory = paths['result_directory']
    model_name = paths['dataset_name']
    save_fig_prefix = result_directory + '/model_images'
    if not os.path.isdir(save_fig_prefix):
        os.makedirs(save_fig_prefix)
        print("Created folder: ", save_fig_prefix)


    #figure_path_prefix = model_directory + '/images/' + model_name


    # Load saved results
    r = open(svm_pipe_first_selection, "rb")
    pipe_run_best_first_selection = pickle.load(r)

    # Based on the kernel, get the initial range of continuous parameters
    parameter_svm = exe.get_continuous_parameter_range_for_SVM_based_on_kernel(pipe_run_best_first_selection)

    # Execute iterated random search where parameters are even more limited
    param_final, results_run2 = execute_search_iterations_random_search_SVM(X_train, y_train, parameter_svm, pipe_run_best_first_selection, scorers,
                                                refit_scorer_name,
                                                save_fig_prefix=save_fig_prefix + '/' + model_name)

    # Enhance kernel with found parameters
    pipe_run_best_first_selection['svm'].C = param_final['C']
    pipe_run_best_first_selection['svm'].gamma = param_final['gamma']

    print("Model parameters defined", pipe_run_best_first_selection)

    print("Save model")
    # Save best pipe
    dump(pipe_run_best_first_selection, open(svm_pipe_final_selection, 'wb'))
    print("Stored pipe_run_best_first_selection at ", svm_pipe_final_selection)

    # Save results
    dump(results_run2, open(results_run2_file_path, 'wb'))
    print("Stored results ", results_run2_file_path)

    result_save = results_run2.copy()
    #sup.list_to_name(selected_features, list(feature_dict.keys()), result_save['param_feat__cols'])
    results_run2.round(4).to_csv(results_run2_file_path + "_results.csv", sep=";")

    #result['pipe'].to_json(results_file_path + ".csv", sep=";")

    with open(results_run2_file_path + "_pipe.txt", 'w') as f:
        print(pipe_run_best_first_selection, file=f)

    print("Method end")


if __name__ == "__main__":

    # Execute narrow search
    execute_narrow_search(args.config_path)

    print("=== Program end ===")