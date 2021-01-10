#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Train wide Search
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

#from IPython.core.display import display
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

import sklearn_utils as modelutil
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

# Own modules
import data_visualization_functions as vis
import data_handling_support_functions as sup
import execution_utils as exe
from evaluation_utils import Metrics
from filepaths import Paths

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

parser = argparse.ArgumentParser(description='Step 4 - Execute wide grid search for SVM')
parser.add_argument("-exe", '--execute_wide', default="True", help='Execute Wide Training run', required=False)
parser.add_argument("-debug", '--debug_param', default=False, action='store_true', help='Use debug parameters')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxs30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()


def load_input(conf):
    '''
    Load input model and data from a prepared pickle file

    :args:
        input_path: Input path of pickle file with prepared data
    :return:
        X_train: Training data
        y_train: Training labels as numbers
        X_test: Test data
        y_test: Test labels as numbers
        y_classes: Class names assigned to numbers
        scorer: Scorer for the evaluation, default f1

    '''
    # Load input
    X_train_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('features_train_in'))
    y_train_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('outcomes_train_in'))
    X_val_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('features_val_in'))
    y_val_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('outcomes_val_in'))

    # paths, model, train, test = step40.load_training_files(paths_path)
    X_train, _, y_train = exe.load_data(X_train_path, y_train_path)
    X_val, _, y_val = exe.load_data(X_val_path, y_val_path)

    labels = exe.load_labels(conf['Paths'].get('labels_path'))

    y_classes = labels  # train['label_map']

    print("Load feature columns")
    feature_columns_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('selected_feature_columns_in'))
    selected_features, feature_dict, df_feature_columns = exe.load_feature_columns(feature_columns_path, X_train)

    print("Load metrics")
    metrics = Metrics(conf)
    paths = Paths(conf).path
    scorers = metrics.scorers  # model['scorers']
    refit_scorer_name = metrics.refit_scorer_name  # model['refit_scorer_name']

    return X_train, y_train, X_val, y_val, y_classes, selected_features, feature_dict, paths, scorers, refit_scorer_name

def execute_wide_search(config, use_debug_parameters=False):
    ''' Execute the wide search algorithm

    :args:
        exe:
        data_path:
    :return:
        Nothing
    '''

    subset_share = float(config['Training'].get('subset_share')) #0.1

    # Load complete training input
    X_train, y_train, X_val, y_val, y_classes, selected_features, \
    feature_dict, paths, scorers, refit_scorer_name = exe.load_training_input_input(config)

    results_file_path = paths['svm_run1_result_filename']
    if not os.path.isdir(os.path.dirname(results_file_path)):
        os.makdir(os.path.dirname(results_file_path))
        print("Directory created: ", os.path.dirname(results_file_path))

    # Define parameters as an array of dicts in case different parameters are used for different optimizations
    params_debug = [{'scaler': [StandardScaler()],
                     'sampling': [modelutil.Nosampler(), SMOTE(), SMOTEENN(), ADASYN()],
                     'feat__cols': selected_features[0:2],
                     'svm__kernel': ['linear'],
                     'svm__C': [0.1, 1, 10],
                     'svm__gamma': [0.1, 1, 10],
                     },
                    {
                        'scaler': [StandardScaler(), Normalizer()],
                        'sampling': [modelutil.Nosampler()],
                        'feat__cols': selected_features[0:1],
                        'svm__C': [1],  # default C=1
                        'svm__kernel': ['rbf'],
                        'svm__gamma': [1]
                        # Only relevant in rbf, default='auto'=1/n_features
                    }]

    if use_debug_parameters:
        grid_search_run1, params_run1, pipe_run1, results_run1 = exe.run_basic_svm(X_train, y_train,
                                                                                      selected_features,
                                                                                      scorers, refit_scorer_name,
                                                                                      subset_share=0.01, n_splits=2,
                                                                                      parameters=params_debug)
    else:

        grid_search_run1, params_run1, pipe_run1, results_run1 = exe.run_basic_svm(X_train, y_train, selected_features,
                                                                              scorers, refit_scorer_name,
                                                                              subset_share=subset_share, n_splits=3,
                                                                              )

    print('Final score is: ', grid_search_run1.score(X_val, y_val))

    result = dict()
    result['parameter'] = params_run1
    result['model'] = grid_search_run1
    result['result'] = results_run1
    result['pipe'] = pipe_run1

    ## %%
    # merged_params_run1={}
    # for d in params_run1:
    #    merged_params_run1.update(d)

    # results_run1 = modelutil.generate_result_table(grid_search_run1, merged_params_run1, refit_scorer_name)
    # print("Result size=", results_run1.shape)
    # print("Number of NaN results: {}. Replace them with 0".format(np.sum(results_run1['mean_test_' + refit_scorer_name].isna())))
    # results_run1[results_run1['mean_test_' + refit_scorer_name].isna()]=0
    # print("Result size after dropping NAs=", results_run1.shape)
    print(result['result'].round(4).head(10))

    # Save results
    dump(result, open(results_file_path, 'wb'))

    print("Stored results of run 1 to ", results_file_path)


def extract_categorical_visualize_graphs(config, top_percentage = 0.2):
    '''
    Of the results of a wide search algorithm, find the top x percent (deafult=20%), calculate the median value for
    each parameter and select the parameter value with the best median result.

    Visualize results of the search algorithm.

    :args:
        data_path: Path to the pickle with the complete results of the run
        top_percentage: Top share of results to consider. def: 0.2.
    :return:
        Nothing
    '''

    # Get necessary data from the data preparation
    #X_train, y_train, X_val, y_val, y_classes = load_input(config)
    X_train, y_train, X_val, y_val, y_classes, selected_features, \
    feature_dict, paths, scorers, refit_scorer_name = exe.load_training_input_input(config)

    #feature_columns_path = os.path.join(config['Paths'].get('prepared_data_directory'), config['Training'].get('selected_feature_columns_in'))
    #selected_features, feature_dict, df_feature_columns = exe.load_feature_columns(feature_columns_path, X_train)

    #model_directory = paths['model_directory']
    result_directory = paths['result_directory']
    results_file_path = paths['svm_run1_result_filename']

    svm_pipe_first_selection = paths['svm_pipe_first_selection']

    s = open(results_file_path, "rb")
    results = pickle.load(s)
    results_run1 = results['result']
    params_run1 = results['parameter']

    #Create result table
    #merged_params_run1 = {}
    #for d in params_run1:
    #    merged_params_run1.update(d)

    # Get the top x% values from the results
    # number of results to consider
    #top_percentage = 0.2
    number_results = np.int(results_run1.shape[0] * top_percentage)
    print("The top {}% of the results are used, i.e {} samples".format(top_percentage * 100, number_results))
    results_subset = results_run1.iloc[0:number_results,:]

    # Prepare the inputs: Replace the lists with strings
    result_subset_copy = results_subset.copy()
    print("Convert feature lists to names")
    sup.list_to_name(selected_features, list(feature_dict.keys()), result_subset_copy['param_feat__cols'])

    # Replace lists in the parameters with strings
    params_run1_copy = copy.deepcopy(params_run1)
    sup.replace_lists_in_grid_search_params_with_strings(selected_features, feature_dict, params_run1_copy)

    # Plot the graphs
    save_fig_prefix = result_directory + '/model_images'
    if not os.path.isdir(save_fig_prefix):
        os.makedirs(save_fig_prefix)
        print("Created folder: ", save_fig_prefix)

    _, scaler_medians = vis.visualize_parameter_grid_search('scaler', params_run1, results_subset, refit_scorer_name,
                                                            save_fig_prefix=save_fig_prefix + "/")
    _, sampler_medians = vis.visualize_parameter_grid_search('sampling', params_run1, results_subset, refit_scorer_name,
                                                             save_fig_prefix=save_fig_prefix + "/")
    _, kernel_medians = vis.visualize_parameter_grid_search('svm__kernel', params_run1, results_subset, refit_scorer_name,
                                                            save_fig_prefix=save_fig_prefix + "/")
    _, feat_cols_medians = vis.visualize_parameter_grid_search('feat__cols', params_run1_copy, result_subset_copy, refit_scorer_name,
                                                               save_fig_prefix=save_fig_prefix + "/")

    ## Get the best parameters

    # Get the best scaler from median
    best_scaler = max(scaler_medians, key=scaler_medians.get)
    print("Best scaler: ", best_scaler)
    best_sampler = max(sampler_medians, key=sampler_medians.get)
    print("Best sampler: ", best_sampler)
    best_kernel = max(kernel_medians, key=kernel_medians.get)
    print("Best kernel: ", best_kernel)

    # Get best feature result
    # Get the best kernel
    best_feat_cols = max(feat_cols_medians, key=feat_cols_medians.get)  # source.idxmax()
    # print("Best {}: {}".format(name, best_feature_combi))
    best_columns = feature_dict.get(best_feat_cols)

    print("Best feature selection: ", best_feat_cols)
    print("Best column indices: ", best_columns)  # feature_dict.get((results_run1[result_columns_run1].loc[indexList]['param_feat__cols'].iloc[best_feature_combi])))
    print("Best column names: ", list(X_train.columns[best_columns]))

    # Define pipeline, which is constant for all tests
    pipe_run_best_first_selection = Pipeline([
        ('scaler', best_scaler),
        ('sampling', best_sampler),
        ('feat', modelutil.ColumnExtractor(cols=best_columns)),
        ('svm', SVC(kernel=best_kernel))
    ])

    print(pipe_run_best_first_selection)

    # Save best pipe
    dump(pipe_run_best_first_selection, open(svm_pipe_first_selection, 'wb'))
    print("Stored pipe_run_best_first_selection at ", svm_pipe_first_selection)

    result_save = results_run1.copy()
    sup.list_to_name(selected_features, list(feature_dict.keys()), result_save['param_feat__cols'])
    result_save.to_csv(results_file_path + ".csv", sep=";")

    #result['pipe'].to_json(results_file_path + ".csv", sep=";")

    with open(results_file_path + "_pipe.txt", 'w') as f:
        print("=== Best results of run1 ===", file=f)
        print("Best scaler: ", best_scaler, file=f)
        print("Best sampler: ", best_sampler, file=f)
        print("Best kernel: ", best_kernel, file=f)
        print("Best feature selection: ", best_feat_cols, file=f)
        print("Best column indices: ", best_columns, file=f)
        print("Best column names: ", list(X_train.columns[best_columns]), file=f)
        print("=== Best pipe after discrete parameters have been fixed ===", file=f)
        print(pipe_run_best_first_selection, file=f)

    print("Method end")

def execute_wide_run(config_path, execute_search=True, debug_parameters=False):
    '''
    Execute a wide hyperparameter grid search, visualize the results and extract the best categorical parameters
    for SVM

    :args:
        execute_search: True if the search shall be executed; False if no search and only visualization and extraction
            of the best features
        data_path: Path to the pickle with the complete results of the run
    :return:
        Nothing
    '''

    conf = sup.load_config(config_path)
    #metrics = Metrics(conf)

    #Execute algotihm
    if execute_search==True:
        if debug_parameters:
            print("WARNING: Debug parameters are used, which only use a small subset of the search.")
        print("Execute grid search")
        execute_wide_search(conf, use_debug_parameters=debug_parameters)
    else:
        print("No grid search performed. Already existing model loaded.")

    #Visualize
    extract_categorical_visualize_graphs(conf)


if __name__ == "__main__":

    # Execute wide search
    execute_wide_run(args.config_path, execute_search=args.execute_wide=="True", debug_parameters=args.debug_param)

    print("=== Program end ===")