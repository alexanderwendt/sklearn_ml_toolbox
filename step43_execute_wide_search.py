import argparse
import pickle

from IPython.core.display import display
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

import step40_functions as step40
import data_handling_support_functions as sup
import Sklearn_model_utils as modelutil
import numpy as np
import copy
import data_visualization_functions as vis

## %% First run with a wide grid search
# Minimal set of parameter to test different grid searches
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from pickle import dump


def execute_wide_search(data_input_path="04_Model" + "/" + "prepared_input.pickle"):
    ''' Execute the wide search algorithm

    :args:
        exe:
        data_path:
    :return:
        Nothing
    '''

    ### %% Load input
    #data_input_path = "04_Model" + "/" + "prepared_input.pickle"
    f = open(data_input_path, "rb")
    prepared_data = pickle.load(f)
    print("Loaded data: ", prepared_data)
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    y_classes = prepared_data['y_classes']
    scorers = prepared_data['scorers']
    refit_scorer_name = prepared_data['refit_scorer_name']
    selected_features = prepared_data['selected_features']
    results_file_path = prepared_data['paths']['svm_run1_result_filename']

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

    #grid_search_run1, params_run1, pipe_run1, results_run1 = step40.run_basic_svm(X_train, y_train, selected_features,
    #                                                                      scorers, refit_scorer_name,
    #                                                                      subset_share=0.10, n_splits=3,
    #                                                                      )
    grid_search_run1, params_run1, pipe_run1, results_run1 = step40.run_basic_svm(X_train, y_train, selected_features,
                                                                          scorers, refit_scorer_name,
                                                                          subset_share=0.01, n_splits=2,
                                                                          parameters=params_debug)
    print('Final score is: ', grid_search_run1.score(X_test, y_test))

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
    display(result['result'].round(4).head(10))

    # Save results
    dump(result, open(results_file_path, 'wb'))
    print("Stored results of run 1 to ", results_file_path)


def extract_categorical_visualize_graphs(data_input_path="04_Model" + "/" + "prepared_input.pickle", top_percentage = 0.2):
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
    r = open(data_input_path, "rb")
    prepared_data = pickle.load(r)

    model_name = prepared_data['paths']['dataset_name']
    model_directory = prepared_data['paths']['model_directory']
    results_file_path = prepared_data['paths']['svm_run1_result_filename']
    refit_scorer_name = prepared_data['refit_scorer_name']
    selected_features = prepared_data['selected_features']
    feature_dict = prepared_data['feature_dict']
    X_train = prepared_data['X_train']
    svm_pipe_first_selection = prepared_data['paths']['svm_pipe_first_selection']

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
    print("The top {} of the results are used, i.e {} samples".format(top_percentage * 100, number_results))
    results_subset = results_run1.iloc[0:number_results,:]

    ## %% Plot graphs

    # Prepare the inputs: Replace the lists with strings
    result_subset_copy = results_subset.copy()
    print("Convert feature lists to names")
    sup.list_to_name(selected_features, list(feature_dict.keys()), result_subset_copy['param_feat__cols'])

    # Replace lists in the parameters with strings
    params_run1_copy = copy.deepcopy(params_run1)
    sup.replace_lists_in_grid_search_params_with_strings(selected_features, feature_dict, params_run1_copy)

    # Plot the graphs
    _, scaler_medians = vis.visualize_parameter_grid_search('scaler', params_run1, results_subset, refit_scorer_name,
                                                            save_fig_prefix=model_directory + '/images/' + model_name)
    _, sampler_medians = vis.visualize_parameter_grid_search('sampling', params_run1, results_subset, refit_scorer_name,
                                                             save_fig_prefix=model_directory + '/images/' + model_name)
    _, kernel_medians = vis.visualize_parameter_grid_search('svm__kernel', params_run1, results_subset, refit_scorer_name,
                                                            save_fig_prefix=model_directory + '/images/' + model_name)
    _, feat_cols_medians = vis.visualize_parameter_grid_search('feat__cols', params_run1_copy, result_subset_copy, refit_scorer_name,
                                                               save_fig_prefix=model_directory + '/images/' + model_name)

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

    display(pipe_run_best_first_selection)

    # Save best pipe
    dump(pipe_run_best_first_selection, open(svm_pipe_first_selection, 'wb'))
    print("Stored pipe_run_best_first_selection at ", svm_pipe_first_selection)

    print("Method end")

def execute_wide_run(execute_search=True, data_input_path="04_Model" + "/" + "prepared_input.pickle"):
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


    #Execute algotihm
    if execute_search=='True':
        print("Execute grid search")
        execute_wide_search(data_input_path)
    else:
        print("No grid search performed. Already existing model loaded.")

    #Visualize
    extract_categorical_visualize_graphs(data_input_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.3 - Execute wide grid search for SVM')
    parser.add_argument("-exe", '--execute_wide', default=False,
                        help='Execute Training', required=False)
    parser.add_argument("-d", '--data_path', default="04_Model/prepared_input.pickle",
                        help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    execute_wide_run(execute_search=args.execute_wide, data_input_path=args.data_path)

    print("=== Program end ===")