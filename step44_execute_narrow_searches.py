import argparse
import os
import pickle

from IPython.core.display import display

import step40_functions as step40
import data_visualization_functions_for_SVM as svmvis
import matplotlib.pyplot as plt

## %% First run with a wide grid search
# Minimal set of parameter to test different grid searches
from pickle import dump

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
        new_parameter_rand, results_random_search, clf = step40.run_random_cv_for_SVM(X_train, y_train, new_parameter_rand,
                                                                   pipe_run_random, scorers,
                                                                   refit_scorer_name, number_of_samples=sample_size,
                                                                   kfolds=folds,
                                                                   n_iter_search=iterations, plot_best=selection)
        print("Got best parameters: ")
        display(new_parameter_rand)

        # Display random search results
        ax = svmvis.visualize_random_search_results(clf, refit_scorer_name)
        ax_enhanced = svmvis.add_best_results_to_random_search_visualization(ax, results_random_search, selection)

        plt.gca()
        plt.savefig(save_fig_prefix + '_' + 'run2_subrun_' + str(i) + '_samples' + str(sample_size) + '_fold'
                    + str(folds) + '_iter' + str(iterations) + '_sel' + str(selection), dpi=300)
        plt.show()

        print("===============================================================")

    ##
    print("Best parameter limits: ")
    display(new_parameter_rand)

    print("Best results: ")
    display(results_random_search.round(3).head(10))

    param_final = {}
    param_final['C'] = results_random_search.iloc[0]['param_svm__C']
    param_final['gamma'] = results_random_search.iloc[0]['param_svm__gamma']

    #param_final = new_parameter_rand[0]
    print("Hyper parameters found")
    display(param_final)

    return param_final, results_random_search


def execute_narrow_search(paths_path = "config/paths.pickle"):
    '''
    Execute a narrow search on the subset of data


    '''

    # Load file paths
    paths, model, train, test = step40.load_training_files(paths_path)
    #f = open(data_input_path, "rb")
    #prepared_data = pickle.load(f)
    #print("Loaded data: ", prepared_data)

    #results_run1_file_path = prepared_data['paths']['svm_run1_result_filename']

    X_train = train['X']
    y_train = train['y']
    scorers = model['scorers']
    refit_scorer_name = model['refit_scorer_name']
    results_run2_file_path = paths['svm_run2_result_filename']
    svm_pipe_first_selection = paths['svm_pipe_first_selection']
    svm_pipe_final_selection = paths['svm_pipe_final_selection']
    model_directory = paths['model_directory']
    result_directory = paths['result_directory']
    model_name = paths['dataset_name']
    save_fig_prefix = result_directory + '/model_images'
    if not os.path.isdir(save_fig_prefix):
        os.mkdir(save_fig_prefix)
        print("Created folder: ", save_fig_prefix)


    #figure_path_prefix = model_directory + '/images/' + model_name


    # Load saved results
    r = open(svm_pipe_first_selection, "rb")
    pipe_run_best_first_selection = pickle.load(r)

    # Based on the kernel, get the initial range of continuous parameters
    parameter_svm = step40.get_continuous_parameter_range_for_SVM_based_on_kernel(pipe_run_best_first_selection)

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
    print("Stored pipe_run_best_first_selection at ", results_run2_file_path)

    print("Method end")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.4 - Execute narrow incremental search for SVM')
    parser.add_argument("-exe", '--execute_narrow', default=True,
                        help='Execute narrow training', required=False)
    parser.add_argument("-d", '--data_path', default="config/paths.pickle",
                        help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    #execute_wide_run(execute_search=args.execute_wide, data_input_path=args.data_path)

    # Execute narrow search
    execute_narrow_search(paths_path=args.data_path)

    print("=== Program end ===")