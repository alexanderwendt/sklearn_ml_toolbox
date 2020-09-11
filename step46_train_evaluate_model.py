import argparse
import json
import pickle
from datetime import time
import numpy as np

import Sklearn_model_utils as model_util

#Global settings
np.set_printoptions(precision=3)

#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)


def train_model_for_evaluation(data_input_path="04_Model" + "/" + "prepared_input.pickle"):
    # Get data
    # Load file paths
    print("load inputs: ", data_input_path)
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
    results_run2_file_path = prepared_data['paths']['svm_run2_result_filename']
    svm_pipe_first_selection = prepared_data['paths']['svm_pipe_first_selection']
    svm_pipe_final_selection = prepared_data['paths']['svm_pipe_final_selection']
    svm_external_parameters_filename = prepared_data['paths']['svm_external_parameters_filename']
    model_directory = prepared_data['paths']['model_directory']
    model_name = prepared_data['paths']['dataset_name']

    figure_path_prefix = model_directory + '/images/' + model_name

    # Load model external parameters
    with open(svm_external_parameters_filename, 'r') as fp:
        external_params = json.load(fp)

    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: ", pr_threshold)

    # Load model
    r = open(svm_pipe_final_selection, "rb")
    model_pipe = pickle.load(r)
    model_pipe['svm'].probability = True
    print("")
    print("Original final pipe: ", model_pipe)
    number_of_samples = 1000000

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    clf = model_pipe.fit(X_train, y_train)
    t_end = time.time() - t
    print("Training took {0:.2f}s".format(t_end))

    print("Predict training data")
    y_train_pred = clf.predict(X_train.values)
    y_train_pred_scores = clf.decision_function(X_train.values)
    y_train_pred_proba = clf.predict_proba(X_train.values)

    print("Predict test data")
    y_test_pred = clf.predict(X_test.values)
    y_test_pred_proba = clf.predict_proba(X_test.values)
    y_test_pred_scores = clf.decision_function(X_test.values)

    if len(y_classes) > 2:
        y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)  # (y_train_pred_scores>=pr_threshold).astype('int')
        y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)  # (y_test_pred_scores>=pr_threshold).astype('int')
        print("This is a binarized problem. Apply optimal threshold to precision/recall.")
    else:
        y_train_pred_adjust = y_train_pred
        y_test_pred_adjust = y_test_pred
        print("This is a multi class problem. No adjustment of scores are made.")



def main():
    train_model_for_evaluation()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.5 - Execute narrow incremental search for SVM')
    parser.add_argument("-exe", '--execute_narrow', default=True,
                        help='Execute narrow training', required=False)
    #parser.add_argument("-d", '--data_path', default="04_Model/prepared_input.pickle",
    #                    help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")



    print("=== Program end ===")