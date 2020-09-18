import argparse
import json
import pickle

import joblib
import numpy as np
import time

from sklearn.metrics import precision_recall_curve

import data_visualization_functions as vis

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
    svm_pipe_final_selection = prepared_data['paths']['svm_pipe_final_selection']
    svm_evaluation_model_filepath = prepared_data['paths']['svm_evaluated_model_filename']
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
    #number_of_samples = 1000000

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

    #Reduce the number of classes only to classes that can be found in the data
    reduced_class_dict_train = model_util.reduce_classes(y_classes, y_train, y_train_pred)
    reduced_class_dict_test = model_util.reduce_classes(y_classes, y_test, y_test_pred)

    if len(y_classes) == 2:
        y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)  # (y_train_pred_scores>=pr_threshold).astype('int')
        y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)  # (y_test_pred_scores>=pr_threshold).astype('int')
        print("This is a binarized problem. Apply optimal threshold to precision/recall. Threshold=", pr_threshold)
    else:
        y_train_pred_adjust = y_train_pred
        y_test_pred_adjust = y_test_pred
        print("This is a multi class problem. No precision/recall adjustment of scores are made.")

    print("Model training finished")

    #Plot graphs
    #If binary class plot precision/recall
    # Plot the precision and the recall together with the selected value for the test set
    if len(y_classes) == 2:
        print("Plot precision recall graphs")
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_scores)
        vis.plot_precision_recall_vs_threshold(precision, recall, thresholds, pr_threshold, save_fig_prefix=figure_path_prefix + "_step46_")

    #Plot evaluation
    vis.plot_precision_recall_evaluation(y_train, y_train_pred_adjust, y_train_pred_proba, reduced_class_dict_train,
                                     save_fig_prefix=figure_path_prefix + "_step46_train_")
    vis.plot_precision_recall_evaluation(y_test, y_test_pred_adjust, y_test_pred_proba, reduced_class_dict_test ,
                                     save_fig_prefix=figure_path_prefix + "_step46_test_")
    #Plot decision boundary plot
    X_decision = X_train.values[0:1000, :]
    y_decision = y_train[0:1000]
    vis.plot_decision_boundary(X_decision, y_decision, clf, save_fig_prefix=figure_path_prefix + "_step46_test_")

    print("Visualization complete")

    print("Store model")
    print("Model to save: ", clf)

    joblib.dump(clf, svm_evaluation_model_filepath)
    print("Saved model at location ", svm_evaluation_model_filepath)




def main():
    train_model_for_evaluation()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.6 - Train evaluation model for final testing')
    #parser.add_argument("-r", '--retrain_all_data', action='store_true',
    #                    help='Set flag if retraining with all available data shall be performed after ev')
    #parser.add_argument("-d", '--data_path', default="04_Model/prepared_input.pickle",
    #                    help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    main()


    print("=== Program end ===")