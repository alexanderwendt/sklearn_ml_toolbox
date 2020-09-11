import argparse
import json
import pickle
import time
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

import data_visualization_functions as vis

import Sklearn_model_utils as model_util

#Global settings
np.set_printoptions(precision=3)

#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

def define_precision_recall_threshold(data_input_path="04_Model" + "/" + "prepared_input.pickle"):
    '''
    Load model data and training data. Check if the problem is a multiclass or single class,
    the precision/recall threshold and save it to a file.

    args:
        data_input_path: Path for pickle file with data. Optional

    return:
        optimal_threshold: Optimal precision/recall threshold
    '''

    # Get data
    # Load file paths
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

    # Check if precision recall can be applied, i.e. it is a binary problem
    if len(y_classes) > 2:
        print("The problem is a multi class problem. No precision/recall optimization will be done.")
        optimal_threshold = 0
    else:
        print("The problem is a binary class problem. Perform precision/recall analysis.")

        # Load model
        # Load saved results
        r = open(svm_pipe_final_selection, "rb")
        model_pipe = pickle.load(r)
        model_pipe['svm'].probability = True

        optimal_threshold = get_optimal_precision_recall_threshold(X_train, y_train, y_classes, model_pipe, figure_path_prefix)

    #Store optimal threshold
    # save the optimal precision/recall value to disk
    print("Save external parameters, precision recall threshold to disk")
    extern_param = {}
    extern_param['pr_threshold'] = optimal_threshold
    with open(svm_external_parameters_filename, 'w') as fp:
        json.dump(extern_param, fp)

    return optimal_threshold

def get_optimal_precision_recall_threshold(X_train_full, y_train_full, y_classes, model_pipe, figure_path_prefix):
    '''


    args:
        X_train_full: Trainingdaten X
        y_train_full: Ground truth y
        y_classes: Class dict key number, value label
        model_pipe: Pipe of the model
        figure_path_prefix: Prefix path for saving images of the graphs

    return:
        optimal_threshold: Threshold calculated to be the optimum

    '''

    # Split the training set in training and cross validation set
    ### WARNING: Data is not shuffled for this example ####
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, random_state=0, test_size=0.2,
                                                                shuffle=True, stratify=y_train_full)  # cross validation size 20
    print("Total number of samples: {}. X_trainsub: {}, X_cross: {}, y_trainsub: {}, y_cross: {}".format(
        X_train.shape[0], X_train.shape, X_val.shape, y_train.shape, y_val.shape))

    print("")
    print("Original final pipe: ", model_pipe)
    number_of_samples = 1000000

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    optclf = model_pipe.fit(X_train, y_train)

    print("Predict training data")
    y_trainsub_pred = optclf.predict(X_train.values)
    y_trainsub_pred_scores = optclf.decision_function(X_train.values)
    y_trainsub_pred_proba = optclf.predict_proba(X_train.values)

    print("Predict y_val")
    y_val_pred = optclf.predict(X_val.values)

    print("Predict probabilities and scores of validation data")
    y_val_pred_proba = optclf.predict_proba(X_val.values)
    y_val_pred_scores = optclf.decision_function(X_val.values)
    print('Model properties: ', optclf)

    reduced_class_dict = model_util.reduce_classes(y_classes, y_val, y_val_pred)

    vis.plot_precision_recall_evaluation(y_train, y_trainsub_pred, y_trainsub_pred_proba, reduced_class_dict, figure_path_prefix + "_training_data_")
    vis.plot_precision_recall_evaluation(y_val, y_val_pred, y_val_pred_proba, reduced_class_dict, figure_path_prefix + "_validation_data_")

    precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_scores)
    # custom_threshold = 0.25

    # Get the optimal threshold
    closest_zero_index = np.argmin(np.abs(precision - recall))
    optimal_threshold = thresholds[closest_zero_index]
    closest_zero_p = precision[closest_zero_index]
    closest_zero_r = recall[closest_zero_index]

    print("Optimal threshold value = {0:.2f}".format(optimal_threshold))
    y_val_pred_roc_adjusted = model_util.adjusted_classes(y_val_pred_scores, optimal_threshold)

    vis.precision_recall_threshold(y_val_pred_roc_adjusted, y_val, precision, recall, thresholds, optimal_threshold,
                                   save_fig_prefix=figure_path_prefix)
    vis.plot_precision_recall_vs_threshold(precision, recall, thresholds, optimal_threshold,
                                           save_fig_prefix=figure_path_prefix)
    print("Optimal threshold value = {0:.2f}".format(optimal_threshold))

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, auc_thresholds = roc_curve(y_val, y_val_pred_scores)
    print("AUC without P/R adjustments: ", auc(fpr, tpr))  # AUC of ROC
    vis.plot_roc_curve(fpr, tpr, label='ROC', save_fig_prefix=figure_path_prefix + "_without adjustments_")

    fpr, tpr, auc_thresholds = roc_curve(y_val, y_val_pred_roc_adjusted)
    print("AUC with P/R adjustments: ", auc(fpr, tpr))  # AUC of ROC
    vis.plot_roc_curve(fpr, tpr, label='ROC', save_fig_prefix=figure_path_prefix + "_with adjustments_")

    print("Classification report without threshold adjustment.")
    print(classification_report(y_val, y_val_pred, target_names=list(reduced_class_dict.values())))
    print("=========================================================")
    print("Classification report with threshold adjustment of {0:.4f}".format(optimal_threshold))
    print(classification_report(y_val, y_val_pred_roc_adjusted, target_names=list(reduced_class_dict.values())))

    # Summarize optimal results
    print("Optimal score threshold: {0:.2f}".format(optimal_threshold))

    return optimal_threshold

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.5 - Define precision/recall')
    #parser.add_argument("-exe", '--execute_narrow', default=True,
    #                    help='Execute narrow training', required=False)
    #parser.add_argument("-d", '--data_path', default="04_Model/prepared_input.pickle",
    #                    help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    #execute_wide_run(execute_search=args.execute_wide, data_input_path=args.data_path)

    # Execute narrow search
    #execute_narrow_search(data_input_path=args.data_path)

    # Define precision/recall
    define_precision_recall_threshold()

    print("=== Program end ===")