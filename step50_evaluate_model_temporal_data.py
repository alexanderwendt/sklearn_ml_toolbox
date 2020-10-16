import argparse
import json
import os
import pickle

import joblib
import pandas as pd
import matplotlib.dates as mdates
import Sklearn_model_utils as model_util
import data_visualization_functions as vis

import step40_functions as step40

def visualize_temporal_data(paths_path = "04_Model/paths.pickle"):
    # Load intermediate model, which has only been trained on training data
    # Get data
    # Load file paths
    paths, model, train, test = step40.load_training_files(paths_path)

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']
    y_classes = train['label_map']

    #svm_pipe_final_selection = prepared_data['paths']['svm_pipe_final_selection']
    svm_evaluation_model_filepath = paths['svm_evaluated_model_filename']
    svm_external_parameters_filename = paths['svm_external_parameters_filename']
    model_directory = paths['model_directory']
    model_name = paths['dataset_name']
    source_path = paths['source_path']
    result_directory = paths['result_directory']

    figure_path_prefix = result_directory + '/eval_images/' + model_name
    if not os.path.isdir(result_directory + '/eval_images'):
        os.mkdir(result_directory + '/eval_images')
        print("Created folder: ", result_directory + '/eval_images')

    # Load model external parameters
    with open(svm_external_parameters_filename, 'r') as fp:
        external_params = json.load(fp)
    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: {0:.2f}".format(pr_threshold))

    # Open evaluation model
    evalclf = joblib.load(svm_evaluation_model_filepath)
    print("Loaded trained evaluation model from ", svm_evaluation_model_filepath)
    print("Model", evalclf)

    # Make predictions
    y_train_pred_scores = evalclf.decision_function(X_train.values)
    #y_train_pred_proba = evalclf.predict_proba(X_train.values)
    y_train_pred_adjust = model_util.adjusted_classes(y_train_pred_scores, pr_threshold)
    y_test_pred_scores = evalclf.decision_function(X_test.values)
    #y_test_pred_proba = evalclf.predict_proba(X_test.values)
    y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)

    # Load original data for visualization
    df_time_graph = pd.read_csv(source_path, delimiter=';').set_index('id')
    df_time_graph['Date'] = pd.to_datetime(df_time_graph['Date'])
    df_time_graph['Date'].apply(mdates.date2num)
    print("Loaded feature names for time graph={}".format(df_time_graph.columns))
    print("X. Shape={}".format(df_time_graph.shape))

    # Create a df from the y array for the visualization functions
    y_order_train = pd.DataFrame(index=X_train.index,
                                 data=pd.Series(data=y_train, index=X_train.index, name="y")).sort_index()

    y_order_train_pred = pd.DataFrame(index=X_train.index,
                                      data=pd.Series(data=y_train_pred_adjust, index=X_train.index, name="y")).sort_index()

    y_order_test = pd.DataFrame(index=X_test.index,
                                data=pd.Series(data=y_test, index=X_test.index, name="y")).sort_index()

    y_order_test_pred = pd.DataFrame(index=X_test.index,
                                     data=pd.Series(data=y_test_pred_adjust, index=X_test.index, name="y")).sort_index()


    #Visualize the results
    print("Plot fpr training data")
    vis.plot_three_class_graph(y_order_train_pred['y'].values,
                               df_time_graph['Close'][y_order_train.index],
                               df_time_graph['Date'][y_order_train.index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               save_fig_prefix=figure_path_prefix + "_train_")
    #vis.plot_two_class_graph(y_order_train, y_order_train_pred,
    #                         save_fig_prefix=figure_path_prefix + "_train_")


    print("Plot for test data")
    vis.plot_three_class_graph(y_order_test_pred['y'].values,
                               df_time_graph['Close'][y_order_test.index],
                               df_time_graph['Date'][y_order_test.index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               save_fig_prefix=figure_path_prefix + "_test_")
    #vis.plot_two_class_graph(y_order_test, y_order_test_pred,
    #                         save_fig_prefix=figure_path_prefix + "_test_")


def main():
    visualize_temporal_data(paths_path=args.data_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.6 - Train evaluation model for final testing')
    #parser.add_argument("-r", '--retrain_all_data', action='store_true',
    #                    help='Set flag if retraining with all available data shall be performed after ev')
    parser.add_argument("-d", '--data_path', default="config/paths.pickle",
                        help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    main()


    print("=== Program end ===")