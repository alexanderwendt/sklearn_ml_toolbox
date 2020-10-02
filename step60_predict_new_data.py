import argparse
import json
import pickle

import joblib
import pandas as pd
import matplotlib.dates as mdates
import Sklearn_model_utils as model_util
import data_visualization_functions as vis

import step40_functions as step40

def predict_unknown_data(paths_path = "04_Model/paths.pickle"):
    # Load intermediate model, which has only been trained on training data
    # Get data
    # Load file paths
    #paths, model, train, test = step40.load_training_files(paths_path)
    paths, inference = step40.load_inference_files(paths_path)

    #X_train = prepared_data['X_train']
    #y_train = prepared_data['y_train']
    #X_test = prepared_data['X_test']
    #y_test = prepared_data['y_test']
    X_inference = inference['X']

    y_classes = inference['label_map']
    #svm_pipe_final_selection = prepared_data['paths']['svm_pipe_final_selection']
    #svm_evaluation_model_filepath = prepared_data['paths']['svm_evaluated_model_filename']
    svm_final_model_filepath = paths['svm_final_model_filename']
    svm_external_parameters_filename = paths['svm_external_parameters_filename']
    model_directory = paths['model_directory']
    model_name = paths['dataset_name']
    source_path = paths['source_path']

    figure_path_prefix = paths['results_directory'] + '/images/' + 'inference_' + model_name

    # Load model external parameters
    with open(svm_external_parameters_filename, 'r') as fp:
        external_params = json.load(fp)
    pr_threshold = external_params['pr_threshold']
    print("Loaded precision/recall threshold: {0:.2f}".format(pr_threshold))

    # Open evaluation model
    evalclf = joblib.load(svm_final_model_filepath)
    print("Loaded trained evaluation model from ", svm_final_model_filepath)
    print("Model", evalclf)

    # Make predictions
    y_inference_pred_scores = evalclf.decision_function(X_inference.values)
    #y_train_pred_proba = evalclf.predict_proba(X_train.values)
    y_inference_pred_adjust = model_util.adjusted_classes(y_inference_pred_scores, pr_threshold)
    #y_test_pred_scores = evalclf.decision_function(X_test.values)
    #y_test_pred_proba = evalclf.predict_proba(X_test.values)
    #y_test_pred_adjust = model_util.adjusted_classes(y_test_pred_scores, pr_threshold)

    # Load original data for visualization
    df_time_graph = pd.read_csv(source_path, delimiter=';').set_index('id')
    df_time_graph['Date'] = pd.to_datetime(df_time_graph['Date'])
    df_time_graph['Date'].apply(mdates.date2num)
    print("Loaded feature names for time graph={}".format(df_time_graph.columns))
    print("X. Shape={}".format(df_time_graph.shape))


    # Create a df from the y array for the visualization functions
    y_order_inference_pred = pd.DataFrame(index=X_inference.index,
                                 data=pd.Series(data=y_inference_pred_adjust, index=X_inference.index, name="y")).sort_index()

    #y_order_train_pred = pd.DataFrame(index=X_train.index,
    #                                  data=pd.Series(data=y_train_pred_adjust, index=X_train.index, name="y")).sort_index()

    #y_order_test = pd.DataFrame(index=X_test.index,
    #                            data=pd.Series(data=y_test, index=X_test.index, name="y")).sort_index()

    #y_order_test_pred = pd.DataFrame(index=X_test.index,
    #                                 data=pd.Series(data=y_test_pred_adjust, index=X_test.index, name="y")).sort_index()


    #Visualize the results
    print("Plot for inference data")
    vis.plot_three_class_graph(y_order_inference_pred['y'].values,
                               df_time_graph['Close'][y_order_inference_pred['y'].index],
                               df_time_graph['Date'][y_order_inference_pred['y'].index], 0, 0, 0,
                               ('close', 'neutral', 'positive', 'negative'),
                               save_fig_prefix=figure_path_prefix + "_train_")
    #vis.plot_two_class_graph(y_order_train, y_order_train_pred,
    #                         save_fig_prefix=figure_path_prefix + "_train_")


    #print("Plot for test data")
    #vis.plot_three_class_graph(y_order_test_pred['y'].values,
    #                           df_time_graph['Close'][y_order_test.index],
    #                           df_time_graph['Date'][y_order_test.index], 0, 0, 0,
    #                           ('close', 'neutral', 'positive', 'negative'),
    #                           save_fig_prefix=figure_path_prefix + "_test_")
    #vis.plot_two_class_graph(y_order_test, y_order_test_pred,
    #                         save_fig_prefix=figure_path_prefix + "_test_")


def main():
    predict_unknown_data()


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