import argparse
import json
import pickle
import time
import numpy as np

import joblib


def train_final_model(data_input_path="04_Model" + "/" + "prepared_input.pickle"):
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
    svm_final_model_filepath = prepared_data['paths']['svm_final_model_filename']
    svm_external_parameters_filename = prepared_data['paths']['svm_external_parameters_filename']
    model_directory = prepared_data['paths']['model_directory']
    model_name = prepared_data['paths']['dataset_name']

    figure_path_prefix = model_directory + '/images/' + model_name

    # Load model external parameters
    #with open(svm_external_parameters_filename, 'r') as fp:
    #    external_params = json.load(fp)

    #pr_threshold = external_params['pr_threshold']
    #print("Loaded precision/recall threshold: ", pr_threshold)

    # Load model
    r = open(svm_pipe_final_selection, "rb")
    model_pipe = pickle.load(r)
    #model_pipe['svm'].probability = True
    print("")
    print("Original final pipe: ", model_pipe)

    #Merge training and test data
    X = X_train.append(X_test)
    y = np.append(y_train, y_test)
    print("Merge training and test data from sizes train {} and test {} to all data {}".format(
        X_train.shape, X_train.shape, X.shape
    ))

    t = time.time()
    local_time = time.ctime(t)
    print("=== Start training the SVM at {} ===".format(local_time))
    clf = model_pipe.fit(X_train, y_train)
    t_end = time.time() - t
    print("Training took {0:.2f}s".format(t_end))

    print("Store model")
    print("Model to save: ", clf)

    joblib.dump(clf, svm_final_model_filepath)
    print("Saved model at location ", svm_final_model_filepath)


def main():
    train_final_model()


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