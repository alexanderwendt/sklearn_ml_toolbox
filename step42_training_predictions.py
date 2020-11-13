import argparse
import pickle #Save data
import execution_utils as step40

from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_input(paths_path = "config/paths.pickle"):
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
    paths, model, train, test = step40.load_training_files(paths_path)

    #f = open(input_path,"rb")
    #prepared_data = pickle.load(f)
    #print("Loaded data: ", prepared_data)

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']
    y_classes = train['label_map']
    scorers = model['scorers']
    refit_scorer_name = model['refit_scorer_name']
    scorer = scorers[refit_scorer_name]

    return X_train, y_train, X_test, y_test, y_classes, scorer


def run_training_estimation(X_train, y_train, X_test, y_test, scorer):
    '''
    Run estimation of scorer (default f1) and duration dependent of subset size of input data

    :args:
        X_train: Training data
        y_train: Training labels as numbers
        X_test: Test data
        y_test: Test labels as numbers
        scorer: Scorer for the evaluation, default f1
    :return:
        Nothing

    '''
    # Estimate training duration
    # run_training_estimation = True

    #if run_training_estimation==True:
        #Set test range
    test_range = list(range(100, 6500+1, 500))
    #test_range = list(range(100, 1000, 200))
    print("Test range", test_range)

    # SVM model
    # Define the model
    model_clf = SVC()
    xaxis, durations, scores = step40.estimate_training_duration(model_clf, X_train, y_train, X_test, y_test, test_range, scorer)

    # Paint figure
    plt.figure()
    plt.plot(xaxis, durations)
    plt.xlabel('Number of training examples')
    plt.ylabel('Duration [s]')
    plt.title("Training Duration")
    plt.show()

    plt.figure()
    plt.plot(xaxis, scores)
    plt.xlabel('Number of training examples')
    plt.ylabel('F1-Score on cross validation set (=the rest). Size={}'.format(X_test.shape[0]))
    plt.title("F1 Score Improvement With More Data")
    plt.show()

def run_training_predictors(data_input_path):
    '''


    '''
    X_train, y_train, X_test, y_test, y_classes, scorer = load_input(data_input_path)

    #Baseline test
    baseline_results = step40.execute_baseline_classifier(X_train, y_train, X_test, y_test, y_classes, scorer)
    print("Baseline results=", baseline_results)

    run_training_estimation(X_train, y_train, X_test, y_test, scorer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.1 - Prepare data for machine learning algorithms')
    parser.add_argument("-d", '--data_path', default="config/paths.pickle",
                        help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    run_training_predictors(data_input_path=args.data_path)

    print("=== Program end ===")