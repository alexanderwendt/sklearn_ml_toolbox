import os
import pickle

from IPython.core.display import display
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import time
import random
import sklearn_utils as modelutil

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
# from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import feature_selection

from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.ensemble import BalancedBaggingClassifier  # Does not work
from imblearn.ensemble import BalancedRandomForestClassifier  # Does not work
from imblearn.ensemble import RUSBoostClassifier  # Does not work

from sklearn.linear_model import LogisticRegression  # For InstanceHardnessThreshold
from sklearn.tree import DecisionTreeClassifier  # For Random Forest Balancer

from imblearn.pipeline import Pipeline

from scipy.stats import reciprocal
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

import data_handling_support_functions as sup
from evaluation_utils import Metrics
from filepaths import Paths


def load_labels(config):
    '''

    '''


def create_feature_dict(df_feature_columns, df_X):
    '''
    Create a dictionary of feature selection names and the column numbers.

    :args:
        df_feature_columns: column numbers for a feature selection method as dataframe
        df_X: features as dataframe
    :return:
        feature_dict: feature selection method name assigned to a list of column numbers as dictionary

    '''

    # Create a list of column indices for the selection of features
    selected_features = [sup.getListFromColumn(df_feature_columns, df_X, i) for i in
                         range(0, df_feature_columns.shape[1])]
    feature_dict = dict(zip(df_feature_columns.columns, selected_features))

    return feature_dict


def load_feature_columns(feature_col_path: str, df_X: pd.DataFrame):
    '''


    '''
    # === Load list of feature columns ===#
    df_feature_columns = pd.read_csv(feature_col_path, delimiter=';')
    print("Selected features: {}".format(feature_col_path))
    print(df_feature_columns)

    feature_dict = create_feature_dict(df_feature_columns, df_X)

    # model_data['selected_features'] = list(feature_dict.values()) #selected_features
    # model_data['feature_dict'] = feature_dict

    selected_features = list(feature_dict.values())

    return selected_features, feature_dict, df_feature_columns


def load_labels(labels_path):
    '''


    '''
    # === Load labels ===#
    df_labels = pd.read_csv(labels_path, delimiter=';', header=None)
    labels_inverse = sup.inverse_dict(df_labels.set_index(df_labels.columns[0]).to_dict()[1])
    print("Loaded classes into dictionary: {}".format(labels_inverse))

    return labels_inverse


def load_data(X_path, y_path):
    '''

    '''

    # === Load Features ===#
    df_X = pd.read_csv(X_path, sep=';').set_index('id')  # Set ID to be the data id
    print("Loaded feature names for X={}".format(df_X.columns))
    print("X. Shape={}".format(df_X.shape))

    # === Load y values ===#
    if y_path is not None:
        df_y = pd.read_csv(y_path, delimiter=';').set_index('id')
        y = df_y.values.flatten()
        print("Indexes of X={}".format(df_X.index.shape))
        print("y. Shape={}".format(y.shape))
    else:
        df_y = None
        y = None
        print("Warning: No outcome values available. It is OK for inference.")

    return df_X, df_y, y


def load_training_input_input(conf):
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
    labels_path = os.path.join(conf['Paths'].get('prepared_data_directory'), conf['Training'].get('labels_in'))

    X_train, _, y_train = load_data(X_train_path, y_train_path)
    X_val, _, y_val = load_data(X_val_path, y_val_path)

    labels = load_labels(labels_path)

    y_classes = labels  # train['label_map']

    print("Load feature columns")
    feature_columns_path = os.path.join(conf['Paths'].get('prepared_data_directory'),
                                        conf['Training'].get('selected_feature_columns_in'))
    selected_features, feature_dict, df_feature_columns = load_feature_columns(feature_columns_path, X_train)

    print("Load metrics")
    metrics = Metrics(conf)
    scorers = metrics.scorers
    refit_scorer_name = metrics.refit_scorer_name

    print("Load paths")
    paths = Paths(conf).path

    return X_train, y_train, X_val, y_val, y_classes, selected_features, feature_dict, paths, scorers, refit_scorer_name


# def load_training_data(config):
#     '''
#     Load all prepared and relevant files for trainf the model
#
#
#     '''
#     training_data_directory = config['Paths'].get("prepared_data_directory")
#     # dataset_name = conf['Common'].get("dataset_name")
#     # class_name = conf['Common'].get("class_name")
#
#     model_features_filename = os.path.join(training_data_directory, config['Training'].get('features_in'))
#     model_outcomes_filename = os.path.join(training_data_directory, config['Training'].get('outcomes_in'))
#     model_labels_filename = os.path.join(training_data_directory, config['Training'].get('labels_in'))
#
#     f = open(paths_path, "rb")
#     paths = pickle.load(f)
#     print("Loaded paths from: ", paths_path)
#
#     model_input_path = paths['model_input']
#     train_record_path = paths['train_record']
#     test_record_path = paths['test_record']
#
#     f = open(model_input_path, "rb")
#     model = pickle.load(f)
#     print("Loaded model input from: ", model_input_path)
#
#     f = open(train_record_path, "rb")
#     train = pickle.load(f)
#     print("Loaded training record: ", train_record_path)
#
#     f = open(test_record_path, "rb")
#     test = pickle.load(f)
#     print("Loaded test record: ", test_record_path)
#
#     return paths, model, train, test
#
#
# def load_inference_files(paths_path="04_Model/paths.pickle"):
#     '''
#     Load inference path
#
#
#     '''
#
#     f = open(paths_path, "rb")
#     paths = pickle.load(f)
#     print("Loaded paths from: ", paths_path)
#
#     inference_record_path = paths['inference_record']
#
#     f = open(inference_record_path, "rb")
#     inference = pickle.load(f)
#     print("Loaded test record: ", inference_record_path)
#
#     return paths, inference


def execute_baseline_classifier(X_train, y_train, X_test, y_test, y_classes, scorer):
    '''Baseline classifiers Most frequent class and stratified results '''

    # Dummy Classifier Most Frequent
    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train.values, y_train)

    # X_converted, y_converted = check_X_y(X=X_cross, y=y_cross)

    # Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_pred = dummy_majority.predict(X_test.values)

    confusion = confusion_matrix(y_test, y_dummy_pred)

    class_names = np.array(list(y_classes.keys()))

    metric_majority = dummy_majority.score(X_test,
                                           y_test)  # scorer(dummy_majority, y_test, y_dummy_pred) #f1_score(y_test, y_dummy_pred, average=f1_average_method)

    print("Accuracy of the most frequent predictor on training data: " + str(dummy_majority.score(X_train, y_train)))
    print('Most frequent class (dummy classifier)\n', confusion)
    print('F1 score={}'.format(metric_majority))

    # #### Stratified Class Prediction Results

    # Dummy classifier for stratified results, i.e. according to class distribution
    np.random.seed(0)
    dummy_majority = DummyClassifier(strategy='stratified').fit(X_train.values, y_train)
    y_dummy_pred = dummy_majority.predict(X_test.values)

    confusion = confusion_matrix(y_test, y_dummy_pred)

    metric_stratified = dummy_majority.score(X_test,
                                             y_test)  # scorer(y_test, y_dummy_pred) #f1_score(y_test, y_dummy_pred, average=f1_average_method)

    print(
        "Accuracy of the stratified (generates predictions by respecting the training set’s class distribution) predictor on training data: " + str(
            dummy_majority.score(X_train, y_train)))
    print('Stratified classes (dummy classifier)\n', confusion)
    print('F1 score={}'.format(metric_stratified))

    return {'majority': metric_majority, 'stratified': metric_stratified}


def estimate_training_duration(model_clf, X_train, y_train, X_test, y_test, sample_numbers, scorer):
    """ Estimate training duration by executing smaller sizes of training data and measuring training time and score
    Example SVM scaling: O(n_samples^2 * n_features). 5000->40s, 500000 -> 40*10000=400000
    input: sample_numbers: list(range(100, 6500+1, 500))
    input: model_clf: svm.SVC(probability=True, C=1, gamma=0.01, kernel='rbf', verbose=False)
    """
    np.random.seed(0)
    durations = []
    scores = []
    for i in sample_numbers:
        # Set the number of samples fr
        numberOfSamples = i

        if X_train.shape[0] > numberOfSamples:
            _, X_train_subset, _, y_train_subset = train_test_split(X_train, y_train, random_state=0,
                                                                    test_size=numberOfSamples / X_train.shape[0],
                                                                    shuffle=True, stratify=y_train)
        else:
            X_train_subset = X_train
            y_train_subset = y_train
            print("No change of data. Size of X_train={}. Current size={}".format(X_train.shape[0], i))

        t = time.time()
        # local_time = time.ctime(t)
        # print("Start training the SVM at ", local_time)
        model_clf.fit(X_train_subset.values, y_train_subset)
        elapsed = time.time() - t
        durations.append(elapsed)
        y_test_pred = model_clf.predict(X_test)
        score = model_clf.score(X_test,
                                y_test)  # f1_score(y_test, y_test_pred, average=f1_average_method)  # Micro, consider skewed data for the whole dataset
        scores.append(score)
        print("Training of {} examples; duration {}s; f1-score={}".format(i, np.round(elapsed, 3), np.round(score, 3)))

    return sample_numbers, durations, scores


def get_top_median_method(method_name, model_results, refit_scorer_name, top_share=0.10):
    '''
    inputs: name='scaler'

    '''
    # View best scaler

    # Merge results from all subsets
    merged_params_of_model_results = {}
    for d in model_results:
        merged_params_of_model_results.update(d)

    indexList = [model_results.loc[
                     model_results['param_' + method_name] == model_results['param_' + method_name].unique()[i]].iloc[0,
                 :].name for i in range(0, len(merged_params_of_model_results[method_name]))]
    print("Plot best {} values".format(method_name))
    display(model_results.loc[indexList].round(3))

    # number of results to consider
    number_results = np.int(model_results.shape[0] * top_share)
    print("The top 10% of the results are used, i.e {} samples".format(number_results))
    hist_label = model_results['param_' + method_name][0:number_results]  # .apply(str).apply(lambda x: x[:20])
    source = hist_label.value_counts() / number_results  #

    import data_visualization_functions as vis

    median_values = vis.get_median_values_from_distributions(method_name, merged_params_of_model_results[method_name],
                                                             model_results, refit_scorer_name)

    return median_values, source


def run_basic_svm(X_train, y_train, selected_features, scorers, refit_scorer_name, subset_share=0.1, n_splits=5,
                  parameters=None):
    '''Run an extensive grid search over all parameters to find the best parameters for SVM Classifier.
    The search shall be done only with a subset of the data. Default subset is 0.1. Input is training and test data.

    subset_share=0.1

    '''

    # Create a subset to train on
    print("[Step 1]: Create a data subset")
    subset_min = 300  # Minimal subset is 100 samples.

    if subset_share * X_train.shape[0] < subset_min:
        number_of_samples = subset_min
        print("minimal number of samples used: ", number_of_samples)
    else:
        number_of_samples = subset_share * X_train.shape[0]

    X_train_subset, y_train_subset = modelutil.extract_data_subset(X_train, y_train, number_of_samples)
    print("Got subset sizes X train: {} and y train: {}".format(X_train_subset.shape, y_train_subset.shape))

    print("[Step 2]: Define test parameters")
    if parameters is None:  # If no parameters have been defined, then do full definition
        # Guides used from
        # https://www.kaggle.com/evanmiller/pipelines-gridsearch-awesome-ml-pipelines
        # Main set of parameters for the grid search run 1: Select scaler, sampler and kernel for the problem
        test_scaler = [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()]
        test_sampling = [modelutil.Nosampler(),
                         ClusterCentroids(),
                         RandomUnderSampler(),
                         #NearMiss(version=1),
                         #EditedNearestNeighbours(),
                         AllKNN(),
                         #CondensedNearestNeighbour(random_state=0),
                         #InstanceHardnessThreshold(random_state=0,
                         #                          estimator=LogisticRegression(solver='lbfgs', multi_class='auto')),
                         SMOTE(),
                         SMOTEENN(),
                         SMOTETomek(),
                         ADASYN()]
        test_C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

        # gamma default parameters
        param_scale = 1 / (X_train.shape[1] * np.mean(X_train.var()))

        parameters = [
            {
                'scaler': test_scaler,
                'sampling': test_sampling,
                'feat__cols': selected_features,
                'svm__C': test_C,  # default C=1
                'svm__kernel': ['linear', 'sigmoid']
            },
            {
                'scaler': test_scaler,
                'sampling': test_sampling,
                'feat__cols': selected_features,
                'svm__C': test_C,  # default C=1
                'svm__kernel': ['poly'],
                'svm__degree': [2, 3]  # Only relevant for poly
            },
            {
                'scaler': test_scaler,
                'sampling': test_sampling,
                'feat__cols': selected_features,
                'svm__C': test_C,  # default C=1
                'svm__kernel': ['rbf'],
                'svm__gamma': [param_scale, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
                # Only relevant in rbf, default='auto'=1/n_features
            }]

        # If no missing values, only one imputer strategy shall be used
        if X_train.isna().sum().sum() > 0:
            parameters['imputer__strategy'] = ['mean', 'median', 'most_frequent']
            print("Missing values used. Test different imputer strategies")
        else:
            print("No missing values. No imputer necessary")

        print("Selected Parameters: ", parameters)
    else:
        print("Parameters defined in the input: ", parameters)

    # Main pipeline for the grid search
    pipe_run1 = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', StandardScaler()),
        ('sampling', modelutil.Nosampler()),
        ('feat', modelutil.ColumnExtractor(cols=None)),
        ('svm', SVC())
    ])

    print("Pipeline: ", pipe_run1)

    print("Stratified KFold={} used.".format(n_splits))
    skf = StratifiedKFold(n_splits=n_splits)

    pipe_run1 = pipe_run1
    params_run1 = parameters  # params_debug #params_run1
    grid_search_run1 = GridSearchCV(pipe_run1, params_run1, verbose=1, cv=skf, scoring=scorers, refit=refit_scorer_name,
                                    return_train_score=True, iid=True, n_jobs=-1).fit(X_train_subset, y_train_subset)

    results_run1 = modelutil.generate_result_table(grid_search_run1, params_run1, refit_scorer_name)
    print("Result size=", results_run1.shape)
    print("Number of NaN results: {}. Replace them with 0".format(
        np.sum(results_run1['mean_test_' + refit_scorer_name].isna())))

    return grid_search_run1, params_run1, pipe_run1, results_run1


def get_continuous_parameter_range_for_SVM_based_on_kernel(pipe_run_best_first_selection):
    '''
    From the selected pipe, extract the range of continuous parameters to test on based on SVM kernel

    :args:
        :pipe_run_best_first_selection: Pipe for the best selection of variables for SVM

    :return:
        :parameter_svm: Continuous parameter range for continuous variables

    '''
    best_kernel = str(pipe_run_best_first_selection['svm'].get_params()['kernel']).strip()
    print("Best kernel", best_kernel)
    display(pipe_run_best_first_selection)
    # Define pipeline, which is constant for all tests
    pipe_run_random = pipe_run_best_first_selection  # Use the best pipe from the best run
    # Main set of parameters for the grid search run 2: Select solver parameter
    # Initial parameters
    if best_kernel == 'rbf':
        params_run2 = {
            'svm__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
            'svm__gamma':
                [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
        }
    elif best_kernel == 'linear' or best_kernel == 'sigmoid':
        params_run2 = {
            'svm__C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
            'svm__gamma': [1e0, 1.01e0]
        }
    elif best_kernel == 'poly':
        params_run2 = {
            'svm__C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
            'svm__gamma': [1e0, 1.01e0]
        }
    else:
        raise Exception('wrong kernel:{}'.format(best_kernel))
    print("Parameters for kernel {} are {}".format(best_kernel, params_run2))
    # Get limits of the best values and focus in this area
    param_svm_C_minmax = pd.Series(
        data=[np.min(params_run2['svm__C']),
              np.max(params_run2['svm__C'])],
        index=['min', 'max'],
        name='param_svm__C')
    param_svm_gamma_minmax = pd.Series(data=[
        np.min(params_run2['svm__gamma']),
        np.max(params_run2['svm__gamma'])],
        index=['min', 'max'],
        name='param_svm__gamma')
    parameter_svm = pd.DataFrame([param_svm_C_minmax, param_svm_gamma_minmax])
    print("The initial parameters are in this area")
    display(parameter_svm)

    return parameter_svm


def generate_parameter_limits_for_SVM(results, plot_best=20):
    '''
    From a result structure, extract the min. and max. values of C and gamma as series and add them to a dataframe

    :args:
        :results: Result set with measurements of gamma and C
        :plot_best: Get the best x values to extract the range from. Default value=20

    :return:
        :parameter_svm: Min and max parameter ranges from the top x results of the result data frame

    '''
    # Get limits of the best values and focus in this area
    param_svm_C_minmax = pd.Series(
        data=[np.min(results['param_svm__C'].head(plot_best)), np.max(results['param_svm__C'].head(plot_best))],
        index=['min', 'max'], name='param_svm__C')

    param_svm_gamma_minmax = pd.Series(
        data=[np.min(results['param_svm__gamma'].head(plot_best)), np.max(results['param_svm__gamma'].head(plot_best))],
        index=['min', 'max'], name='param_svm__gamma')

    parameter_svm = pd.DataFrame([param_svm_C_minmax, param_svm_gamma_minmax])

    return parameter_svm


def run_random_cv_for_SVM(X_train, y_train, parameter_svm, pipe_run, scorers, refit_scorer_name, number_of_samples=400,
                          kfolds=5,
                          n_iter_search=2000, plot_best=20):
    '''
    Execute random search cv

    :args:
        :X_train: feature dataframe X
        :y_train: ground truth dataframe y
        :parameter_svm: Variable parameter range for C and gamma
        :pipe_run:  Pipe to run
        :scorers: Scorers
        :refit_scorer_name: Refit scrorer name
        :number_of_samples: Number of samples to use from the training data. Default=400
        :kfolds: Number of folds for cross validation. Default=5
        :n_iter_search: Number of random search iterations. Default=2000
        :plot_best: Number of top results selected for narrowing the parameter range. Default=20

    :return:


    '''

    # Extract data subset to train on
    X_train_subset, y_train_subset = modelutil.extract_data_subset(X_train, y_train, number_of_samples)

    # Main set of parameters for the grid search run 2: Select solver parameter
    # Reciprocal for the logarithmic range
    params_run = {
        'svm__C': reciprocal(parameter_svm.loc['param_svm__C']['min'], parameter_svm.loc['param_svm__C']['max']),
        'svm__gamma': reciprocal(parameter_svm.loc['param_svm__gamma']['min'],
                                 parameter_svm.loc['param_svm__gamma']['max'])
    }

    # K-Fold settings
    skf = StratifiedKFold(n_splits=kfolds)

    # run randomized search
    random_search_run = RandomizedSearchCV(pipe_run, param_distributions=params_run, n_jobs=-1,
                                           n_iter=n_iter_search, cv=skf, scoring=scorers,
                                           refit=refit_scorer_name, return_train_score=True,
                                           iid=True, verbose=5).fit(X_train_subset, y_train_subset)

    print("Best parameters: ", random_search_run.best_params_)
    print("Best score: {:.3f}".format(random_search_run.best_score_))

    # Create the result table
    results = modelutil.generate_result_table(random_search_run, params_run, refit_scorer_name)

    # Get limits of the best values and focus in this area
    parameter_svm = generate_parameter_limits_for_SVM(results, plot_best)
    # display(parameter_svm)

    # Display results
    display(results.round(3).head(5))
    display(parameter_svm)

    return parameter_svm, results, random_search_run
