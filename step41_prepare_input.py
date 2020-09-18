# %% Script for testing step 4
# Import libraries
import argparse
import importlib #Reload imports for changed functions
import json

from pickle import dump #Save data

# %matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import data_visualization_functions as vis
import data_visualization_functions_for_SVM as vissvm
import Sklearn_model_utils as model
from Sklearn_model_utils import Nosampler, ColumnExtractor
from IPython.core.display import display
from sklearn.model_selection import train_test_split

#Import local libraries
import step40_functions as step40
import data_handling_support_functions as sup

#Set Global settings
#Print settings as precision 3
np.set_printoptions(precision=3)

#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, f1_score, confusion_matrix

#Load skip cell kernel extension
#Source: https://stackoverflow.com/questions/26494747/simple-way-to-choose-which-cells-to-run-in-ipython-notebook-during-run-all
# #%%skip True  #skips cell
# #%%skip False #won't skip
#should_skip = True
# #%%skip $should_skip
# %load_ext skip_kernel_extension

# Load config files
#config_file_path = "config_LongTrend_Debug_Training.json"
#config_file_path = "config_LongTrend_Training.json"


def generate_default_config():
    '''
    Generate a default configuration

    args:
        Nothing
    return:
        default_config: default configuration
    '''

    #Default notebook parameters as dict
    default_config = dict()
    default_config['use_training_settings'] = True
    default_config['dataset_name'] = "omxs30_train"
    default_config['source_path'] = '01_Source/^OMX_1986-2018.csv'
    default_config['class_name'] = "LongTrend"
    #Binarize labels
    default_config['binarize_labels'] = True
    default_config['class_number'] = 1   #Class number in outcomes, which shall be the "1" class
    default_config['binary_1_label'] = "Pos. Trend"
    default_config['binary_0_label'] = "Neg. Trend"
    #Load model
    default_config['use_stored_first_run_hyperparameters'] = True

    return default_config

def load_config(config_file_path = "config_LongTrend_Debug_Training.json"):
    '''
    Load configuration or use a default configuration for testing purposes

    args:
        config_file_path: File path to config
    return:
        conf: loaded or default configuration
    '''
    if config_file_path is None:
        # Use file default or set config
        # Use default
        conf = generate_default_config()

    else:
        # A config path was given
        # Load config from path
        with open(config_file_path, 'r') as fp:
            conf = json.load(fp)

        print("Loaded notebook parameters from config file: ", config_file_path)

    print("Loaded config: ", json.dumps(conf, indent=2))

    return conf

def generate_paths(conf):
    '''
    Generate paths to save the files that we generate during the machine learning cycle. It adapt paths according to
    the configuration file, e.g. considering the dataset name in the model path

    :args:
        conf: Configuration file
    :return:
        paths: dict with paths adapted to the configuration

    '''


    # Define constants from config input
    # Constants for all notebooks in the Machine Learning Toolbox
    paths = dict()

    # Model name
    paths['dataset_name'] = conf['dataset_name']

    # Generating directories
    print("Directories")
    paths['training_data_directory'] = "02_Training_Data"
    paths['target_directory'] = paths['training_data_directory']
    print("Training data directory: ", paths['training_data_directory'])

    paths['model_directory'] = "04_Model"
    print("Model directory: ", paths['model_directory'])

    paths['results_directory'] = "05_Results"
    print("Results directory: ", paths['results_directory'])

    # Dump file name
    paths['model_input_path'] = paths['model_directory'] + "/" + "prepared_input.pickle"

    # Generating filenames for saving the files
    paths['model_features_filename'] = paths['target_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_features_for_model" + ".csv"
    paths['model_outcomes_filename'] = paths['target_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_outcomes_for_model" + ".csv"
    paths['source_filename'] = paths['target_directory'] + "/" + conf['dataset_name'] + "_source" + ".csv"
    #Modified labels
    paths['model_labels_filename'] = paths['target_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_labels_for_model" + ".csv"
    #Columns for feature selection
    paths['selected_feature_columns_filename'] = paths['target_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_selected_feature_columns.csv"

    #Model specifics
    paths['svm_evaluated_model_filename'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_evaluated_model" + ".sav"
    paths['svm_final_model_filename'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_final_model" + ".sav"
    paths['svm_external_parameters_filename'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model_ext_parameters" + ".json"
    #paths['svm_default_hyper_parameters_filename'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model_hyper_parameters" + ".json"

    paths['svm_pipe_first_selection'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + '_pipe_run1_selection.pkl'
    paths['svm_pipe_final_selection'] = paths['model_directory'] + "/" + conf['dataset_name'] + "_" + conf[
        'class_name'] + '_pipe_run2_final.pkl'

    #Results
    paths['svm_run1_result_filename'] = paths['results_directory'] + "/" + conf['dataset_name'] + "_" + conf['class_name'] + '_results_run1.pkl'
    paths['svm_run2_result_filename'] = paths['results_directory'] + "/" + conf['dataset_name'] + "_" + conf[
        'class_name'] + '_results_run2.pkl'

    print("=== Paths ===")
    print("Used file paths: ", paths)
    #print("Prepared Features: ", model_features_filename)
    #print("Prepared Outcome: ", model_outcomes_filename)
    #print("Original source: ", source_filename)
    #print("Labels for the model: ", model_labels_filename)
    #print("Selected feature columns: ", selected_feature_columns_filename)

    #print("Model to save/load: ", svm_model_filename)
    #print("External parameters to load: ", svm_external_parameters_filename)
    #print("Default hyper parameters to load: ", svm_default_hyper_parameters_filename)
    #print("Saved pipe from for discrete variables: ", svm_pipe_first_selection)

    #Hyperparameter optimization methods
    # Skip the first hyper parameter search and load values from files instead
    #skip_first_run_hyperparameter_optimization = conf['use_stored_first_run_hyperparameters']

    return paths #skip_first_run_hyperparameter_optimization

def load_files(paths):
    '''
    Load files

    :args:
        paths: Dictionary of paths
    :return:
        df_X: Training data with features as dataframe X
        df_y: Labels y as dataframe
        y_classes: dict of class labels and assigned integers
        df_feature_columns: columns of column names for feature selection e.g. through lasso. The type is dataframe
    '''
    #=== Load features from X ===#
    df_X = pd.read_csv(paths['model_features_filename'], delimiter=';').set_index('id')
    print("Loaded feature names for X={}".format(df_X.columns))
    print("X. Shape={}".format(df_X.shape))
    print("Indexes of X={}".format(df_X.index.shape))

    #=== Load y values ===#
    df_y = pd.read_csv(paths['model_outcomes_filename'], delimiter=';').set_index('id')
    y = df_y.values.flatten()
    print("y. Shape={}".format(y.shape))

    #=== Load classes ===#
    df_y_classes = pd.read_csv(paths['model_labels_filename'], delimiter=';', header=None)
    y_classes_source = sup.inverse_dict(df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1])
    print("Loaded classes into dictionary: {}".format(y_classes_source))

    #=== Load list of feature columns ===#
    df_feature_columns = pd.read_csv(paths['selected_feature_columns_filename'], delimiter=';')
    print("Selected features: {}".format(paths['selected_feature_columns_filename']))
    display(df_feature_columns)

    y_classes = get_class_information(y, y_classes_source)

    return df_X, y, y_classes, df_feature_columns


def get_class_information(y, y_classes_source):
    '''
    Get class information and remove unused classes.

    :args:
        y: y values for each feature x
        y_classes_source: class labels
    :return:
        y_classes: class to number assignment as dictionary

    '''

    a, b = np.unique(y, return_counts=True)
    vfunc = np.vectorize(lambda x: y_classes_source[x])  # Vectorize a function to process a whole array
    print("For the classes with int {} and names {} are the counts {}".format(a, vfunc(a), b))
    y_classes = {}
    for i in a:
        y_classes[i] = y_classes_source[i]
    print("The following classes remain", y_classes)

    # Check if y is binarized
    if len(y_classes) == 2 and np.max(list(y_classes.keys())) == 1:
        is_multiclass = False
        print("Classes are binarized, 2 classes.")
    else:
        is_multiclass = True
        print(
            "Classes are not binarized, {} classes with values {}. For a binarized class, the values should be [0, 1].".
            format(len(y_classes), list(y_classes.keys())))

    return y_classes


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
    selected_features = [sup.getListFromColumn(df_feature_columns, df_X, i) for i in range(0, df_feature_columns.shape[1])]
    feature_dict = dict(zip(df_feature_columns.columns, selected_features))

    return feature_dict

def create_prepared_data(df_X, y, y_classes, df_feature_columns, paths):
    '''
    Create a dict with prepared data like models, training data and paths. This object can then be put into a pickle structure

    :args:
        df_X: matrix with features, X as dataframe
        y: label numbers for X
        y_classes: label to class number as dictionary
        df_feature_columns: column number for each feature selection method as dataframe
        paths:
    :return:
        prepared_data: input data for a machine learning algorithm as dataframe

    '''

    #Split test and training sets
    ### WARNING: Data is not shuffled for this example ####
    #X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state=0, test_size=0.2, shuffle=True, stratify = y) #cross validation size 20
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state=0, test_size=0.2, shuffle=False) #cross validation size 20
    print("Total number of samples: {}. X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(df_X.shape[0], X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    #
    #Create scorers to be used in GridSearchCV and RandomizedSearchCV

    #average_method = 'micro' #Add all FP1..FPn together and create Precision or recall from them. It is like weighted classes
    average_method = 'macro' #Calculate Precision1...Precisionn and Recall1...Recalln separately and average. It is good to increase
    #the weight of smaller classes

    scorers = {
        'precision_score': make_scorer(precision_score, average=average_method),
        'recall_score': make_scorer(recall_score, average=average_method),
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, average=average_method)
    }

    refit_scorer_name = list(scorers.keys())[3]
    print("Refit scorer name: ", refit_scorer_name)

    # Create feature dictionary
    feature_dict = create_feature_dict(df_feature_columns, df_X)

    # Dump prepared data
    prepared_data = dict()
    prepared_data['X_train'] = X_train
    prepared_data['y_train'] = y_train
    prepared_data['X_test'] =  X_test
    prepared_data['y_test'] = y_test
    prepared_data['y_classes'] = y_classes
    prepared_data['scorers'] = scorers
    prepared_data['refit_scorer_name'] = refit_scorer_name
    prepared_data['paths'] = paths
    prepared_data['selected_features'] = list(feature_dict.values()) #selected_features
    prepared_data['feature_dict'] = feature_dict

    return prepared_data

def prepare_data(config_file_path):
    '''
    Load config file, load training and other files, create a pickle

    :args:
        config_file_path: Filepath for configuration with dataset name and file paths
    :return:
        Nothing

    '''
    conf = load_config(config_file_path)
    paths = generate_paths(conf)
    df_X, y, y_classes, df_feature_columns = load_files(paths)
    prepared_data = create_prepared_data(df_X, y, y_classes, df_feature_columns, paths)

    # Save results
    dump(prepared_data, open(paths['model_input_path'], 'wb'))
    print("Stored prepared file input to: ", paths['model_input_path'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.1 - Prepare data for machine learning algorithms')
    parser.add_argument("-conf", '--config_path', default="config_LongTrend_Debug_Training.json",
                        help='Configuration file path', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    prepare_data(args.config_path)

    print("=== Program end ===")