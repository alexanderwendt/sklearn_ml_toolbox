import configparser
import json
import os
import random
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


def inverse_dict(dictionary):
    '''
    Inverse a dictionary

    :args:
        dictionary: Dictionary [k, V]
    :return:
        dictionary_reverse: Inverse dictionary [V, k]
    '''

    dictionary_reverse = {}
    for k, v in dictionary.items():
        dictionary_reverse[v] = k
    return dictionary_reverse

def get_random_data_subset_index(numberOfSamples, X):
    '''
    Get a random subset of data from a set

    :args:
        numberOfSamples: Number of samples
        X: Training data
    :return:
        X_index_subset: Random subset of the data

    '''

    np.random.seed(0)
    if X.shape[0] > numberOfSamples:
        X_index_subset = random.sample(list(range(0, X.shape[0], 1)), k=numberOfSamples)
        print("Cutting the data to ", numberOfSamples)
    else:
        X_index_subset = list(range(0, X.shape[0], 1))
        print("No change of data. Size remains ", X.shape[0])
    print("Created a training subset")
    
    return X_index_subset

def is_multi_class(y_classes):
    '''
    Check if y is binarized

    :args:
        y_classes: List of class numbers
    :return:
        is_multiclass: True if a class is multiclass, False if binarized

    '''

    if len(y_classes) == 2 and np.max(list(y_classes.keys())) == 1:
        is_multiclass = False
        print("Classes are binarized, 2 classes.")
    else:
        is_multiclass = True
        print("Classes are not binarized, {} classes with values {}. "
              "For a binarized class, the values should be [0, 1].".format(len(y_classes), list(y_classes.keys())))
    return is_multiclass

def getListFromColumn(df, df_source, col_number):
    '''
    Get a list of column indices from a source


    '''
    #Get list of column names from a column number
    col_names = list(df[df.columns[col_number]].dropna().values)
    #Get list of column indices in the other data frame.
    col_index = [i for i, col in enumerate(df_source) if col in col_names]

    return col_index

def get_unique_values_from_list_of_dicts(key, list_of_dicts, is_item_string=True):
    '''
        Get all unique values from a list of dictionaries for a certain key

    :args:
        key: key in the dicts
        list_of_dicts: list of dictionaries
        is_item_string: If true, then all values are converted to string and then compared. If False, object id
        is compared

    :return: list of unique values
    '''

    # Get all values from all keys scaler in a list
    sublist = [x[key] for x in list_of_dicts] # Get a list of lists with all values from all keys
    flatten = lambda l: [item for sublist in l for item in sublist]  # Lambda flatten function
    flattended_list = flatten(sublist) #Flatten the lists of lists

    if is_item_string==True: #Make string
        elements = [str(element) for element in flattended_list]
    else:
        elements = flattended_list
    unique_list = list(set(elements)) #Get unique values of list by converting it into a set and then to list

    # Replace strings with their original values
    unique_list_instance = list()
    for element_string in unique_list:
        for element in flattended_list:
            if element_string == str(element):
                unique_list_instance.append(element)
                break

    return unique_list_instance

def get_median_values_from_distributions(method_name, unique_param_values, model_results, refit_scorer_name):
    '''Extract the median values from a list of distributions

    :args:
        :method_name: Parameter name, e.g. scaler
        :unique_param_values: list of unique parameter values
        :model_results: grid search results
        :refit_scorer_name: refit scorer name for the end scores

    :return: dict of median values for each unique parameter value

    '''
    median_result = dict()
    for i in unique_param_values:
        p0 = model_results[model_results['param_' + method_name].astype(str) == str(i)]['mean_test_' + refit_scorer_name]
        if (len(p0) > 0):
            median_hist = np.round(np.percentile(p0, 50), 3)
            median_result[i] = median_hist

    return median_result

def list_to_name(list_of_lists, list_names, result):
    '''
    In a series, replace a list of integers with a string. This is used in grid search to give a list of columns
    a certain name.

    :list_of_lists: list of lists with selected features [[list1], [list2]]. This list have the keys
    :list_names: list of names for the list of lists [name1, name2]
    :result: Input Series, where the values shall be replaced. The values in the format of a list are replaced by
    strings. This is done inplace

    :return: None

    '''

    for k, value in enumerate(result):
        indices = [i for i, x in enumerate(list_of_lists) if x == value]
        if len(indices) > 0:
            first_index = indices[0]
            result.iloc[k] = list_names[first_index]
        if k % 50 == 0:
            print("run ", k)

def replace_lists_in_grid_search_params_with_strings(selected_features, feature_dict, params_run1_copy):
    '''
    In a list of dict, which are parameters for a grid search, replace certain lists with a string. This is used to
    replace a list of selected features in the grid search

    :selected_features: list of lists with selected features [[list1], [list2]]. This list have the keys
    :feature_dict: list of names for the list of lists [name1, name2]
    :params_run1_copy: [dict(), dict()] with parameters

    :return: None

    '''
    for i, value in enumerate(params_run1_copy):
        #print(value['feat__cols'])
        for k, f in enumerate(value['feat__cols']):
            indices = [i for i, x in enumerate(selected_features) if x == f]
            if len(indices) > 0:
                first_index = indices[0]
                new_value = list(feature_dict.keys())[first_index]
                params_run1_copy[i]['feat__cols'][k] = new_value
                print("Replaced {} with {}".format(selected_features[first_index], new_value))


def load_config_bak(config_file_path):
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
        raise TypeError

    else:
        # A config path was given
        # Load config from path
        with open(config_file_path, 'r') as fp:
            conf = json.load(fp)

        print("Loaded parameters from config file: ", config_file_path)

    print("Loaded config: ", json.dumps(conf, indent=2))

    return conf

def load_config(config_file_path):
    #config_path = Path(sys.path[0], "config", "classification.cfg")
    config = configparser.ConfigParser()
    config.read(config_file_path)

    return config

def load_data_source(source_filename):
    '''


    '''
    source = pd.read_csv(source_filename, sep=';').set_index('id')  # Set ID to be the data id
    print(source.head(1))

    source = pd.read_csv(source_filename, delimiter=';').set_index('id')
    source['Date'] = pd.to_datetime(source['Date'])
    source['Date'].apply(mdates.date2num)
    print("Loaded source time graph={}".format(source.columns))
    print("X. Shape={}".format(source.shape))
    print(source.head())

    return source

def load_class_labels(labels_path):
    '''


    '''

    # === Load Class Labels ===#
    # Get classes into a dict from outcomes
    # class_labels = dict(zip(outcomes_raw[class_name].unique(), list(range(1,len(outcomes_raw[class_name].unique())+1, 1))))
    # print(class_labels)
    # Load class labels file
    df_y_classes = pd.read_csv(labels_path, delimiter=';', header=None)
    class_labels = inverse_dict(df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1])
    print("Loaded  classes from file", class_labels)
    # === Define classes manually ===#
    # class_labels = {
    #    0 : 'class1',
    #    1 : 'class2'
    # }
    print(class_labels)

    return class_labels

def load_features(conf):
    '''


    '''

    training_data_directory = conf['Paths'].get("prepared_data_directory")
    #dataset_name = conf['Common'].get("dataset_name")
    #class_name = conf['Common'].get("class_name")

    model_features_filename = os.path.join(training_data_directory, conf['Preparation'].get('features_out'))
    model_outcomes_filename = os.path.join(training_data_directory, conf['Preparation'].get('outcomes_out'))
    model_labels_filename = os.path.join(training_data_directory, conf['Preparation'].get('labels_out'))

    # === Load Features ===#
    features = pd.read_csv(model_features_filename, sep=';').set_index('id')  # Set ID to be the data id
    print(features.head(1))

    # === Load y values ===#
    df_y = pd.read_csv(model_outcomes_filename, delimiter=';').set_index('id')
    y = df_y.values.flatten()

    #=== Load classes ===#
    class_labels = load_class_labels(model_labels_filename)

    print("Loaded files to analyse")

    return features, y, df_y, class_labels

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
    selected_features = [getListFromColumn(df_feature_columns, df_X, i) for i in range(0, df_feature_columns.shape[1])]
    feature_dict = dict(zip(df_feature_columns.columns, selected_features))

    return feature_dict

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
