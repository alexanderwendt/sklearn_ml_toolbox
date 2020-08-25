# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Machine Learning Algorithm Support Vector Machine

# %% [markdown]
# In this notebook, the following happens: Data is loaded. Features are visualized. A Model is trained. Model performance is evaluated.
#
# The following things can be optimized:
# - Feature columns
# - Scaler to use
# - Algorithm itself

# %% [markdown]
# ## Parameter

# %%
#Define a config file here to have a static start. If nothing is 
#config_file_path = "config_5dTrend_Training.json"
#config_file_path = "config_LongTrend_Training.json"

config_file_path = "config_LongTrend_Debug_Training.json"
#config_file_path = None

# %%
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

# %%
#import joblib
import json

if config_file_path is None:
    #Use file default or set config
    #Use default
    conf = default_config
    #config = default_test_config
    
else:
    #A config path was given
    #Load config from path
    with open(config_file_path, 'r') as fp:
        conf = json.load(fp)
        
    print("Loaded notebook parameters from config file: ", config_file_path)
print("Loaded config: ",json.dumps(conf, indent=2)) 

# %%
# Constants for all notebooks in the Machine Learning Toolbox

print("Directories")
training_data_directory = "02_Training_Data"
target_directory = training_data_directory
print("Training data directory: ", training_data_directory)

#training_image_save_directory = training_data_directory + '/images'
#print("Training data image save directory: ", training_image_save_directory)

#test_data_directory = "03_Test_Prepared_Data"
#print("Test data directory: ", test_data_directory)

#test_image_save_directory = test_data_directory + '/images'
#print("Training data image save directory: ", test_image_save_directory)

model_directory = "04_Model"
print("Model directory: ", model_directory)

results_directory = "05_Results"

#Hyperparameter optimization methods
# Skip the first hyper parameter search and load values from files instead
skip_first_run_hyperparameter_optimization = conf['use_stored_first_run_hyperparameters']

# %%
# Generating filenames for saving the files
#features_filename = target_directory + "/" + conf['dataset_name'] + "_features" + ".csv"
model_features_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_features_for_model" + ".csv"
#outcomes_filename = target_directory + "/" + conf['dataset_name'] + "_outcomes" + ".csv"
model_outcomes_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_outcomes_for_model" + ".csv"
#labels_filename = target_directory + "/" + conf['dataset_name'] + "_labels" + ".csv"
source_filename = target_directory + "/" + conf['dataset_name'] + "_source" + ".csv"
#Modified labels
model_labels_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_labels_for_model" + ".csv"
#Columns for feature selection
selected_feature_columns_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_selected_feature_columns.csv"

#Model specifics
svm_model_filename = model_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model" + ".sav"
svm_external_parameters_filename = model_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model_ext_parameters" + ".json"
svm_default_hyper_parameters_filename = model_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_svm_model_hyper_parameters" + ".json"
svm_pipe_first_selection = model_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + '_pipe_run_best_first_selection.pkl'

#Results
svm_run1_result_filename = results_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + '_results_run1.pkl'

print("=== Paths ===")
#print("Input Features: ", features_filename)
print("Prepared Features: ", model_features_filename)
#print("Input Outcomes: ", outcomes_filename)
print("Prepared Outcome: ", model_outcomes_filename)
#print("Labels: ", labels_filename)
print("Original source: ", source_filename)
print("Labels for the model: ", model_labels_filename)
print("Selected feature columns: ", selected_feature_columns_filename)

print("Model to save/load: ", svm_model_filename)
print("External parameters to load: ", svm_external_parameters_filename)
print("Default hyper parameters to load: ", svm_default_hyper_parameters_filename)
print("Saved pipe from for discrete variables: ", svm_pipe_first_selection)

# %% [markdown]
# ## Load Training and Test Data

# %%
#Imports
# %matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib as m

import data_handling_support_functions as sup
import data_visualization_functions as vis
import data_visualization_functions_for_SVM as vissvm
import Sklearn_model_utils as model
from Sklearn_model_utils import Nosampler, ColumnExtractor
from IPython.core.display import display

#from sklearn.model_selection import train_test_split

#Global settings
np.set_printoptions(precision=3)

#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

#Load skip cell kernel extension
#Source: https://stackoverflow.com/questions/26494747/simple-way-to-choose-which-cells-to-run-in-ipython-notebook-during-run-all
# #%%skip True  #skips cell
# #%%skip False #won't skip
#should_skip = True
# #%%skip $should_skip
# %load_ext skip_kernel_extension

# %%
#=== Load features from X ===#
df_X = pd.read_csv(model_features_filename, delimiter=';').set_index('id')
print("Loaded feature names for X={}".format(df_X.columns))
print("X. Shape={}".format(df_X.shape))
print("Indexes of X={}".format(df_X.index.shape))

#=== Load y values ===#
df_y = pd.read_csv(model_outcomes_filename, delimiter=';').set_index('id')
y = df_y.values.flatten()
print("y. Shape={}".format(y.shape))

#=== Load classes ===#
df_y_classes = pd.read_csv(model_labels_filename, delimiter=';', header=None)
y_classes_source = sup.inverse_dict(df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1])
print("Loaded classes into dictionary: {}".format(y_classes_source))

#=== Load list of feature columns ===#
df_feature_columns = pd.read_csv(selected_feature_columns_filename, delimiter=';')
print("Selected features: {}".format(selected_feature_columns_filename))
display(df_feature_columns)

# %%
# Get class information and remove unused classes
a, b = np.unique(y, return_counts=True)
vfunc = np.vectorize(lambda x:y_classes_source[x]) #Vectorize a function to process a whole array
print("For the classes with int {} and names {} are the counts {}".format(a, vfunc(a), b))

y_classes = {}

for i in a:
    y_classes[i] = y_classes_source[i]
    
print("The following classes remain", y_classes)

#Check if y is binarized
if len(y_classes) == 2 and np.max(list(y_classes.keys())) == 1:
    is_multiclass = False
    print("Classes are binarized, 2 classes.")
else:
    is_multiclass = True
    print("Classes are not binarized, {} classes with values {}. For a binarized class, the values should be [0, 1].".format(len(y_classes), list(y_classes.keys())))

# %% [markdown]
# ## Split Data into Train and Cross Validation Test Data

# %%
#print("No train/test splitting necessary")
from sklearn.model_selection import train_test_split

#Split test and training sets
### WARNING: Data is not shuffled for this example ####
#X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state=0, test_size=0.2, shuffle=True, stratify = y) #cross validation size 20
X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state=0, test_size=0.2, shuffle=False) #cross validation size 20
print("Total number of samples: {}. X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(df_X.shape[0], X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# %% [markdown]
# ## Handle Missing Values in the Training Data Manually
#
# In the hyper parameter optimization, automatic methods for handling missing data are tested.

# %%
#Remove rows with too many missing values
#print("Dataset size: ", df_dig.shape)
#Drop all rows with > 50% missing values
#df_compl = df_dig[np.sum(df_dig.iloc[:].isna(), axis=1)/len(df_dig.columns[0:-1])<0.5]
#print("Dataset size: ", df_compl.shape)

# %%
#Use linear regression or knn to set the feature values

# %%
#replace NaNs with impute, mean
#meansModelFromTrainingData = df_compl.mean()

#Replace all NaN with their column means
#df_compl.fillna(meansModelFromTrainingData, inplace=True)

print("Missing data model")
print(X_train.isna().sum())

# %%
#Save the missing data model as a file for the test set
#filename = filedataresultdirectoy + "/" + filenameprefix + "_missing_value_model_.csv"
#meansModelFromTrainingData.to_csv(missingValueModelPath, sep=';', index=True, header=False)
#print("Saved the missing data model with mean values of the columns to " + missingValueModelPath)

# %% [markdown]
# # Baseline Prediction
# to see if predicted values are significant. The following classifiers are used:
# - Majority class prediction
# - Stratified class prediction

# %%
#Create scorers to be used in GridSearchCV and RandomizedSearchCV

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, f1_score, confusion_matrix

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

# %% [markdown]
# #### Most Frequent
# Implement a dummy classifier for the most frequent value

# %%
# Get class information
a, b = np.unique(y_train, return_counts=True)
vfunc = np.vectorize(lambda x:y_classes[x]) #Vectorize a function to process a whole array
print("For the classes with int {} and names {} are the counts {}".format(a, vfunc(a), b))

# %%
#Dummy Classifier Most Frequent
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_val_score

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train.values, y_train)

#X_converted, y_converted = check_X_y(X=X_cross, y=y_cross)

# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_pred = dummy_majority.predict(X_test.values)

confusion = confusion_matrix(y_test, y_dummy_pred)

class_names = np.array(list(y_classes.keys()))

print("Accuracy of the most frequent predictor on training data: " + str(dummy_majority.score(X_train, y_train)))
print('Most frequent class (dummy classifier)\n', confusion)
print('F1 score={}'.format(f1_score(y_test, y_dummy_pred, average=average_method)))

# %% [markdown]
# #### Stratified Class Prediction Results

# %%
#Dummy classifier for stratified results, i.e. according to class distribution
np.random.seed(0)
dummy_majority = DummyClassifier(strategy = 'stratified').fit(X_train.values, y_train)
y_dummy_pred = dummy_majority.predict(X_test.values)

confusion = confusion_matrix(y_test, y_dummy_pred)

print("Accuracy of the stratified (generates predictions by respecting the training set’s class distribution) predictor on training data: " + str(dummy_majority.score(X_train, y_train)))
print('Stratified classes (dummy classifier)\n', confusion)
print('F1 score={}'.format(f1_score(y_test, y_dummy_pred, average=average_method)))

# %% [markdown]
# ## Training Duration Estimation
# Purposes:
# 1. Check how well the algorithm scale with much data
# 2. Check how much data is necessary to perform a gridsearch

# %%
#Set test range
test_range = list(range(100, 6500+1, 500))
test_range

# %%
#Estimate training duration
# %matplotlib inline
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import time
import random

run_training_estimation = False

if run_training_estimation==True:
    np.random.seed(0)
    #SVM scaling: O(n_samples^2 * n_features). 5000->40s, 500000 -> 40*10000=400000

    examples = test_range
    durations=[]
    scores=[]
    for i in examples:
        #Set the number of samples fr
        numberOfSamples = i

        if X_train.shape[0] > numberOfSamples:
            _, X_train_subset, _, y_train_subset = train_test_split(X_train, y_train, random_state=0, test_size=numberOfSamples/X_train.shape[0], shuffle=True, stratify = y_train)
            #X_train_index_subset = random.sample(list(range(0, X_train_shuffled.shape[0], 1)), k=numberOfSamples)
            #print("Cutting the data to size ", numberOfSamples)
        else:
            X_train_subset = X_train #list(range(0, X_train.shape[0], 1))
            y_train_subset = y_train
            #print("No change of data")

        t=time.time()
        local_time = time.ctime(t)
        #print("Start training the SVM at ", local_time)
        optclf = svm.SVC(probability=True, C=1, gamma=0.01, kernel='rbf', verbose=False)
        optclf.fit(X_train_subset.values, y_train_subset)
        elapsed = time.time() - t
        durations.append(elapsed)
        y_test_pred = optclf.predict(X_test)
        score = f1_score(y_test, y_test_pred, average=average_method) #Micro, consider skewed data for the whole dataset
        scores.append(score)
        print("Training of {} examples; duration {}s; f1-score={}".format(i, np.round(elapsed, 3), np.round(score, 3)))

    plt.figure()
    plt.plot(examples, durations)
    plt.xlabel('Number of training examples')
    plt.ylabel('Duration [s]')
    plt.title("Training Duration")

    plt.figure()
    plt.plot(examples, scores)
    plt.xlabel('Number of training examples')
    plt.ylabel('F1-Score on cross validation set (=the rest). Size={}'.format(X_test.shape[0]))
    plt.title("F1 Score Improvement With More Data")

# %% [markdown]
# # Hyper Parameter Optimization
# gridsearch, random grid search, hyperopt, automl. Things to optimize:
# - What type of normalization
# - Feature selection possibilites
# - Different algorithms, like knn, svm, random forest
# - For each algorithm, the parameters
#
# #### Method
# 1. Run 1: Grid Search on 10% of the data to determine all discrete parameters
# 2. Run 2: Grid Search on 10% for retraining continuous parameters for the best discrete parameters. Select top 20 value ranges to narrow the range
# 3. Run 3: Random Search on 10%, 20%, 50% of the data to narrow the range with more data
# 4. Run 4: Within the narrowed range, perform tests with 10%, 50% and 100% to find the best total value

# %% [markdown]
# ## Grid Search
# Do the following steps
# 1. Select how much data shall be used in the searches. Maybe all data is not necessary
# 2. Perform a broad gridsearch to find the optimal parameter range
# 3. Perform a narrow random search within the best areas of the gridsearch to find optimal values

# %% [markdown]
# #### Custom Estimators
# Here, custom estimators are put, which are added to the search pipe.

# %% [markdown]
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# #Column extractor object
#
# class Nosampler(BaseEstimator, TransformerMixin):
#     '''The nosampler class do not do any type of sampling. It shall be used to compare with common over, under and 
#     combined samplers'''
#
#     #def __init__(self):
#         #self.cols = cols
#
#     def transform(self, X):
#         return X
#
#     def fit(self, X, y=None):
#         return self

# %% [markdown]
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# #Column extractor object
#
# class ColumnExtractor(BaseEstimator, TransformerMixin):
#     '''Column extractor method to extract selected columns from a list. This is used as a feature selector. Source
#     https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline,
#     http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/.'''
#
#     def __init__(self, cols=[0]):
#         self.cols = cols
#
#     def transform(self, X):
#         col_list = []
#         if self.cols is not None:
#             return X[:,self.cols]
#         else:
#             return X
#         #for c in self.cols:
#         #    col_list.append(X[:, c:c+1])
#         #return np.concatenate(col_list, axis=1)
#         #return X[self.cols].values
#
#     def fit(self, X, y=None):
#         return self
#
# #X_train[['MA2Norm', 'MA5Norm']]
# #colList = [0, 1]
# #col = ColumnExtractor(cols=colList).transform(X_train.values)
# #col

# %%
#def getListFromColumn(df, df_source, col_number):
#    col_names = list(df[df.columns[col_number]].dropna().values)
#    col_index = [i for i, col in enumerate(df_source) if col in col_names]
#    return col_index

selected_features = [sup.getListFromColumn(df_feature_columns, df_X, i) for i in range(0, df_feature_columns.shape[1])]
feature_dict = dict(zip(df_feature_columns.columns, selected_features))
#display(feature_dict)

#Index(pd.Series(data=list(df_X.columns))).get_loc(x)

# %% [markdown]
# ### Run 1: Range finder - Selection of Scaler, Sampler and Kernel
# All continuous and discrete values are tested in a normal area. The goal is to select the best discrete variables. We use the following method:
# - For discrete variables, select wide ranges. For continuous variables, select normal ranges
# - Train with cross validation on 10% of the data
# - Select top 10% of the grid search results
# - calculate the median result for each discrete hyperparameter
# - Select the median with the best median result.
# - Goto run 2 to retrain continuous variables with a wider range

# %% [markdown]
# #Select a random subset to optimize on
# from sklearn.model_selection import train_test_split
#
# def extract_data_subset(X_train, y_train, number_of_samples, shuffled=True):
#     '''Extract subset of a dataset with X and y. The subset size is set and if the data shall be shuffled'''
#     
#     print("Original size X: ", X_train.shape)
#     print("Original size y: ", y_train.shape)
#     if number_of_samples<X_train.shape[0]:
#         print("Quota of samples used in the optimization: {0:.2f}".format(number_of_samples/X_train.shape[0]))
#         _, X_train_subset, _, y_train_subset = train_test_split(X_train, y_train, random_state=0, test_size=number_of_samples/X_train.shape[0], shuffle=shuffled, stratify = y_train)
#     else:
#         X_train_subset = X_train
#         y_train_subset = y_train
#         
#     print("Subset size X: ", X_train_subset.shape)
#     print("Subset size y: ", y_train_subset.shape)
#     
#     return X_train_subset, y_train_subset

# %%
number_of_samples = 300
X_train_subset, y_train_subset = model.extract_data_subset(X_train, y_train, number_of_samples)

# %%
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
#from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import feature_selection

from sklearn.impute import SimpleImputer
#from sklearn.impute import IterativeImputer

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
from imblearn.ensemble import BalancedBaggingClassifier #Does not work
from imblearn.ensemble import BalancedRandomForestClassifier #Does not work
from imblearn.ensemble import RUSBoostClassifier #Does not work

from sklearn.linear_model import LogisticRegression #For InstanceHardnessThreshold
from sklearn.tree import DecisionTreeClassifier #For Random Forest Balancer

from imblearn.pipeline import Pipeline

#Guides used from
#https://www.kaggle.com/evanmiller/pipelines-gridsearch-awesome-ml-pipelines

#Old Pipeline
#scalers_to_test = [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()]
#params = {'scaler': [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()],
#        'sampling': [Nosampler(), SMOTE(), SMOTEENN()],
#        'feat__k': [5, 10, 15, 20, 40, X_train.shape[1]],,
#        'svm__kernel':('linear', 'rbf'), 
#       'svm__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
#        'svm__gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 100, 1000]
#        }
#pipe = Pipeline([
#        ('scaler', StandardScaler()),
#        ('sampling', Nosampler()),
#        ('feat', feature_selection.SelectKBest()),
#        ('svm', SVC())
#        ])

#Main set of parameters for the grid search run 1: Select scaler, sampler and kernel for the problem

#Set preconditions for run1
#params_run1 = {
#        'scaler': [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()],
#        'sampling': [None, SMOTE(), SMOTEENN(), SMOTETomek(), ADASYN()],
#        'feat__cols': selected_features,
#        'svm__kernel':('linear', 'poly', 'sigmoid', 'rbf'), 
#        'svm__C':[0.1, 1, 10], #default C=1
#        'svm__gamma': [param_scale, 0.1, 1, 10], #Only relevant in rbf, default='auto'=1/n_features
##        'svm__degree': [2, 3]
#        }

test_scaler = [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()]
test_sampling = [Nosampler(), ClusterCentroids(), RandomUnderSampler(), NearMiss(version=1), EditedNearestNeighbours(), 
                 AllKNN(), CondensedNearestNeighbour(random_state=0), 
                 InstanceHardnessThreshold(random_state=0, estimator=LogisticRegression(solver='lbfgs', multi_class='auto')),
                 SMOTE(), SMOTEENN(), SMOTETomek(), ADASYN()]
test_C = [1e-2, 1e-1, 1e0, 1e1, 1e2]

#gamma default parameters
param_scale=1/(X_train.shape[1]*np.mean(X_train.var()))

params_run1 = [
    {
        'scaler': test_scaler,
        'sampling': test_sampling,
        'feat__cols': selected_features, 
        'svm__C': test_C, #default C=1
        'svm__kernel': ['linear', 'sigmoid']
    },
    {
        'scaler': test_scaler,
        'sampling': test_sampling,
        'feat__cols': selected_features, 
        'svm__C': test_C, #default C=1
        'svm__kernel':['poly'],
        'svm__degree': [2, 3] #Only relevant for poly
    },
    {
        'scaler': test_scaler,
        'sampling': test_sampling,
        'feat__cols': selected_features,
        'svm__C': test_C, #default C=1
        'svm__kernel':['rbf'], 
        'svm__gamma': [param_scale, 1e-2, 1e-1, 1e0, 1e1, 1e2] #Only relevant in rbf, default='auto'=1/n_features
    }]

#If no missing values, only one imputer strategy shall be used
if X_train.isna().sum().sum()>0:
    params_run1['imputer__strategy'] = ['mean', 'median', 'most_frequent']
    print("Missing values used. Test different imputer strategies")
else:
    print("No missing values. No imputer necessary")

#Main set of parameters for the grid search run 2: Select solver parameter
#params_run2 = {'scaler': [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()],
#        'sampling': [Nosampler(), SMOTE(), SMOTEENN(), SMOTETomek(), ADASYN()],
#        'feat__cols': selected_features,
#        'svm__kernel':('linear', 'rbf'), 
#        'svm__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
#        'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#        }

#Minimal set of parameter to test different grid searches
#params_debug = {'scaler': [StandardScaler()],
#        'sampling': [Nosampler(), SMOTE(), SMOTEENN(), ADASYN()],
#        'feat__cols': selected_features[0:1],
#        'svm__kernel':['linear'],
#        'svm__C':[0.1, 1, 10],
#        'svm__gamma': [0.1, 1, 10],
#        }

#Main pipeline for the grid search
pipe_run1 = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', StandardScaler()),
        ('sampling', Nosampler()),
        ('feat', ColumnExtractor(cols=None)),
        ('svm', SVC())
        ])

#Main pipeline for the grid search
#pipe_run2 = Pipeline([
#        ('imp', SimpleImputer(missing_values=np.nan, strategy='mean')),
#        ('scaler', StandardScaler()),
#        ('sampling', Nosampler()),
#        ('feat', ColumnExtractor(cols=None)),
#        ('svm', SVC())
#        ])

#Test pipeline for the grid search to test different methods
#pipe_debug = Pipeline([
#        ('scaler', StandardScaler()),
#        ('sampling', Nosampler()),
#        ('feat', ColumnExtractor(cols=None)),
#        ('svm', SVC())
#        ])

skf = StratifiedKFold(n_splits=10)

pipe_run1=pipe_run1
params_run1=params_run1
gridsearch_run1 = GridSearchCV(pipe_run1, params_run1, verbose=3, cv=skf, scoring=scorers, refit=refit_scorer_name, 
                               return_train_score=True, iid=True, n_jobs=-1).fit(X_train_subset, y_train_subset)
print('Final score is: ', gridsearch_run1.score(X_test, y_test))

# %% [markdown]
# ### Run1: Visualize best results

# %% [markdown]
# #Create the result table
# def generate_result_table(gridsearch_run, params_run, refit_scorer_name):
#     '''Generate a result table from a sklearn model run.
#     gridsearch_run: the run
#     params_run: parameters for the grid search
#     refit_scorer_name: refir scorer name'''
#     
#     if isinstance(params_run, list):
#         table_parameters = ['param_' + x for x in params_run[0].keys()]
#     else:
#         table_parameters = ['param_' + x for x in params_run.keys()]
#     metric_parameters = ['mean_test_' + refit_scorer_name, 'std_test_' + refit_scorer_name]
#     result_columns = metric_parameters
#     result_columns.extend(table_parameters)
#
#     results = pd.DataFrame(gridsearch_run.cv_results_)
#     results = results.sort_values(by='mean_test_' + refit_scorer_name, ascending=False)
#     results[result_columns].round(3).head(20)
#     
#     return results[result_columns]

# %%
merged_params_run1={}
for d in params_run1:
    merged_params_run1.update(d)

results_run1 = model.generate_result_table(gridsearch_run1, merged_params_run1, refit_scorer_name)
print("Result size=", results_run1.shape)
print("Number of NaN results: {}. Replace them with 0".format(np.sum(results_run1['mean_test_' + refit_scorer_name].isna())))
#results_run1[results_run1['mean_test_' + refit_scorer_name].isna()]=0
#print("Result size after dropping NAs=", results_run1.shape)
results_run1.round(4).head(20)

# %%
#Save results
from pickle import dump

#Save results
dump(results_run1, open(svm_run1_result_filename, 'wb'))
print("Stored results of run 1 to ", svm_run1_result_filename)

# %%
#from scipy.stats import ks_2samp
#import seaborn as sns
# #%matplotlib inline
#For an identical distribution, we cannot reject the null hypothesis since the p-value is high, 41%. To reject the null 
#hypothesis, the p value shall be <5%

#def calculate_significance_matrix(name, param_values, results, alpha_limit=0.05):
#    '''Function that calculate a matrix for the significance of the inputs with the Computes the
#    Kolmogorov-Smirnov statistic on 2 samples.'''
#
#    print(param_values)
#    print(name)
#    significance_calculations=pd.DataFrame(index=param_values, columns=param_values)#
#
#    for i in significance_calculations:
#        for j in significance_calculations.columns:
#            p0 = results[results['param_' + name]==i]['mean_test_' + refit_scorer_name]
#            p1 = results[results['param_' + name]==j]['mean_test_' + refit_scorer_name]
#            if (len(p0)==0 or len(p1)==0):
#                significance_calculations[j].loc[i] = 0
#            else:
#                significance_calculations[j].loc[i] = ks_2samp(p0, p1).pvalue
#
#
#    label = list(map(str, param_values))
#    label = list(map(lambda x: x[0:20], label)) #param_values#[str(t)[:9] for t in merged_params_run1[name]]
#
#    index= label
#    cols = label
#    df = pd.DataFrame(significance_calculations.values<alpha_limit, index=index, columns=cols)
#
#    plt.Figure()
#    sns.heatmap(df, cmap='Blues', linewidths=0.5, annot=True, )
#    plt.title("Statistical Difference Significance Matrix for " + name)
#
#    return significance_calculations
#    #plt.show()


# %%
# def plotOverlayedHistorgrams(name, param_values, results):
#     '''Plot layered histograms from feature distributions'''
#     min_range = np.percentile(results['mean_test_' + refit_scorer_name], 25)  #
#
#     median_result = dict()
#
#     plt.figure(figsize=(12, 8))
#     for i in param_values:
#         #print(i)
#         p0 = results[results['param_' + name] == i]['mean_test_' + refit_scorer_name]
#         if (len(p0) > 0):
#             bins = 100
#             counts, _ = np.histogram(p0, bins=bins, range=(min_range, 1))
#             #print(counts)
#             median_hist = np.round(np.percentile(p0, 50), 3)
#             median_result[i] = median_hist
#             s = str(i).split('(', 1)[0] + ": " + str(median_hist)
#             label = str(i).split('(', 1)[0]
#             plt.hist(p0,
#                      alpha=0.5,
#                      bins=bins,
#                      range=(min_range, 1),
#                      label=label)
#             plt.vlines(median_hist, 0, np.max(counts))
#             plt.text(median_hist, np.max(counts) + 1, s, fontsize=12)
#             #print("Median for {}:{}".format(s, median_hist))
#         else:
#             print("No results for ", i)
#         #plt.hist(p1, alpha=0.5)
#     plt.legend(loc='upper left')
#     plt.title("Distribution of different {}".format(name))
#     plt.xlabel('mean_test_f1_micro_score')
#     plt.ylabel('Number of occurances')
#
#     return median_result


# %%
# Compared 3 best estimators if they are significantly better or worse

#1. Get parameter settings for each estimator
#2. Train each estimator on a training set
#3. Test each estimator on the test set

# %% [markdown]
# ### Run1: Best scaler
# Determine the best scaler

# %% [markdown]
# #Load results
# import pickle
#
# f = open(svm_pipe_first_selection,"rb")
#
# results_run1 = pickle.load(f)
# display(results_run1)

# %%
# View best scaler
name = 'scaler'

indexList=[results_run1.loc[results_run1['param_' + name]==results_run1['param_' + name].unique()[i]].iloc[0,:].name for i in range(0,len(merged_params_run1[name]))]
print("Plot best {} values".format(name))
display(results_run1.loc[indexList].round(3))

#number of results to consider
number_results = np.int(results_run1.shape[0]*0.10)
print("The top 10% of the results are used, i.e {} samples".format(number_results))
histlabel = results_run1['param_' + name][0:number_results]#.apply(str).apply(lambda x: x[:20])
source = histlabel.value_counts()/number_results
f = vis.paintBarChartForCategorical(source.index, source)

# %%
_ = vis.calculate_significance_matrix(name, merged_params_run1[name], results_run1, refit_scorer_name)

# %%
median_list = vis.plotOverlayedHistorgrams(name, merged_params_run1[name], results_run1, refit_scorer_name)

# %%
#Get the best scaler from median
best_scaler = max(median_list, key=median_list.get) #source.idxmax()
print("Best scaler: ", best_scaler)

# %% [markdown]
# ### Run1: Best Sampler

# %%
# View best sampler
name = 'sampling'
print("Plot best {} values".format(name))

#results[result_columns].loc[results.param_scaler.ne('Normalizer(copy=True, norm=\'l2\')').idxmax()]
indexList=[results_run1.loc[results_run1['param_' + name]==results_run1['param_' + name].unique()[i]].iloc[0,:].name for i in range(0,len(merged_params_run1[name]))]
display(results_run1.loc[indexList].round(3))

#number of results to consider
number_results = np.int(results_run1.shape[0]*0.10)
print("The top 10% of the results are used, i.e {} samples".format(number_results))
histlabel = results_run1['param_' + name][0:number_results]#.apply(str).apply(lambda x: x[:9])
#print("Column {} is a categorical string".format(results.columns['param_scaler']))
source = histlabel.value_counts()/number_results
f=vis.paintBarChartForCategorical(source.index, source)

# %%
_ = vis.calculate_significance_matrix(name, merged_params_run1[name], results_run1, refit_scorer_name)

# %%
median_list = vis.plotOverlayedHistorgrams(name, merged_params_run1[name], results_run1, refit_scorer_name)

# %%
#Get the best sampler
best_sampler = max(median_list, key=median_list.get) #source.idxmax()
print("Best {}: {}".format(name, best_sampler))

# %% [markdown]
# ### Run1: Best kernel

# %%
# View best kernel
name = 'svm__kernel'

kernels = ['linear', 'sigmoid', 'poly', 'rbf'] #Hack. This should be taken from the dicts
indexList=[results_run1.loc[results_run1['param_' + name]==results_run1['param_' + name].unique()[i]].iloc[0,:].name for i in range(0,len(kernels))]
display(results_run1.loc[indexList].round(3))

#number of results to consider
number_results = np.int(results_run1.shape[0]*0.10)
print("The top 10% of the results are used, i.e {} samples".format(number_results))
histlabel = results_run1['param_' + name][0:number_results]#.apply(str).apply(lambda x: x[:20])
source = histlabel.value_counts()/number_results
f=vis.paintBarChartForCategorical(source.index, source)

# %%
_ = vis.calculate_significance_matrix(name, kernels, results_run1, refit_scorer_name)

# %%
median_list = vis.plotOverlayedHistorgrams(name, kernels, results_run1, refit_scorer_name)

# %%
#Get the best kernel
best_kernel = max(median_list, key=median_list.get) #source.idxmax()
print("Best {}: {}".format(name, best_kernel))

# %% [markdown]
# ### Run 1: Best Feature Selection

# %%
#Replace the parameters in the model with the name to better see the parameter setting
#feature_dict2 = dict(zip(df_feature_columns.columns, selected_features))
#number_results = 500
number_results = np.int(results_run1.shape[0]*0.10)
print("The top 10% of the results are used, i.e {} samples".format(number_results))

#display(feature_dict2)

result_subset_run1 = results_run1.iloc[0:number_results,:].copy()

def list_to_name(list_of_lists, list_names, result):
    for k, value in enumerate(result):
        indices = [i for i, x in enumerate(list_of_lists) if x == value]
        if len(indices)>0:
            first_index = indices[0]
            result.iloc[k] = list_names[first_index]
        if k%50==0:
            print("run ", k)
    
    #for j, k in enumerate(result_subset_run1['param_feat__cols']):
    #    for i in feature_dict.keys():
    #        if result_subset_run1['param_feat__cols'].iloc[j] == feature_dict.get(i):
    #            result_subset_run1['param_feat__cols'].iloc[j] = i
    #            #print(j)

print("Convert list to name")
list_to_name(selected_features, df_feature_columns.columns, result_subset_run1['param_feat__cols'])

# %%
# View best number of features
name = 'feat__cols'

indexList=[result_subset_run1.loc[result_subset_run1['param_' + name]==result_subset_run1['param_' + name].unique()[i]].iloc[0,:].name for i in range(0,len(result_subset_run1['param_' + name].unique()))]
#ndexList
display(result_subset_run1.loc[indexList].round(3))

#number of results to consider
#number_results = 20
number_results = np.int(results_run1.shape[0]*0.10)

print("The top 10% of the results are used, i.e {} samples".format(number_results))
histlabel = result_subset_run1['param_' + name][0:number_results]
source = histlabel.value_counts()/number_results
f=vis.paintBarChartForCategorical(source.index, source)

# %% [markdown]
# The statistical significance matrix shows if distributions differ significantly from each other. If the value between two distributions are 0, they do not differ. If the value is 1, there is a significant difference between the distributions.

# %%
_ = vis.calculate_significance_matrix(name, df_feature_columns.columns, result_subset_run1, refit_scorer_name)

# %%
median_list = vis.plotOverlayedHistorgrams(name, df_feature_columns.columns, result_subset_run1, refit_scorer_name)

# %%
#Get best feature result
#Get the best kernel
best_feature_combi = max(median_list, key=median_list.get) #source.idxmax()
#print("Best {}: {}".format(name, best_feature_combi))
best_columns = feature_dict.get(best_feature_combi)

print("Best feature selection: ", best_feature_combi)#results_run1[result_columns_run1].loc[indexList]['param_feat__cols'].iloc[select_index])
print("Best column indices: ", best_columns)#feature_dict.get((results_run1[result_columns_run1].loc[indexList]['param_feat__cols'].iloc[best_feature_combi])))
print("Best column names: ", X_train.columns[best_columns])#X_train.columns[feature_dict.get((results_run1[result_columns_run1].loc[indexList]['param_feat__cols'].iloc[best_feature_combi]))])


# %%
#Get top parameters from the first gridsearch to fix them
#run1_bestparameters = gridsearch_run1.best_params_
#best_scaler = run1_bestparameters['scaler']
print("Best scaler", best_scaler)
#best_sampling = run1_bestparameters['sampling']
print("Best sampling", best_sampler)
#best_columns = run1_bestparameters['feat__cols']
print("Best selected features", best_feature_combi)
print("Columns={}".format(X_train.columns[best_columns]))
#best_kernel = run1_bestparameters['svm__kernel']
print("Best kernel", best_kernel)
#print("Best result: ", gridsearch_run1.best_score_)

# %%
from pickle import dump

#Define pipeline, which is constant for all tests
pipe_run_best_first_selection = Pipeline([
        ('scaler', best_scaler),
        ('sampling', best_sampler),
        ('feat', ColumnExtractor(cols=best_columns)),
        ('svm', SVC(kernel=best_kernel))
        ])

display(pipe_run_best_first_selection)

#Save best pipe
dump(pipe_run_best_first_selection, open(svm_pipe_first_selection, 'wb'))
print("Stored pipe_run_best_first_selection.")


# %% [markdown]
# ### Run 2: Exhaustive Hyper Parameter Selection Through Wide Grid Search of Subset
# The basic parameters have been set. Now make an exhaustive parameter search for tuning parameters. Only a few samples are used and low kfold just to find the parameter limits of C and gamma. We apply the following method
# - Use the defined continouos values
# - Create a wide range for the continuous values
# - Train with 5-fold cross validation and 10% of the data
# - Select the top 20 values
# - Extract the min and the max values

# %% [markdown]
# #Main set of parameters for the grid search run 2: Select solver parameter
# number_of_samples = 300
# kfold = 5

# %% [markdown]
# #Main set of parameters for the grid search run 2: Select solver parameter
#
# if best_kernel is 'rbf':
#     params_run2 = { 
#             'svm__C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5], 
#             'svm__gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
#             }
# elif best_kernel is 'linear'or best_kernel is 'sigmoid':
#     params_run2 = { 
#             'svm__C':[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], 
#             'svm__gamma': [1e0, 1.01e0]
#             }
# elif best_kernel is 'poly':
#         params_run2 = { 
#             'svm__C':[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], 
#             'svm__gamma': [1e0, 1.01e0]
#             }
#
# pipe_run2 = Pipeline([
#         ('scaler', best_scaler),
#         ('sampling', best_sampler),
#         ('feat', ColumnExtractor(cols=best_columns)),
#         ('svm', SVC(kernel=best_kernel))
#         ])

# %% [markdown]
# results, gridsearch_run2 = grid_search_for_results(X_train, y_train, pipe_run2, params_run2, number_of_samples=number_of_samples, kfold=kfold)
# results.round(3).head(20)

# %%
#results = generate_result_table(gridsearch_run2, params_run2, refit_scorer_name)


# %% [markdown]
# #Visualize the difference between training and test results for C and gamma
#
# plot_grid_search_validation_curve(gridsearch_run2, 'svm__C', log=True, ylim=(0.50, 1.01))
# plot_grid_search_validation_curve(gridsearch_run2, 'svm__gamma', log=True, ylim=(0.50, 1.01))

# %%
#Select variables to visualize
# indexes = [i for i, item in enumerate(gridsearch_run2.cv_results_['params'])]
#
# #Create a matrix of y=alpha, x=hidden layers
# scores = gridsearch_run2.cv_results_['mean_test_' + refit_scorer_name][indexes].reshape(len(params_run2['svm__C']), len(params_run2['svm__gamma']))
# plot_heatmap_xy(scores, params_run2, 'svm__gamma', 'svm__C', 'F1 score, kernel=rbf')

# %% [markdown]
# parameter_svm = generate_parameter_limits(results)
# print("The optimal parameters are in this area")
# display(parameter_svm)

# %% [markdown]
# ### Run 3: Randomized space search to find optimal maximum
# After the parameter area has been found, intensive random search is used within it

# %%
import pickle

r = open(svm_pipe_first_selection,"rb")

pipe_run_best_first_selection = pickle.load(r)

best_kernel=str(pipe_run_best_first_selection['svm'].get_params()['kernel']).strip()
print("Best kernel", best_kernel)

display(pipe_run_best_first_selection)

# %%
#Define pipeline, which is constant for all tests
pipe_run_random = pipe_run_best_first_selection  #Use the best pipe from the best run

#Main set of parameters for the grid search run 2: Select solver parameter
#Initial parameters
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

print("Run parameters", params_run2)

#Get limits of the best values and focus in this area
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


# %%
# %matplotlib inline

# def plot_heatmap_xy(scores, parameters, xlabel, ylabel, title, normalizeScale=False):
#     # Source of inspiration: https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
#
#     #Plot a heatmap of 2 variables
#     fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True, sharey=True)
#     #fig = plt.figure(figsize=(7,5), constrained_layout=True)
#     #ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
#     #ax=axs[0]
#
#     #ax=plt.gca()
#     colorMap = plt.cm.bone #plt.cm.gist_gray #plt.cm.hot
#
#     if normalizeScale==True:
#         im1 = plt.imshow(scores, interpolation='catrom', cmap=colorMap, vmin=0 , vmax=1)
#     else:
#         #im1 = plt.imshow(scores, interpolation='nearest', cmap=colorMap)
#         im1 = plt.imshow(scores, interpolation='catrom', origin='lower', cmap=colorMap)
#
#     levels = np.linspace(np.min(scores), np.max(scores), 20)
#
#    #contours = plt.contour(scores, 10, colors='black')
#     #contours = plt.contour(scores, 10, colors='black')
#     contours = ax.contourf(scores, levels=levels, cmap=plt.cm.bone)
#
#     ax.contour(scores, levels=levels, colors='k', linestyles='solid', alpha=1, linewidths=.5, antialiased=True)
#
#     #plt.contourf =
#     ax.clabel(contours, inline=True, fontsize=8, colors='r')
#
#     #Get best value
#     def get_Top_n_values_from_array(arr, n):
#         result = []
#         for i in range(1,n+1):
#             x = np.partition(arr.flatten(), -2)[-i]
#             r = np.where(arr == x)
#             #print(r[0][0], r[1][0])
#             value = [r[0][0], r[1][0]]
#             result.append(value)
#         return result
#
#     bestvalues = get_Top_n_values_from_array(scores, 10)
#     [plt.plot(pos[1], pos[0], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="best_value") for pos in bestvalues]
#     #best = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
#     #print(best)
#     #plt.plot(parameters[xlabel][best[1]], parameters[ylabel][best[0]], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="value1")
#     #plt.plot(3, 8, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="value1")
#
#     plt.sca(ax)
#     plt.xticks(np.arange(len(parameters[xlabel])), ['{:.1E}'.format(x) for x in parameters[xlabel]], rotation='vertical')
#     plt.yticks(np.arange(len(parameters[ylabel])), ['{:.1E}'.format(x) for x in parameters[ylabel]])
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.xlim(0, len(parameters[xlabel])-1)
#     plt.ylim(0, len(parameters[ylabel])-1)
#     ax.set_title(title)
#
#     #cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.8])
#     cbar = fig.colorbar(im1, ax=ax) #cax=axs[1])
#     plt.show()


# %% [markdown]
# #Generate Scatter plot with results
# def visualize_random_search_results(random_search, refit_scorer_name):
#     '''Generate a 2D scatter plot with results for SVM'''
#     cols = random_search.cv_results_['mean_test_' + refit_scorer_name]
#     x = random_search.cv_results_['param_svm__C']
#     y = random_search.cv_results_['param_svm__gamma']
#
#     fig = plt.figure()
#     ax = plt.gca()
#     sc = ax.scatter(x=x,
#                     y=y,
#                     s=50,
#                     c=cols,
#                     alpha=0.5,
#                     edgecolors='none',
#                     cmap=plt.cm.bone)
#     #ax.pcolormesh(x, y, cols, cmap=plt.cm.BuGn_r)
#
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_xlim([np.min(x), np.max(x)])
#     ax.set_ylim([np.min(y), np.max(y)])
#     plt.grid(True)
#     plt.colorbar(sc)
#
#     return ax

# %%
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from scipy.stats import reciprocal

def generate_parameter_limits(results, plot_best=20):
    #Get limits of the best values and focus in this area
    param_svm_C_minmax = pd.Series(data=[np.min(results['param_svm__C'].head(plot_best)), np.max(results['param_svm__C'].head(plot_best))], 
              index=['min', 'max'], name='param_svm__C')

    param_svm_gamma_minmax = pd.Series(data=[np.min(results['param_svm__gamma'].head(plot_best)), np.max(results['param_svm__gamma'].head(plot_best))], 
              index=['min', 'max'], name='param_svm__gamma')

    parameter_svm = pd.DataFrame([param_svm_C_minmax, param_svm_gamma_minmax])
    
    return parameter_svm


# %%
from scipy.stats import uniform
from scipy.stats import reciprocal
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import pandas as pd 
#from sklearn.model_selection import train_test_split
#import time

def random_cv(X_train, y_train, parameter_svm, pipe_run, refit_scorer_name, number_of_samples=400, kfolds=5, n_iter_search=2000, plot_best=20):
    '''Method for executing random search cv with graphics'''
    #Extract data subset to train on
    X_train_subset, y_train_subset = model.extract_data_subset(X_train, y_train, number_of_samples)
    
    #Main set of parameters for the grid search run 2: Select solver parameter
    #Reciprocal for the logarithmic range
    params_run = { 
            'svm__C': reciprocal(parameter_svm.loc['param_svm__C']['min'], parameter_svm.loc['param_svm__C']['max']), 
            'svm__gamma': reciprocal(parameter_svm.loc['param_svm__gamma']['min'], parameter_svm.loc['param_svm__gamma']['max'])
            }
    
    #K-Fold settings
    skf = StratifiedKFold(n_splits=kfolds)

    # run randomized search
    random_search_run = RandomizedSearchCV(pipe_run, param_distributions=params_run, n_jobs=-1,
                                       n_iter=n_iter_search, cv=skf, scoring=scorers, 
                                       refit=refit_scorer_name, return_train_score=True, 
                                           iid=True, verbose=5).fit(X_train_subset, y_train_subset)

    print("Best parameters: ", random_search_run.best_params_)
    print("Best score: {:.3f}".format(random_search_run.best_score_))
    
    #Create the result table
    results = model.generate_result_table(random_search_run, params_run, refit_scorer_name)
    #Display results
    display(results.round(3).head(5))
    
    #Plot results
    ax = vissvm.visualize_random_search_results(random_search_run, refit_scorer_name)
    #Plot best results
    [ax.plot(p[0], p[1], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="best_value") for p in zip(results['param_svm__C'].head(plot_best).values, results['param_svm__gamma'].head(plot_best).values)]    
    ax.set_ylabel("gamma")
    ax.set_xlabel("C")
    
    #Visualize the difference between training and test results for C and gamma
    #print(random_search_run)
    #plot_grid_search_validation_curve(random_search_run, 'svm__C', refit_scorer_name, log=True, ylim=(0.50, 1.01))
    #plot_grid_search_validation_curve(random_search_run, 'svm__gamma', refit_scorer_name, log=True, ylim=(0.50, 1.01))
    
    plt.show()

    #Get limits of the best values and focus in this area
    parameter_svm = generate_parameter_limits(results)
    #display(parameter_svm)
    
    return parameter_svm, results, random_search_run


# %% [markdown]
# df_cv_results = pd.DataFrame(clf.cv_results_)
# train_scores_mean = df_cv_results['mean_train_' + refit_scorer_name]
# valid_scores_mean = df_cv_results['mean_test_' + refit_scorer_name]
# train_scores_std = df_cv_results['std_train_' + refit_scorer_name]
# valid_scores_std = df_cv_results['std_test_' + refit_scorer_name]
#
# param_range = np.logspace(-6, -1, 5)
# plt.title("Validation Curve with SVM")
# plt.xlabel(r"$\gamma$")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
# plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()

# %%
# def plot_grid_search_validation_curve(grid, param_to_vary, refit_scorer_name, title='Validation Curve', ylim=None, xlim=None, log=None):
#     """Plots train and cross-validation scores from a GridSearchCV instance's
#     best params while varying one of those params."""
#     #https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results
#
#     df_cv_results = pd.DataFrame(grid.cv_results_)
#     train_scores_mean = df_cv_results['mean_train_' + refit_scorer_name]
#     valid_scores_mean = df_cv_results['mean_test_' + refit_scorer_name]
#     train_scores_std = df_cv_results['std_train_' + refit_scorer_name]
#     valid_scores_std = df_cv_results['std_test_' + refit_scorer_name]
#
#     param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
#     #print(grid.cv_results_)
#     param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
#     param_ranges_lengths = [len(pr) for pr in param_ranges]
#
#     train_scores_mean = np.array(train_scores_mean).reshape(*param_ranges_lengths)
#     valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)
#     train_scores_std = np.array(train_scores_std).reshape(*param_ranges_lengths)
#     valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)
#
#     param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))
#
#     slices = []
#     for idx, param in enumerate(grid.best_params_):
#         if (idx == param_to_vary_idx):
#             slices.append(slice(None))
#             continue
#         best_param_val = grid.best_params_[param]
#         idx_of_best_param = 0
#         if isinstance(param_ranges[idx], np.ndarray):
#             idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
#         else:
#             idx_of_best_param = param_ranges[idx].index(best_param_val)
#         slices.append(idx_of_best_param)
#
#     train_scores_mean = train_scores_mean[tuple(slices)]
#     valid_scores_mean = valid_scores_mean[tuple(slices)]
#     train_scores_std = train_scores_std[tuple(slices)]
#     valid_scores_std = valid_scores_std[tuple(slices)]
#
#     plt.figure(figsize=(5,5))
#     plt.clf()
#
#     plt.title(title)
#     plt.xlabel(param_to_vary)
#     plt.ylabel('Score')
#
#     if (ylim is None):
#         plt.ylim(0.0, 1.1)
#     else:
#         plt.ylim(*ylim)
#
#     if (not (xlim is None)):
#         plt.xlim(*xlim)
#
#     lw = 2
#
#     plot_fn = plt.plot
#     if log:
#         plot_fn = plt.semilogx
#
#     param_range = param_ranges[param_to_vary_idx]
#     #if (not isinstance(param_range[0], numbers.Number)):
#     #    param_range = [str(x) for x in param_range]
#     plot_fn(param_range, train_scores_mean, label='Training score', color='r',
#             lw=lw)
#     plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color='r', lw=lw)
#     plot_fn(param_range, valid_scores_mean, label='Cross-validation score',
#             color='b', lw=lw)
#     plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
#                      valid_scores_mean + valid_scores_std, alpha=0.1,
#                      color='b', lw=lw)
#
#     plt.legend(loc='lower right')
#     plt.grid()
#
#     plt.show()


# %%
#Iterated pipeline with increasing number of tries
sample_size = [600, 600, 600]
kfolds = [3, 3, 3]
number_of_interations = [100, 100, 20]
select_from_best = [10, 10, 10]

combined_parameters = zip(sample_size, kfolds, number_of_interations, select_from_best)

new_parameter_rand = parameter_svm #Initialize the system with the parameter borders

# %%
#Iterated pipeline with increasing number of tries
#sample_size = [400, 600, 2000, 6000]
#kfolds = [3, 3, 5, 5]
#number_of_interations = [2000, 1500, 1000, 100]
#select_from_best = [20, 20, 20, 20]

#sample_size = [300, 600, 800]
#kfolds = [3, 5]
#number_of_interations = [2000, 6000]
#select_from_best = [50, 50]



for sample_size, folds, iterations, selection in combined_parameters:
    print("Start random optimization with the following parameters: ")
    print("Sample size: ", sample_size)
    print("Number of folds: ", folds)
    print("Number of tries: ", iterations)
    print("Number of best results to select from: ", selection)
    
    new_parameter_rand, results_random_search, clf = random_cv(X_train, y_train, new_parameter_rand, pipe_run_random, 
                                                               refit_scorer_name, number_of_samples=sample_size, kfolds=folds, 
                                                               n_iter_search=iterations, plot_best=selection)
    print("Got best parameters: ")
    display(new_parameter_rand)
    print("===============================================================")

# %%
print("Best parameter limits: ")
display(new_parameter_rand)

print("Best results: ")
display(results_random_search.round(3).head(20))

# %%

# %% [markdown]
# ### Run 4: Run best parameters from the intense random search
# Make a lot of random searches and narrow it by testing less points but with more samples until only 10 is missing

# %%
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import feature_selection
from imblearn.pipeline import Pipeline

def grid_search_for_results(X_train, y_train, pipe_run, params, number_of_samples=400, kfold=5):
    '''Execute grid seach for a given pipe and params'''
    X_train_subset, y_train_subset = model.extract_data_subset(X_train, y_train, number_of_samples)

    skf = StratifiedKFold(n_splits=kfold)

    gridsearch_run = GridSearchCV(pipe_run, params, verbose=3, cv=skf, scoring=scorers, refit=refit_scorer_name, 
                   return_train_score=True, iid=True, n_jobs=-1).fit(X_train_subset, y_train_subset)
    print('Final score is: ', gridsearch_run.score(X_test, y_test))
    print('Bets estimator: ', gridsearch_run.best_estimator_)
    
    results = model.generate_result_table(gridsearch_run, params, refit_scorer_name)
    
    return results, gridsearch_run


# %%
def create_parameter_grid_for_svm(results, top_results=None):
    '''Get the top x results from the result list and create a parameter list from it'''
    
    if top_results is None:
        top_results = results.shape[0]
    
    params_new = []
    for i in range(top_results):
        new_dict = {}
        new_dict['svm__C'] = [results.iloc[i]['param_svm__C']]
        new_dict['svm__gamma'] = [results.iloc[i]['param_svm__gamma']]
        params_new.append(new_dict)
        
    return params_new


# %%
pipe_run_grid = Pipeline([
        ('scaler', best_scaler),
        ('sampling', best_sampler),
        ('feat', ColumnExtractor(cols=best_columns)),
        ('svm', SVC(kernel=best_kernel))
        ])

# %%
params_grid_iter = create_parameter_grid_for_svm(results_random_search, top_results=100)
print("Number of initial combinations", len(params_grid_iter))
display(params_grid_iter[0:10])

# %%
sample_size = [400, 1000, 3000, 6000]
kfolds = [5, 5, 5]
select_from_best = [50, 10, 3, 3] #This number must always <= previous value. We start with the top 100 values.

combined_parameters = zip(sample_size, kfolds, select_from_best)

new_parameter_rand = params_grid_iter

for sample_size, folds, selection in combined_parameters:
    print("Start random optimization with the following parameters: ")
    print("Sample size: ", sample_size)
    print("Number of folds: ", folds)
    print("Number of best results to select from: ", selection)
    
    results_deep, gridsearch_run = grid_search_for_results(X_train, y_train, pipe_run_grid, new_parameter_rand, number_of_samples=sample_size, kfold=folds)
    display(results_deep.round(3).head(5))

    print("Got best parameters: ")
    parameter_limits = generate_parameter_limits(new_parameter_rand)
    display(parameter_limits)
    
    #Plot results
    ax = vis.visualize_random_search_results(clf, refit_scorer_name)
    #Plot best results
    [ax.plot(p[0], p[1], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="best_value") for p in zip(results['param_svm__C'].head(selection).values, results['param_svm__gamma'].head(selection).values)]    
    plt.show()
    
    new_parameter_rand = create_parameter_grid_for_svm(results_deep, top_results=selection)
    print("Created new ")


# %% [markdown]
# #### Print best result, the final result for hyper parameters

# %%
#Create the result table
display(results_deep.round(3).head(1))
    
param_final = new_parameter_rand[0]
print("Hyper parameters found")
display(param_final)

# %%
#Save the hyper parameters
import json

# save the optimal precision/recall value to disk
print("Save hyper parameters to disk")
with open(svm_default_hyper_parameters_filename, 'w') as fp:
    json.dump(param_final, fp)
print("Saved hyper parameters to disk: ", svm_default_hyper_parameters_filename)

# %% [markdown]
# # Optimize Precision/Recall Threshold

# %% [markdown]
# Split the training set into 2 partitions and find the optimal precision/recall threshold from that training set. Train on the whole training dataset, first without any precision/recall adjustment. Then, if the classes are binarized, precision/recall optimization is performed.
#
# Note: This adjustment is only done on binarized classes and skipped for multi class problems.

# %%
#Default optimal precision/recall threshold
optimal_threshold = 0.0

if is_multiclass is True:
    print("The problem is a multi class problem. No precision/recall optimization will be done.")
else:
    print("The problem is a binary class problem. Perform precision/recall analysis.")

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

#Split the training set in a train_sub and cross_sub set to select the optimal precision/recall threshold

from sklearn.model_selection import train_test_split

#Split test and training sets
### WARNING: Data is not shuffled for this example ####
X_trainsub, X_cross, y_trainsub, y_cross = train_test_split(X_train, y_train, random_state=0, test_size=0.2, shuffle=True, stratify = y_train) #cross validation size 20
print("Total number of samples: {}. X_trainsub: {}, X_cross: {}, y_trainsub: {}, y_cross: {}".format(X_train.shape[0], X_trainsub.shape, X_cross.shape, y_trainsub.shape, y_cross.shape))

# %% [markdown]
# #### Load existing model and/or parameters

# %%
#Apply final parameters
#Either use the currently found parameters
print("Original final parameters: ", param_final)

#or use customized, known parameters
#param_final = {'svm__C': [35.00964852688157], 'svm__gamma': [0.1761212691035545]}
#print("New final parameters: ", param_final)

#TODO Add possibility to load parameters from file as final parameter

# %%
# Load an existing model to be used optimized with precision recall parameters
#from sklearn.externals import joblib

#loaded_model = joblib.load(model_filename)
#print("Loaded trained model from ", model_filename)
#print("Model", loaded_model)
print("Use currently trained model.")

# %% [markdown]
# #### Train new model from optimal parameters

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

# Train an SVM
from sklearn import svm
import time

number_of_samples = 1000000

display(param_final)
pipe_final = Pipeline([
        ('scaler', best_scaler),
        ('sampling', best_sampler),
        ('feat', ColumnExtractor(cols=best_columns)),
        ('svm', SVC(kernel=best_kernel, probability=True))
        ])

display(pipe_final)

t=time.time()
local_time = time.ctime(t)
print("=== Start training the SVM at {} ===".format(local_time))

results, gridsearch_pr = grid_search_for_results(X_trainsub, y_trainsub, pipe_final, param_final, number_of_samples=number_of_samples, kfold=5)

#optclf = svm.SVC(probability=True, C=0.5, gamma=1, kernel='rbf', verbose=True)
#optclf = gridsearch_run2#svm.SVC(probability=True, C=564, gamma=0.013, kernel='rbf', verbose=True)
#optclf.fit(X_train_shuffled.iloc[X_train_index_subset,:].values, y_train_shuffled[X_train_index_subset])
elapsed = time.time() - t
print("Training took {}s".format(elapsed))
print("Training finished")

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

#Select the optimal model
#If new model has been trained
optclf = gridsearch_pr.best_estimator_

#If old model is used
#optclf = loaded_model

# %% [markdown]
# #### Create Predictions

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

print("Predict training data")
y_trainsub_pred = optclf.predict(X_trainsub.values)
y_trainsub_pred_scores = optclf.decision_function(X_trainsub.values)
y_trainsub_pred_proba = optclf.predict_proba(X_trainsub.values)

print("Predict y_cross")
y_cross_pred = optclf.predict(X_cross.values)
print("Predict probabilities and scores")
y_cross_pred_proba = optclf.predict_proba(X_cross.values)
y_cross_pred_scores = optclf.decision_function(X_cross.values)
print('Model properties: ', optclf)

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

#Prepare the class list for classes, in case a class does not exist in the y_true and y_pred, which exist in y_train
a=np.unique(y_cross)
b=np.unique(y_cross_pred)
c=np.hstack((a, b))
d=np.unique(c)
existing_classes=[]
[existing_classes.append(y_classes[i]) for i in d]
existing_classes

# %% [markdown]
# #### Evaluate performance on Training data

# %%
# %matplotlib inline

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt

def plot_evaluation(y_trainsub, y_trainsub_pred, y_trainsub_pred_proba):
    #Calculate Precision, Recall, Accuracy, F1, Confusion Matrix, ROC
    np.set_printoptions(precision=3)
    
    print("Accuracy: ", accuracy_score(y_trainsub, y_trainsub_pred))
    print(classification_report(y_trainsub, y_trainsub_pred, target_names=existing_classes))

    cnf_matrix = confusion_matrix(y_trainsub, y_trainsub_pred, labels=d)
    
    # Plot non-normalized confusion matrix
    matrixSize = len(y_classes)*2
    #plt.figure(figsize=(matrixSize, matrixSize))
    _ = vis.plot_confusion_matrix_multiclass(cnf_matrix, classes=existing_classes, title='Confusion matrix with normalization', normalize=True)
    _ =skplt.metrics.plot_precision_recall_curve(np.array(y_trainsub), np.array(y_trainsub_pred_proba), figsize=(10,10))
    _ = skplt.metrics.plot_roc(np.array(y_trainsub), np.array(y_trainsub_pred_proba), figsize=(10,10))
    plt.show()


# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

plot_evaluation(y_trainsub, y_trainsub_pred, y_trainsub_pred_proba)

# %% [markdown]
# #### Evaluate performance on Test data

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

plot_evaluation(y_cross, y_cross_pred, y_cross_pred_proba)

#Calculate Precision, Recall, Accuracy, F1, Confusion Matrix, ROC
#print("Accuracy: ", accuracy_score(y_cross, y_cross_pred))
#print(classification_report(y_cross ,y_cross_pred, target_names=existing_classes))

#cnf_matrix = confusion_matrix(y_cross, y_cross_pred, labels=d)
#np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
#matrixSize = len(y_classes)*2
#plt.figure(figsize=(matrixSize, matrixSize))
#_ = vis.plot_confusion_matrix_multiclass(cnf_matrix, classes=existing_classes, title='Confusion matrix with normalization', normalize=True)
#plt.show()

#plt.figure(figsize=(12, 8))
#print("Plot ROC curve")
#_ = skplt.metrics.plot_roc(np.array(y_cross), np.array(y_cross_pred_proba))
#_ = skplt.metrics.plot_precision_recall_curve(np.array(y_cross), np.array(y_cross_pred_proba), figsize=(10,10))
#_ = skplt.metrics.plot_roc(np.array(y_cross), np.array(y_cross_pred_proba), figsize=(10,10))
#plt.show()

# %% [markdown]
# #### Precision/Recall Adjustments
# The optimum between precision and recall is calculated and automatically adjusted

# %%
# Create adjusted precision-recall curves
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


# %%
def precision_recall_threshold(y_scores, y_test, p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.fill_between(r, p, step='post', alpha=0.2, color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)
    plt.show()


# %%
#https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, optimal_threshold):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    close_default_clf = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.plot(optimal_threshold, precisions[close_default_clf], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="Optimal threshold")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.grid()
    plt.show()


# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

precision, recall, thresholds = precision_recall_curve(y_cross, y_cross_pred_scores)
#custom_threshold = 0.25

#Get the optimal threshold
closest_zero_index = np.argmin(np.abs(precision-recall))
optimal_threshold = thresholds[closest_zero_index]
closest_zero_p = precision[closest_zero_index]
closest_zero_r = recall[closest_zero_index]

print("Optimal threshold value = {0:.2f}".format(optimal_threshold))

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem
y_cross_pred_roc_adjusted = adjusted_classes(y_cross_pred_scores, optimal_threshold) #(y_cross_pred_scores>=optimal_threshold).astype('int')

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem
precision_recall_threshold(y_cross_pred_scores, y_cross, precision, recall, thresholds, optimal_threshold)
plot_precision_recall_vs_threshold(precision, recall, thresholds, optimal_threshold)
print("Optimal threshold value = {0:.2f}".format(optimal_threshold))


# %%
#https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
    plt.show()


# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

from sklearn.metrics import roc_curve, auc

fpr, tpr, auc_thresholds = roc_curve(y_cross, y_cross_pred_scores)
print("AUC without P/R adjustments: ", auc(fpr, tpr)) # AUC of ROC
plot_roc_curve(fpr, tpr, label='ROC')

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

fpr, tpr, auc_thresholds = roc_curve(y_cross, y_cross_pred_roc_adjusted)
print("AUC with P/R adjustments: ", auc(fpr, tpr)) # AUC of ROC
plot_roc_curve(fpr, tpr, label='ROC')

# %%
#precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_scores)
#closest_zero = np.argmin(np.abs(thresholds))
#closest_zero_p = precision[closest_zero]
#closest_zero_r = recall[closest_zero]

#plt.figure()
#plt.xlim([0.0, 1.01])
#plt.ylim([0.0, 1.01])
#plt.plot(precision, recall, label='Precision-Recall Curve')
#plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
#plt.xlabel('Precision', fontsize=16)
#plt.ylabel('Recall', fontsize=16)
#plt.title("Precision/Recall curve")
#plt.axes().set_aspect('equal')
#plt.gca().set_aspect('equal')
#print("closest precision={}, clostest recall={}".format(closest_zero_p, closest_zero_r))
#plt.show()

#Adjust precision
#optimalRocValue = thresholds[closest_zero]
#print("Optimal threshold=", optimalRocValue)

#Calculate new classification for cross_validation
#y_test_pred_roc_adjusted = (y_test_pred_scores>=optimal_threshold).astype('int')
#display(y_test_pred_scores)
#display(y_test_pred_roc_adjusted)

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

print("Classification report without threshold adjustment.")
print(classification_report(y_cross, y_cross_pred, target_names=existing_classes))
print("=========================================================")
print("Classification report with threshold adjustment of {0:.4f}".format(optimal_threshold))
print(classification_report(y_cross, y_cross_pred_roc_adjusted, target_names=existing_classes))

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

# Summarize optimal results
print("Optimal score threshold: {0:.2f}".format(optimal_threshold))

# %% [markdown]
# # Train the model with optimal values for Validation
# Train the model on the whole training data set nd include optimal precision-recall shift. Validate the model on the test data. This will be the final model.

# %%
# Train an SVM
from sklearn import svm
import time

number_of_samples = 1000000

display(param_final)
pipe_final = Pipeline([
        ('scaler', best_scaler),
        ('sampling', best_sampler),
        ('feat', ColumnExtractor(cols=best_columns)),
        ('svm', SVC(kernel=best_kernel, probability=True))
        ])

display(pipe_final)

t=time.time()
local_time = time.ctime(t)
print("=== Start training the SVM at {} ===".format(local_time))

results, gridsearch_final = grid_search_for_results(X_train, y_train, pipe_final, param_final, number_of_samples=number_of_samples, kfold=10)

#optclf = svm.SVC(probability=True, C=0.5, gamma=1, kernel='rbf', verbose=True)
#optclf = gridsearch_run2#svm.SVC(probability=True, C=564, gamma=0.013, kernel='rbf', verbose=True)
#optclf.fit(X_train_shuffled.iloc[X_train_index_subset,:].values, y_train_shuffled[X_train_index_subset])
elapsed = time.time() - t
print("Training took {}s".format(elapsed))
print("Training finished")

# %%
# Calculate the 
optclf = gridsearch_final.best_estimator_

print("Predict training data")
y_train_pred = optclf.predict(X_train.values)
y_train_pred_proba = optclf.predict_proba(X_train.values)
y_train_pred_scores = optclf.decision_function(X_train.values)

print("Predict y_cross")
y_test_pred = optclf.predict(X_test.values)
print("Predict probabilities and scores")
y_test_pred_proba = optclf.predict_proba(X_test.values)
y_test_pred_scores = optclf.decision_function(X_test.values)
print('Model properties: ', optclf)

# %% [markdown]
# #### Adjust output by precision/recal optimal threshold

# %%
#Adjust for the optimal value of precision/recall curve
print("Optimal value: ", optimal_threshold)
pr_threshold = optimal_threshold
#pr_threshold = 0.35
print("Selected threshold: ", pr_threshold)

#Set the optimal threshol in the parameters
extern_param_final = {}
extern_param_final['pr_threshold']=pr_threshold

# %%
#Adjust training data for precision/recall

if is_multiclass == False:
    y_train_pred_adjust = adjusted_classes(y_train_pred_scores, pr_threshold) #(y_train_pred_scores>=pr_threshold).astype('int')
    y_test_pred_adjust = adjusted_classes(y_test_pred_scores, pr_threshold) # (y_test_pred_scores>=pr_threshold).astype('int')
    print("This is a binarized problem. Apply optimal threshold to precision/recall.")
else:
    y_train_pred_adjust = y_train_pred
    y_test_pred_adjust = y_test_pred
    print("This is a multi class problem. No adjustment of scores are made.")

# %%
# %%skip $is_multiclass #skip if the problem is a multiclass problem

#Plot the precision and the recall together with the selected value for the test set
precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred_scores)
plot_precision_recall_vs_threshold(precision, recall, thresholds, pr_threshold)

# %%
#Prepare the class list for classes, in case a class does not exist in the y_true and y_pred, which exist in y_train
a=np.unique(y_test)
b=np.unique(y_test_pred)
c=np.hstack((a, b))
d=np.unique(c)
existing_classes=[]
[existing_classes.append(y_classes[i]) for i in d]
existing_classes

# %% [markdown]
# #### Evaluate performance on Training data

# %%
print("Classes: ", y_classes)

# %%
# %matplotlib inline

plot_evaluation(y_train, y_train_pred_adjust, y_train_pred_proba)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt

#Calculate Precision, Recall, Accuracy, F1, Confusion Matrix, ROC
#print("Accuracy: ", accuracy_score(y_train, y_train_pred_adjust))
#print(classification_report(y_train, y_train_pred_adjust, target_names=existing_classes))
#print(classification_report(y_train, y_train_pred_adjust, target_names=list(y_classes.values())))

#cnf_matrix = confusion_matrix(y_train, y_train_pred_adjust, labels=d)
#np.set_printoptions(precision=3)

# Plot normalized confusion matrix
#matrixSize = len(y_classes)*2
#plt.figure(figsize=(matrixSize, matrixSize))
#_ = vis.plot_confusion_matrix_multiclass(cnf_matrix, classes=existing_classes, title='Confusion matrix with normalization', normalize=True)
#_ = skplt.metrics.plot_precision_recall_curve(np.array(y_train), np.array(y_train_pred_proba), figsize=(10,10))
#_ = skplt.metrics.plot_roc(np.array(y_train), np.array(y_train_pred_proba), figsize=(10,10))
#plt.show()

# %% [markdown]
# #### Evaluate performance on Test data

# %%
#Calculate Precision, Recall, Accuracy, F1, Confusion Matrix, ROC

plot_evaluation(y_test, y_test_pred_adjust, y_test_pred_proba)

#print("Accuracy: ", accuracy_score(y_test, y_test_pred_adjust))
#print(classification_report(y_test ,y_test_pred_adjust, target_names=existing_classes))

#cnf_matrix = confusion_matrix(y_test, y_test_pred_adjust, labels=d)
#np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
#matrixSize = len(y_classes)*2
#plt.figure(figsize=(matrixSize, matrixSize))
#_ = vis.plot_confusion_matrix_multiclass(cnf_matrix, classes=existing_classes, title='Confusion matrix with normalization', normalize=True)
#plt.show()

#plt.figure(figsize=(12, 8))
#print("Plot ROC curve")
#_ = skplt.metrics.plot_roc(np.array(y_test), np.array(y_test_pred_proba), figsize=(10,10))
#_ = skplt.metrics.plot_precision_recall_curve(np.array(y_test), np.array(y_test_pred_proba), figsize=(10,10))
#plt.show()

# %% [markdown]
# #### Decision Boundary Plot

# %%
# %matplotlib inline

import numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.datasets.base import load_iris
from sklearn.manifold import TSNE
#from sklearn.linear_model.logistic import LogisticRegression

# replace the below by your data and model
#iris = load_iris()
#X,y = iris.data, iris.target
X_decision=X_train.values[0:1000,:]
y_decision=y_train[0:1000]
X_Train_embedded = TSNE(n_components=2).fit_transform(X_decision)
print(X_Train_embedded.shape)
#model = LogisticRegression().fit(X,y)
model = optclf
y_predicted = model.predict(X_decision)
# replace the above by your data and model

# create meshgrid
resolution = 100 # 100x100 background pixels
X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:,0]), np.max(X_Train_embedded[:,0])
X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:,1]), np.max(X_Train_embedded[:,1])
xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

# approximate Voronoi tesselation on resolution x resolution grid using 1-NN
background_model = KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted) 
voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
voronoiBackground = voronoiBackground.reshape((resolution, resolution))

#plot
plt.figure(figsize=(11.5, 7))
plt.contourf(xx, yy, voronoiBackground)
plt.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=y_decision)
plt.title("Decision Boundary Plot Projected")
plt.show()

# %% [markdown]
# ## Visualize Results

# %% [markdown]
# #### Viualize the results compared to the initial results

# %%
#Load original data for visualization
import matplotlib.dates as mdates
import datetime
#filename_timegraph = filedataresultdirectory + "/" + filenameprefix + "_timegraph" + ".csv"

df_timegraph = pd.read_csv(source_filename, delimiter=';').set_index('id')
df_timegraph['Date'] = pd.to_datetime(df_timegraph['Date'])
df_timegraph['Date'].apply(mdates.date2num)
print("Loaded feature names for time graph={}".format(df_timegraph.columns))
print("X. Shape={}".format(df_timegraph.shape))
display(df_timegraph.head())

# %%
#Map the X_train indices to y_train again to get the correct y.

#X_train.iloc[0:10,:]
#Create a df from the y array
y_order_train = pd.DataFrame(index=X_train.index, data=pd.Series(data=y_train, index=X_train.index, name="y")).sort_index()
#display(y_order_train[0:10])
#print(y_order_train.size)

y_order_train_pred = pd.DataFrame(index=X_train.index, data=pd.Series(data=y_train_pred_adjust, index=X_train.index, name="y")).sort_index()
#display(y_order_train_pred[0:10])
#print(y_order_train_pred.size)

y_order_test = pd.DataFrame(index=X_test.index, data=pd.Series(data=y_test, index=X_test.index, name="y")).sort_index()

y_order_test_pred = pd.DataFrame(index=X_test.index, data=pd.Series(data=y_test_pred_adjust, index=X_test.index, name="y")).sort_index()

# %%
#def amplifyForPlot(binaryArray, targetArray, distance):
#    return binaryArray * targetArray * (1-distance)

# %% [markdown]
# %matplotlib notebook
#
# def plot_three_class_graph(y_class, y_ref, y_time, offset1, offset2, offset3, legend):
#     
#     y0 = (y_class==0)*1
#     y1 = (y_class==1)*1
#     y2 = (y_class==2)*1
#     
#     plot_data_OK = amplifyForPlot(y0, y_ref, offset1)
#     plot_data_blim = amplifyForPlot(y1, y_ref, offset2)
#     plot_data_tlim = amplifyForPlot(y2, y_ref, offset3)
#     
#     # Plot test data
#     plt.figure(num=None, figsize=(11.5, 7), dpi=80, facecolor='w', edgecolor='k')
#
#     plt.plot(y_time, y_ref)
#     plt.plot(y_time, plot_data_OK, color='grey')
#     plt.plot(y_time, plot_data_blim, color='green')
#     plt.plot(y_time, plot_data_tlim, color='red')
#     plt.title("Prediction Results")
#     plt.ylim([np.min(y_ref)*0.99999, np.max(y_ref)*1.00002])
#     plt.grid()
#     plt.legend(legend)
#     plt.show()

# %%
#Present long term term data

vis.plot_three_class_graph(y_order_train_pred['y'].values, 
                       df_timegraph['Close'][y_order_train.index], 
                       df_timegraph['Date'][y_order_train.index], 0,0,0,('close', 'neutral', 'positive', 'negative'))

# %%
# %matplotlib notebook

# Plot training data
# Load graph for Borsdata


plt.figure(num=None, figsize=(11.5, 7), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(df_timegraph['Time'],df_timegraph['Close'])
#plt.plot(df_timegraph['Time'],amplifyForPlot(y, df_timegraph['Close'].values, 0.00), color='green')

plt.plot(df_timegraph['Date'][y_order_train.index],df_timegraph['Close'][y_order_train.index])
plt.plot(df_timegraph['Date'][y_order_train.index],vis.amplifyForPlot(y_order_train['y'].values, df_timegraph['Close'][y_order_train.index].values, 0.00), color='green')
plt.plot(df_timegraph['Date'][y_order_train.index],vis.amplifyForPlot(y_order_train_pred['y'].values, df_timegraph['Close'][y_order_train.index].values, 0.00), color='yellow')
plt.title("Plot results")
plt.show()

# %%
#Present long term term data
vis.plot_three_class_graph(y_order_test_pred['y'].values, 
                       df_timegraph['Close'][y_order_test.index], 
                       df_timegraph['Date'][y_order_test.index], 0,0,0,('close', 'neutral', 'positive', 'negative'))

# %%
# %matplotlib notebook

# Plot test data
plt.figure(num=None, figsize=(11.5, 7), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(df_timegraph['Time'],df_timegraph['Close'])
#plt.plot(df_timegraph['Time'],amplifyForPlot(y, df_timegraph['Close'].values, 0.00), color='green')

plt.plot(df_timegraph['Date'][X_test.index],df_timegraph['Close'][X_test.index])
plt.plot(df_timegraph['Date'][X_test.index],vis.amplifyForPlot(y_order_test['y'].values, df_timegraph['Close'][X_test.index].values, 0.00), color='green')
plt.plot(df_timegraph['Date'][X_test.index],vis.amplifyForPlot(y_order_test_pred['y'].values, df_timegraph['Close'][X_test.index].values, 0.01), color='yellow')
plt.title("Plot results")
plt.show()

# %%
# #%matplotlib notebook
# Try scores that are smoothed
#import statsmodels.api as sm
#lowess = sm.nonparametric.lowess
#y_cross_pred_scores
#z = lowess(y_cross_pred_scores, range(len(y_cross_pred_scores)))

# Plot test data
#plt.figure(num=None, figsize=(11.5, 7), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(df_timegraph['Time'],df_timegraph['Close'])
#plt.plot(df_timegraph['Time'],amplifyForPlot(y, df_timegraph['Close'].values, 0.00), color='green')
#d = (z[:, 1]>=0).astype('int')

#plt.plot(df_timegraph['Time'][X_cross.index],df_timegraph['Close'][X_cross.index])
#plt.plot(df_timegraph['Time'][X_cross.index],amplifyForPlot(y_cross.flatten(), df_timegraph['Close'][X_cross.index].values, 0.00), color='green')
#plt.plot(df_timegraph['Time'][X_cross.index],amplifyForPlot(d, df_timegraph['Close'][X_cross.index].values, 0.00), color='yellow')
#plt.title("Plot results")
#plt.show()



# %%
# Plot the results of unknown data

# %% [markdown]
# # Train Complete Model With Optimal Parameters
# - Train system with all data and best parameters
# - Save the model with parameters

# %%
# Train an SVM
from sklearn import svm
import time

number_of_samples = 1000000
kfold=10

display(param_final)
pipe_final = Pipeline([
        ('scaler', best_scaler),
        ('sampling', best_sampler),
        ('feat', ColumnExtractor(cols=best_columns)),
        ('svm', SVC(kernel=best_kernel, probability=True))
        ])

display(pipe_final)
print("Optimal precision/recall threshold: {}".format(pr_threshold))

t=time.time()
local_time = time.ctime(t)
print("=== Start training the SVM at {} ===".format(local_time))

results, gridsearch_final = grid_search_for_results(df_X, y, pipe_final, param_final, number_of_samples=number_of_samples, kfold=kfold)

finalclf = gridsearch_final.best_estimator_

elapsed = time.time() - t
print("Training took {}s".format(elapsed))
print("Training finished")

# %%
#Save the model
import joblib
import json

# save the optimal precision/recall value to disk
print("Save external parameters, precision recall threshold to disk")
with open(svm_external_parameters_filename, 'w') as fp:
    json.dump(extern_param_final, fp)
print("Saved external parameters to disk: ", svm_external_parameters_filename)
#with open(external_parameters_filename, "w") as f:
#    #for s in score:
#    f.write(str(pr_threshold) +"\n")

# save the model to disk
print("Model to save: ", finalclf)

joblib.dump(finalclf, svm_model_filename)
print("Saved model at location ", svm_model_filename)

# %%
#Run all above for model training

# %% [markdown]
# # Debug and Experiment
# Debug

# %% [markdown]
# #Test if the result of two classifiers differ significatly
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from mlxtend.data import iris_data
# from sklearn.model_selection import train_test_split
# from mlxtend.evaluate import paired_ttest_kfold_cv
#
#
# #X, y = iris_data()
# clf1 = LogisticRegression(random_state=1)
# clf2 = DecisionTreeClassifier(random_state=1)
#
# #X_train, X_test, y_train, y_test = \
# #    train_test_split(X, y, test_size=0.25,
# #                     random_state=123)
#
# score1 = clf1.fit(X_train, y_train).score(X_cross, y_cross)
# score2 = clf2.fit(X_train, y_train).score(X_cross, y_cross)
#
# print('Logistic regression accuracy: %.2f%%' % (score1*100))
# print('Decision tree accuracy: %.2f%%' % (score2*100))
#
# t, p = paired_ttest_kfold_cv(estimator1=clf1, estimator2=clf2, X=X_train, y=y_train, random_seed=0)
#
# print('t statistic: %.3f' % t)
# print('p value: %.3f' % p)
# #Since p>t, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different.
# if p>t:
#     print("p>t: We cannot reject the null hypothesis, i.e. that the performance of the algorithms differ significantly")
# else:
#     print("p<t: The performance of the algorithms differ significantly")

# %% [markdown]
# from scipy.stats import expon 
#
# # Random Variates 
# R = expon.rvs(loc=0, scale = 100, size=100) 
# print ("Random Variates : \n", R) 
#
# #C = uniform.rvs(loc=0.001, scale=10, size=100)
# #print(C)

# %% [markdown]
# # Test SMOTE on set
# #https://stackoverflow.com/questions/50245684/using-smote-with-gridsearchcv-in-scikit-learn
#
# from imblearn.over_sampling import SMOTE
# sm = SMOTE()
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
# print(X_train_res.shape)
# print(X_train.shape)

# %% [markdown]
# gridsearch_run2.cv_results_['mean_train_f1_micro_score']
# gridsearch_run2.cv_results_

# %% [markdown]
# def plot_grid_search_validation_curve(grid, param_to_vary, title='Validation Curve', ylim=None, xlim=None, log=None):
#     """Plots train and cross-validation scores from a GridSearchCV instance's
#     best params while varying one of those params."""
#     #https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results
#
#     df_cv_results = pd.DataFrame(grid.cv_results_)
#     train_scores_mean = df_cv_results['mean_train_f1_micro_score']
#     valid_scores_mean = df_cv_results['mean_test_f1_micro_score']
#     train_scores_std = df_cv_results['std_train_f1_micro_score']
#     valid_scores_std = df_cv_results['std_test_f1_micro_score']
#
#     param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
#     param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
#     param_ranges_lengths = [len(pr) for pr in param_ranges]
#
#     train_scores_mean = np.array(train_scores_mean).reshape(*param_ranges_lengths)
#     valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)
#     train_scores_std = np.array(train_scores_std).reshape(*param_ranges_lengths)
#     valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)
#
#     param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))
#
#     slices = []
#     for idx, param in enumerate(grid.best_params_):
#         if (idx == param_to_vary_idx):
#             slices.append(slice(None))
#             continue
#         best_param_val = grid.best_params_[param]
#         idx_of_best_param = 0
#         if isinstance(param_ranges[idx], np.ndarray):
#             idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
#         else:
#             idx_of_best_param = param_ranges[idx].index(best_param_val)
#         slices.append(idx_of_best_param)
#
#     train_scores_mean = train_scores_mean[tuple(slices)]
#     valid_scores_mean = valid_scores_mean[tuple(slices)]
#     train_scores_std = train_scores_std[tuple(slices)]
#     valid_scores_std = valid_scores_std[tuple(slices)]
#
#     plt.figure(figsize=(5,5))
#     plt.clf()
#
#     plt.title(title)
#     plt.xlabel(param_to_vary)
#     plt.ylabel('Score')
#
#     if (ylim is None):
#         plt.ylim(0.0, 1.1)
#     else:
#         plt.ylim(*ylim)
#
#     if (not (xlim is None)):
#         plt.xlim(*xlim)
#
#     lw = 2
#
#     plot_fn = plt.plot
#     if log:
#         plot_fn = plt.semilogx
#
#     param_range = param_ranges[param_to_vary_idx]
#     #if (not isinstance(param_range[0], numbers.Number)):
#     #    param_range = [str(x) for x in param_range]
#     plot_fn(param_range, train_scores_mean, label='Training score', color='r',
#             lw=lw)
#     plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color='r', lw=lw)
#     plot_fn(param_range, valid_scores_mean, label='Cross-validation score',
#             color='b', lw=lw)
#     plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
#                      valid_scores_mean + valid_scores_std, alpha=0.1,
#                      color='b', lw=lw)
#
#     plt.legend(loc='lower right')
#
#     plt.show()
#
# plot_grid_search_validation_curve(gridsearch_run2, 'svm__C', log=True, ylim=(0.80, 1.01))
# plot_grid_search_validation_curve(gridsearch_run2, 'svm__gamma', log=True, ylim=(0.80, 1.01))

# %% [markdown]
# from matplotlib import pyplot as PLT
# from matplotlib import cm as CM
# from matplotlib import mlab as ML
# import numpy as NP
#
# n = 1e5
# x = y = NP.linspace(-5, 5, 100)
# X, Y = NP.meshgrid(x, y)
# Z1 = ML.bivariate_normal(X, Y, 2, 2, 0, 0)
# Z2 = ML.bivariate_normal(X, Y, 4, 1, 1, 1)
# ZD = Z2 - Z1
# x = X.ravel()
# y = Y.ravel()
# z = ZD.ravel()
# gridsize=30
# PLT.subplot(111)
# # if 'bins=None', then color of each hexagon corresponds directly to its count
# # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
# # the result is a pure 2D histogram 
#
# PLT.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=None)
# PLT.axis([x.min(), x.max(), y.min(), y.max()])
#
# cb = PLT.colorbar()
# cb.set_label('mean value')
# PLT.show()   

# %% [markdown]
#

# %% [markdown]
# def plot_heat_from_rand_xy(scores, parameters, xlabel, ylabel, title, normalizeScale=False):
#     # Source of inspiration: https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
#
#     #Plot a heatmap of 2 variables
#     fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True, sharey=True)
#     colorMap = plt.cm.bone #plt.cm.gist_gray #plt.cm.hot
#     
#     if normalizeScale==True:
#         im1 = plt.imshow(scores, interpolation='catrom', cmap=colorMap, vmin=0 , vmax=1)
#     else:
#         #im1 = plt.imshow(scores, interpolation='nearest', cmap=colorMap)
#         im1 = plt.imshow(scores, interpolation='catrom', origin='lower', cmap=colorMap)
#
#     levels = np.linspace(np.min(scores), np.max(scores), 20)
#         
#    # contours = plt.contour(scores, 10, colors='black')
#     #contours = plt.contour(scores, 10, colors='black')
#     contours = ax.contourf(scores, levels=levels, cmap=plt.cm.bone)
#     
#     ax.contour(scores, levels=levels, colors='k', linestyles='solid', alpha=1, linewidths=.5, antialiased=True)
#     
#     #plt.contourf = 
#     ax.clabel(contours, inline=True, fontsize=8, colors='r')    
#     
#     #Get best value
#     def get_Top_n_values_from_array(arr, n):
#         result = []
#         for i in range(1,n+1):
#             x = np.partition(arr.flatten(), -2)[-i]
#             r = np.where(arr == x)
#             #print(r[0][0], r[1][0])
#             value = [r[0][0], r[1][0]]
#             result.append(value)
#         return result
#     
#     bestvalues = get_Top_n_values_from_array(scores, 10)
#     [plt.plot(pos[1], pos[0], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="best_value") for pos in bestvalues]
#     #best = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
#     #print(best)
#     #plt.plot(parameters[xlabel][best[1]], parameters[ylabel][best[0]], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="value1")
#     #plt.plot(3, 8, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="value1")
#     
#     plt.sca(ax)
#     plt.xticks(np.arange(len(parameters[xlabel])), ['{:.1E}'.format(x) for x in parameters[xlabel]], rotation='vertical')
#     plt.yticks(np.arange(len(parameters[ylabel])), ['{:.1E}'.format(x) for x in parameters[ylabel]])
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.xlim(0, len(parameters[xlabel])-1)
#     plt.ylim(0, len(parameters[ylabel])-1)
#     ax.set_title(title)
#     
#     #cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.8])
#     cbar = fig.colorbar(im1, ax=ax) #cax=axs[1])
#     plt.show()
#     
# def grid(x, y, z, resX=100, resY=100):
#     "Convert 3 column data to matplotlib grid"
#     xi = linspace(min(x), max(x), resX)
#     yi = linspace(min(y), max(y), resY)
#     points = np.array([x, y]).T
#     Z = griddata(x, y, z, xi, yi, interp='linear')
#     #print(len(Z))
#     #print(len(points))
#     X, Y = meshgrid(xi, yi)
#     return X, Y, Z
#     
# scores = gridsearch_run2.cv_results_['mean_test_' + refit_scorer_name][indexes].reshape(len(params_run2['svm__C']), len(params_run2['svm__gamma']))
# print(scores)
# print(params_run2)
#
# X, Y, Z = grid(x.data, y.data, cols)
# print(X[0])
# print(Z)
#
# plot_heat_from_rand_xy(scores, params_run2, 'svm__gamma', 'svm__C', 'F1 score, kernel=rbf')

# %% [markdown]
# np.random.seed(0)
# npts = 200
# ngridx = 100
# ngridy = 200
# x = np.random.uniform(-2, 2, npts)
# y = np.random.uniform(-2, 2, npts)
# z = x * np.exp(-x**2 - y**2)
#
# # griddata and contour.
# #start = time.clock()
# plt.subplot(211)
# xi = np.linspace(-2.1, 2.1, ngridx)
# yi = np.linspace(-2.1, 2.1, ngridy)
# zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
#
# #len(np.array(y).astype(np.float)[0:100])
# #x1
# #len(np.array(x).astype(np.float)[0:100])
# x.shape

# %% [markdown]
# x2 = np.array(x)[0:200]
# y2 = np.array(y)[0:200]
# x2.shape

# %% [markdown]
# import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# import numpy.ma as ma
# from numpy.random import uniform, seed
#
# # make up some randomly distributed data
# seed(1234)
# #npts = 200
# #x = uniform(-2,2,npts)
# #y = uniform(-2,2,npts)
# #z = x*np.exp(-x**2-y**2)
# z=cols
# # define grid.
# #xi = np.linspace(-2.1,2.1,100)
# #yi = np.linspace(-2.1,2.1,100)
# resX=100
# resY=100
# xi = np.linspace(min(x), max(x), resX)
# yi = np.linspace(min(y), max(y), resY)
# # grid the data.
# zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
# # contour the gridded data, plotting dots at the randomly spaced data points.
# CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
# CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.bone)
# plt.colorbar() # draw colorbar
# #plt.xscale('log')
# #plt.yscale('log')
# # plot data points.
# #plt.scatter(x,y,marker='o',c='b',s=5)
# #plt.xlim(min(np.log(x)),max(np.log(x)))
# #plt.xlim(1e1,1e4)
# #plt.ylim(1e-1,1e1)
# plt.title('griddata test (%d points)' % npts)
# plt.show()

# %% [markdown]
# points = np.random.rand(1000, 2)
# points
# s = np.array([x.data, y.data]).T
# s
#
#

# %% [markdown]
# from numpy import linspace, meshgrid
# from matplotlib.mlab import griddata
# from matplotlib import ticker, cm
# #from scipy.interpolate import griddata
#
# def grid(x, y, z, resX=100, resY=100):
#     "Convert 3 column data to matplotlib grid"
#     xi = linspace(min(x), max(x), resX)
#     yi = linspace(min(y), max(y), resY)
#     points = np.array([x, y]).T
#     Z = griddata(x, y, z, xi, yi, interp='linear')
#     #print(len(Z))
#     #print(len(points))
#     X, Y = meshgrid(xi, yi)
#     return X, Y, Z
#
# X, Y, Z = grid(x.data, y.data, cols)
# print(Z.shape)
# print(Z)
# print(X.shape)
# print(Y.shape)
# ax = plt.contourf(X, Y, Z, cmap=plt.cm.bone)
# plt.colorbar()

# %% [markdown]
# 10000 training examples need 63s. 20000 examples need 240s. Conclusion: Time inceases ^2 with increasing number of examples 

# %% [markdown]
# #Visualize
# from matplotlib.ticker import FuncFormatter
#
# def log_10_product(x, pos):
#     """The two args are the value and tick position.
#     Label ticks with the product of the exponentiation"""
#     return '%1f' % (x)
#
# score_table = results[['mean_test_f1_micro_score', 'param_C', 'param_gamma']].round(3)
# score_table
#
# fig = plt.figure(figsize=(8,5))
# ax = plt.gca()
# ax.scatter(score_table['param_C'], score_table['param_gamma'], c=score_table['mean_test_f1_micro_score'], cmap=plt.cm.hot)
# ax.set_xscale('log')
# ax.set_yscale('log')
# formatter = FuncFormatter(log_10_product)
# ax.xaxis.set_major_formatter(formatter)
# ax.yaxis.set_major_formatter(formatter)
# #ax.set_yscale('log')
# #ax.set_xscale('log')
# ax.set_xlim(1e1, 1e3+200)
# ax.set_ylim(1e-5, 1.5)
# ax.set_xlabel("C")
# ax.set_ylabel("gamma")
#

# %% [markdown]
# ## Normalize Data
# Z-Normalize the data around zero and divided by standard deviation. Fit the normalizer on the training data and transform the training and the test data. The reason is that the scaler only must depend on the training data, in order to prevent leakage of information from the test data.

# %% [markdown]
# from sklearn import preprocessing
#
# #=== Select the best type of scaler ===#
# scaler = preprocessing.StandardScaler()  #Normal distributed data
# #scaler = preprocessing.MinMaxScaler()
#
# scaler.fit(X_train_unscaled)
# #Use this scaler also for the test data at the end
# X_train = pd.DataFrame(data=scaler.transform(X_train_unscaled), index = X_train_unscaled.index, columns=X_train_unscaled.columns)
# print("Unscaled values")
# display(X_train_unscaled.iloc[0:2,:])
# print("Scaled values")
# display(X_train.iloc[0:2,:])

# %% [markdown]
# X_cross = pd.DataFrame(data=scaler.transform(X_cross_unscaled), index = X_cross_unscaled.index, columns=X_cross_unscaled.columns)
# print("Unscaled values")
# display(X_cross_unscaled.iloc[0:2,:])
# print("Scaled values")
# display(X_cross.iloc[0:2,:])

# %% [markdown]
# #=== Shuffle the training data to get random training of the SVM ===
# from sklearn.utils import shuffle
#
# X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)
# print("Unschuffled X,y data")
# display(X_train.head())
# display(y_train[0:5])
# print("Schuffled X,y data")
# display(X_train_shuffled.head())
# display(y_train_shuffled[0:5])
# print("A list of shuffled x values created")

# %% [markdown]
# #Test that split is ok
# value = X_train.iloc[2].name
# idx = X_train_shuffled.index.get_loc(value)
# y_train_shuffled[idx]

# %% [markdown]
# ### Bayesian Optimization of Parameters 1

# %% [markdown]
# params_run_bayes = { 
#         'svm__C': reciprocal(parameter_svm.loc['param_svm__C']['min'], parameter_svm.loc['param_svm__C']['max']), 
#         'svm__gamma': reciprocal(parameter_svm.loc['param_svm__gamma']['min'], parameter_svm.loc['param_svm__gamma']['max'])
#         }
#
# params_debug = {
#         'svm__C': (0.1, 10, 'log-uniform'), 
#         'svm__gamma': (0.1, 10, 'log-uniform')
#         }

# %% [markdown]
# pipe_run_random = Pipeline([
#         ('scaler', best_scaler),
#         ('sampling', best_sampler),
#         ('feat', ColumnExtractor(cols=best_columns)),
#         ('svm', SVC(kernel=best_kernel))
#         ])

# %% [markdown]
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from sklearn.svm import SVC
#
# # http://blairhudson.com/blog/posts/optimising-hyper-parameters-efficiently-with-scikit-optimize/
# # https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html
# # https://github.com/scikit-optimize/scikit-optimize/issues/762
#
#
# # we're using a logistic regression model
# #clf = LogisticRegression(random_state=1234, verbose=0)
#
# # this is our parameter grid
# #param_grid = {
# #    'solver': ['liblinear', 'saga'],  
# #    'penalty': ['l1','l2'],
# #    'tol': (1e-5, 1e-3, 'log-uniform'),
# #    'C': (1e-5, 100, 'log-uniform'),
# #    'fit_intercept': [True, False]
# #}
#
# pipe = Pipeline([
#     ('model', SVC())
# ])
#
# svc_search = {
#     'model': Categorical([SVC()]),
#     'model__C': Real(1e-1, 1e+1, prior='log-uniform'),
#     'model__gamma': Real(1e-1, 1e+1, prior='log-uniform'),
#     #'model__degree': Integer(1,8),
#     #'model__kernel': Categorical(['linear', 'poly', 'rbf']),
# }
#
# # set up our optimiser to find the best params in 30 searches
# opt = BayesSearchCV(pipe, svc_search, n_iter=20, cv=3)
#
# opt.fit(X_train, y_train)
#
# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))

# %% [markdown]
#

# %% [markdown]
# ### Bayesian Optimization of Parameters
# Use Hyperopt to find the best parameters for the MLP regressor
# Guide: https://blog.goodaudience.com/on-using-hyperopt-advanced-machine-learning-a2dde2ccece7

# %% [markdown]
# from hyperopt import tpe, hp, fmin
# from sklearn.svm import SVC
#
# #Define pipeline, which is constant for all tests
# pipe_run_random = Pipeline([
#         ('scaler', best_scaler),
#         ('sampling', best_sampler),
#         ('feat', ColumnExtractor(cols=best_columns)),
#         ('svm', SVC(kernel=best_kernel))
#         ])
#
#
# def objective_func(args):
#     if args['model']==SVC:
#         C = args['param']['C']
#         gamma = args['param']['gamma']
#         #solver = args['param']['solver']
#         #alpha = args['param']['alpha']
#         #learning_rate = args['param']['learning_rate']
#         #max_iter = args['param']['max_iter']
#         clf = SVC(C=C, gamma=gamma)
#
#     
#     clf.fit(X_train, y_train)
#     y_train_pred = clf.predict(X_train)
#     loss = cross_val_score(clf, X_train, y_train_pred).mean()
#     #print("Test Score:",clf.score(X_cross.values, y_cross.values.flatten()))
#     #print("Train Score:",clf.score(X_train.iloc[X_train_index_subset,:].values, y_train.values.flatten()[X_train_index_subset]))
#     #print("\n=================")
#     return loss
#
# #parameters = {
# #    'hidden_layer_sizes': [(100,), (500,), (700,), (100, 100), (500,500), (100,500), (50,100,50), (100,100,50), (100,100,100)],
# #    'activation': ['tanh', 'relu'],
# #    'solver': ['sgd', 'adam', 'lbfgs'],
# #    'alpha': [0.0001, 0.01, 1, 100, 1000],
# #    'learning_rate': ['constant','adaptive'],
# #}
#
#
# space = hp.choice('classifier',[
#         #{'model': MLPRegressor,
#         # 'param': {'hidden_layer_sizes': 
#         #           hp.choice('hidden_layer_sizes', [(100,), (500,), (700,), (100, 100), (500,500), (100,500), (50,100,50), (100,100,50), (100,100,100)]),
#         #           'activation': hp.choice('activation', ['tanh', 'relu']),
#         #           'solver': hp.choice('solver', ['sgd', 'adam', 'lbfgs']),
#          #          'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e3)),
#          #          'learning_rate': hp.choice('learning_rate', ['constant','adaptive']),
#          #          'max_iter': hp.randint('max_iter', 10000)+200
#          #         }
#         #},
#         {'model': SVC,
#          'param':{'C':hp.loguniform('C', np.log(1e-4), np.log(1e3)),
#                   #'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']),
#                   #'degree':hp.choice('degree',range(1,15)),
#                   'gamma':hp.uniform('gamma',1e-2,1e4)}
#         }
#         ])
# #'C':hp.lognormal('C',0,1),
#
# best_classifier = fmin(objective_func, space, algo=tpe.suggest,max_evals=10)
# print("Best classifier:", best_classifier)

# %% [markdown]
# from hyperopt import tpe, hp, fmin
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error
#
# def objective_func(args):
#     if args['model']==MLPRegressor:
#         hidden_layer_sizes = args['param']['hidden_layer_sizes']
#         activation = args['param']['activation']
#         solver = args['param']['solver']
#         alpha = args['param']['alpha']
#         learning_rate = args['param']['learning_rate']
#         max_iter = args['param']['max_iter']
#         clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
#                            activation=activation, 
#                            solver=solver, 
#                            alpha=alpha, 
#                            learning_rate=learning_rate, 
#                            max_iter=max_iter
#                           )
#
#     clf.fit(X_train.iloc[X_train_index_subset,:].values, y_train.values.flatten()[X_train_index_subset])
#     y_train_pred = clf.predict(X_train.iloc[X_train_index_subset,:].values)
#     loss = mean_squared_error(y_train.values.flatten()[X_train_index_subset], y_train_pred)
#     print("Test Score:",clf.score(X_cross.values, y_cross.values.flatten()))
#     print("Train Score:",clf.score(X_train.iloc[X_train_index_subset,:].values, y_train.values.flatten()[X_train_index_subset]))
#     print("\n=================")
#     return loss
#
# #parameters = {
# #    'hidden_layer_sizes': [(100,), (500,), (700,), (100, 100), (500,500), (100,500), (50,100,50), (100,100,50), (100,100,100)],
# #    'activation': ['tanh', 'relu'],
# #    'solver': ['sgd', 'adam', 'lbfgs'],
# #    'alpha': [0.0001, 0.01, 1, 100, 1000],
# #    'learning_rate': ['constant','adaptive'],
# #}
#
#
# space = hp.choice('classifier',[
#         {'model': MLPRegressor,
#          'param': {'hidden_layer_sizes': 
#                    hp.choice('hidden_layer_sizes', [(100,), (500,), (700,), (100, 100), (500,500), (100,500), (50,100,50), (100,100,50), (100,100,100)]),
#                    'activation': hp.choice('activation', ['tanh', 'relu']),
#                    'solver': hp.choice('solver', ['sgd', 'adam', 'lbfgs']),
#                    'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e3)),
#                    'learning_rate': hp.choice('learning_rate', ['constant','adaptive']),
#                    'max_iter': hp.randint('max_iter', 10000)+200
#                   }
#         },
#         {'model': SVC,
#          'param':{'C':hp.loguniform('alpha', np.log(1e-4), np.log(1e3)),
#                   'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']),
#                   'degree':hp.choice('degree',range(1,15)),
#                   'gamma':hp.uniform('gamma',1e-2,1e4)}
#         }
#         ])
# #'C':hp.lognormal('C',0,1),
#
# best_classifier = fmin(objective_func, space, algo=tpe.suggest,max_evals=10)
# print("Best classifier:", best_classifier)

# %% [markdown]
# ### Bayesian Optimization with Visualization of the Results
# Source: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

# %% [markdown]
# import numpy as np
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# import hyperopt.pyll.stochastic
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import scale, normalize
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
#
# #define X
# X_hyper_train = X_train.iloc[X_train_index_subset,:].values
# #deine y
# y_hyper_train = y_train.values.flatten()[X_train_index_subset]
#
# def f(params):
#     """object function for hyperopt"""
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
# def hyperopt_train_test(params):
#     """Hyperopt train test values"""
#     X_ = X_hyper_train[:]
#     #if 'normalize' in params:
#     #    if params['normalize'] == 1:
#     #        X_ = normalize(X_)
#     #        del params['normalize']
#     #if 'scale' in params:
#     #    if params['scale'] == 1:
#     #        X_ = scale(X_)
#     #        del params['scale']
#     clf = MLPRegressor(**params)
#     return cross_val_score(clf, X_, y_hyper_train, cv=5, n_jobs=-1).mean()
#
# #'hidden_layer_sizes': [(sp_randint.rvs(5,10,1),), (sp_randint.rvs(5,15,1),sp_randint.rvs(5,15,1))],
# #'activation': ['tanh', 'relu'],
# #'solver': ['sgd', 'adam'],
# #'alpha': uniform(1e-4, 1e1),
# #'learning_rate': ['constant','adaptive'],
# #'max_iter': [500]
#
# #Define the searchspace
# #space4mlp = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100,), (500,), (700,), (100, 100), (500,500), (100,500), (50,100,50), (100,100,50), (100,100,100)]),
# #             'activation': hp.choice('activation', ['tanh', 'relu']),
# #             'solver': hp.choice('solver', ['sgd', 'adam', 'lbfgs']),
# #             'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e3)),
# #             'learning_rate': hp.choice('learning_rate', ['constant','adaptive']),
# #             'max_iter': hp.randint('max_iter', 10000)+200
# #            }
#
# space4mlp = {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(3,), (5,), (10,), (15,), (100,), (3, 3), (5,5), (10,10), (15, 15), (5,5,5)]), 
#              'activation': hp.choice('activation', ['tanh', 'relu']),
#              'solver': hp.choice('solver', ['sgd', 'adam', 'lbfgs']),
#              'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e1)),
#              'learning_rate': hp.choice('learning_rate', ['constant','adaptive']),
#              'max_iter': hp.randint('max_iter', 1000)+200
#             }
#
# #print(hyperopt.pyll.stochastic.sample(space4svm))
#
#
# trials = Trials()
# best = fmin(f, space4mlp, algo=tpe.suggest, max_evals=100, trials=trials)

# %% [markdown]
# %matplotlib inline
# print('Best model:')
# print(best)
#
# print("Visualize trials")
# parameters = space4mlp.keys()
# #parameters = ['C', 'kernel', 'gamma']
# cols = len(parameters)
# f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
# cmap = plt.cm.jet
#
# for i, val in enumerate(parameters):
#     xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
#     ys = [-t['result']['loss'] for t in trials.trials]
#     xs, ys = zip(*sorted(zip(xs, ys)))
#     axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i)/len(parameters)))
#     axes[i].set_title(val)
#     axes[i].set_ylim([0.0, 1.0])

# %% [markdown]
# ### Automl
# Using TPOT
# Source: https://epistasislab.github.io/tpot
#
# Other available automl optimizers are: H2O http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html

# %% [markdown]
# #Use the TPOT classifier
# from tpot import TPOTClassifier
#
# tpot = TPOTClassifier(verbosity=3, max_time_mins=10, population_size=10, n_jobs=-1, cv=5, 
#                       use_dask=True, periodic_checkpoint_folder=filedataresultdirectory + "/_temp")
# tpot.fit(X_train, y_train.values.flatten())
# print(tpot.score(X_cross, y_cross.values.flatten()))

# %% [markdown]
# print("Best value:\n", tpot.fitted_pipeline_)

# %% [markdown]
# pipeline_filename = filedataresultdirectory + "/_temp/" + 'tpot_BikeSharing_pipeline.py'
# print("Export pipeline to", pipeline_filename)
# tpot.export(pipeline_filename)

# %%
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

from sklearn.ensemble import RandomForestClassifier
random_forest_clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(random_forest_clf, X, y)

proba = random_forest_clf.fit(X, y).predict_proba(X)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True, figsize=(10,10))
skplt.metrics.plot_precision_recall_curve(np.array(y), np.array(proba), figsize=(10,10))
skplt.metrics.plot_roc(np.array(y), np.array(proba), figsize=(10,10))
plt.show()

# %%
