#!/usr/bin/env python
# coding: utf-8

# # Data Loading, Cleaning and Feature Selection
# The following tasks are executed:
# - Data is loaded from raw files
# - Columnnames are fixed
# - Columns get correct object type
# - Class columns are put as the last columns
# - Missing data is replaced by NaN
# - Data characteristics are visualized
# - Missing data is visualized
# - Data is saved to a new, cleaned file as a df

# ## Parameters
# Here, all parameters of the notebook are set

# In[1]:


#Define a config file here to have a static start. If nothing is 
#config_file_path = "config_5dTrend_Training.json"
#config_file_path = "config_LongTrend_Training.json"

config_file_path = "config_LongTrend_Debug_Training.json"
#config_file_path = None


# In[2]:


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


# In[3]:


#Default test notebook parameters as dict
default_test_config = dict()
default_test_config['use_training_settings'] = False
default_test_config['dataset_name'] = "omxs30_test"
default_test_config['source_path'] = '01_Source/^OMX_2018-2020.csv'
default_test_config['class_name'] = "LongTrend"
#Binarize labels
default_test_config['binarize_labels'] = True
default_config['class_number'] = 1   #Class number in outcomes, which shall be the "1" class
default_test_config['binary_1_label'] = "Pos. Trend"
default_test_config['binary_0_label'] = "Neg. Trend"


# In[4]:


import joblib
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


# In[5]:


# Constants for all notebooks in the Machine Learning Toolbox

print("Directories")
training_data_directory = "02_Training_Data"
print("Training data directory: ", training_data_directory)

training_image_save_directory = training_data_directory + '/images'
print("Training data image save directory: ", training_image_save_directory)

test_data_directory = "03_Test_Prepared_Data"
print("Test data directory: ", test_data_directory)

test_image_save_directory = test_data_directory + '/images'
print("Training data image save directory: ", test_image_save_directory)


# In[6]:


#Secondary parameters for training and test settings
#Allow cropping of data, which are longer than moving averages in the future. For training data, this value shall be 
#true to not make false values. For the test values, which do not use any y values, the value shall be false.
#cut_data = conf['use_training_settings']   #If training settings are used, then cut the data. If test settings, do not cut the data
#print("cut_data =", cut_data)

if conf['use_training_settings']==True:
    target_directory = training_data_directory
    image_save_directory = training_image_save_directory
    
    #To save time in the generation of test data and to skip the feature analysis, set this option to false. For training
    #data, set it to false.
    skip_feature_analysis = False
    skip_feature_selection = False
else:
    target_directory = test_data_directory
    image_save_directory = test_image_save_directory
    
    #To save time in the generation of test data and to skip the feature analysis, set this option to false. For training
    #data, set it to true.
    skip_feature_analysis = True
    skip_feature_selection = True

print("Use training settings=", conf['use_training_settings'])
print("target directory=", target_directory)
print("Image save directory=", image_save_directory)
print("Skip feature analysis=", skip_feature_analysis)
print("Skip feature selection", skip_feature_selection)


# In[7]:


# Generating filenames for saving the files
features_filename = target_directory + "/" + conf['dataset_name'] + "_features" + ".csv"
model_features_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_features_for_model" + ".csv"
outcomes_filename = target_directory + "/" + conf['dataset_name'] + "_outcomes" + ".csv"
model_outcomes_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_outcomes_for_model" + ".csv"
labels_filename = target_directory + "/" + conf['dataset_name'] + "_labels" + ".csv"
source_filename = target_directory + "/" + conf['dataset_name'] + "_source" + ".csv"
#Modified labels
model_labels_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_labels_for_model" + ".csv"
#Columns for feature selection
selected_feature_columns_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_selected_feature_columns.csv"

print("=== Paths ===")
print("Input Features: ", features_filename)
print("Output Features: ", model_features_filename)
print("Input Outcomes: ", outcomes_filename)
print("Output Outcomes: ", model_outcomes_filename)
print("Labels: ", labels_filename)
print("Original source: ", source_filename)
print("Labels for the model: ", model_labels_filename)
print("Selected feature columns: ", selected_feature_columns_filename)


# ## Load dataset

# In[8]:


# Import libraries
#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import scipy.stats
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as m
from matplotlib.ticker import FuncFormatter, MaxNLocator
import data_visualization_functions as vis
import data_handling_support_functions as sup

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Global settings
np.set_printoptions(precision=3)

#Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

#Load skip cell kernel extension
#Source: https://stackoverflow.com/questions/26494747/simple-way-to-choose-which-cells-to-run-in-ipython-notebook-during-run-all
#%%skip True  #skips cell
#%%skip False #won't skip
#should_skip = True
#%%skip $should_skip
get_ipython().run_line_magic('load_ext', 'skip_kernel_extension')


# ### Load Features and Outcomes

# In[9]:


#=== Load Features ===#
features_raw = pd.read_csv(features_filename, sep=';').set_index('id') #Set ID to be the data id
display(features_raw.head(1))

#=== Load Outcomes ===#
outcomes_raw = pd.read_csv(outcomes_filename, sep=';').set_index('id') #Set ID to be the data id
display(outcomes_raw.head(1))

#=== Load Source ===#
source = pd.read_csv(source_filename, sep=';').set_index('id') #Set ID to be the data id
display(source.head(1))

#=== Load Class Labels ===#
#Get classes into a dict from outcomes
#class_labels = dict(zip(outcomes[class_name].unique(), list(range(1,len(outcomes[class_name].unique())+1, 1))))
#print(class_labels)

#Load class labels file
df_y_classes = pd.read_csv(labels_filename, delimiter=';', header=None)
class_labels = sup.inverse_dict(df_y_classes.set_index(df_y_classes.columns[0]).to_dict()[1])
print("Loaded  classes from file", class_labels)

#=== Define classes manually ===#
#class_labels = {
#    0 : 'class1',
#    1 : 'class2'
#}

print(class_labels)

#=== Define index name ===#
#Define name if there is no index name

#df.index.name = 'id'

#=== rename colums ===#
#df.rename(columns={'model.year':'year'}, inplace=True)

#Rename columns with " "
features_raw.columns = [x.replace(" ", "_") for x in features_raw.columns]
features_raw.columns = [x.replace("/", "-") for x in features_raw.columns]

print("Features size : ", features_raw.shape)
display(features_raw.head(5))
print("Outcomes size : ", outcomes_raw.shape)
display(outcomes_raw.head(5))


# ### Load Time Series

# In[10]:


#Load original data for visualization
import matplotlib.dates as mdates
import datetime

source = pd.read_csv(source_filename, delimiter=';').set_index('id')
source['Date'] = pd.to_datetime(source['Date'])
source['Date'].apply(mdates.date2num)
print("Loaded source time graph={}".format(source.columns))
print("X. Shape={}".format(source.shape))
display(source.head())


# ## Data Cleanup of Features and Outcomes before Features are Modified

# In[11]:


#Strip all string values to find the missing data
from pandas.api.types import is_string_dtype

for col in features_raw.columns:
    if is_string_dtype(features_raw[col]):
        print("Strip column {}".format(col))
        features_raw[col]=features_raw[col].str.strip()


# In[12]:


#Replace values for missing data

#=== Replace all missing values with np.nan
for col in features_raw.columns[0:-1]:
    features_raw[col] = features_raw[col].replace('?', np.nan)
    #df[col] = df[col].replace('unknown', np.nan)
    
print("Missing data in the data frame")
print(sum(features_raw.isna().sum()))


# In[13]:


#Get column types
print("Column types:")
print(features_raw.dtypes)
print("\n")


# In[14]:


print("feature columns: {}\n".format(features_raw.columns))
print("Outcome column: {}".format(outcomes_raw[conf['class_name']].name))


# ## Get Feature and Outcome Characteristics

# In[15]:


#Show possible classes
print(class_labels)


# In[16]:


# Get number of samples
numSamples=features_raw.shape[0]
print("Number of samples={}".format(numSamples))

# Get number of features
numFeatures=features_raw.shape[1]
print("Number of features={}".format(numFeatures))

#Get the number of classes for the supervised learning
numClasses = outcomes_raw[conf['class_name']].value_counts().shape[0]
print("Number of classes={}".format(numClasses))


# ## Analyse and Transform time series

# In[17]:


m.rc_file_defaults() #Reset sns

datatitle = conf['dataset_name']

plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
plt.plot(source['Date'],source['Close']) #To get scatter plot, add 'o' as the last parameter
plt.title(datatitle)
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.show()

plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
plt.plot(source['Date'],np.log(source['Close']))
plt.title(datatitle + ' log transformed')
plt.xlabel("Timestamp")
plt.ylabel("Log Price")
plt.show()


# ## Analyse the Features Individually

# In[18]:


# Print graphs for all features

def print_characteristics(features_raw, save_graphs=False):
    for i, d in enumerate(features_raw.dtypes):
        if is_string_dtype(d):
            print("Column {} is a categorical string".format(features_raw.columns[i]))
            s = features_raw[features_raw.columns[i]].value_counts()/numSamples
            fig = vis.paintBarChartForCategorical(s.index, s)
        else:
            print("Column {} is a numerical value".format(features_raw.columns[i]))
            fig = vis.paintHistogram(features_raw, features_raw.columns[i])

        plt.figure(fig.number)
        if save_graphs == True:
            plt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_{}-{}'.format(i, features_raw.columns[i]), dpi=300)
        plt.show()


# In[19]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'save_graphs = True   #If set true, then all images are saved into the image save directory.\n\n# Print graphs for all features\nprint_characteristics(features_raw, save_graphs=save_graphs)')


# In[20]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'save_graphs = True   #If set true, then all images are saved into the image save directory.\n\n# Print graphs for all features\nprint_characteristics(outcomes_raw, save_graphs=save_graphs)')


# In[21]:


#Optional Visualize all numeric values as melted, i.e. visualize the features together

#melted = pd.melt(df, id_vars=[class_Name], value_name="MergedValues")
#melted['MergedValues'].describe()

#%matplotlib inline
#plt.figure(figsize=(12, 4))
#plt.hist(melted['MergedValues'], bins=list(range(0, 500)), log=True)
#plt.xlabel('Value')
#plt.ylabel('Count (log)')
#plt.title("Histogram of Features")
#plt.show()


# In[22]:


#Visualize only the class to see if it is skewed
#vis.paintBarChartForCategorical(df[class_Name].value_counts().index, df[class_Name].value_counts())


# ## Prepare the Feature Columns

# In[23]:


#=== Replace signs for missing values or other values with ===#
features = features_raw.copy()

#Custom replacements, replace only if there is something to replace, else it makes NAN of it
#value_replacements = {
#    'n': 0,
#    'y': 1,
#    'unknown': np.NAN
#}

#=== Replace all custom values and missing values with content from the value_replacement
for col in features.columns:
    #df_dig[col] = df[col].map(value_replacements)
    #df_dig[col] = df[col].replace('?', np.nan)
    
    #Everything to numeric
    features[col] = pd.to_numeric(features[col])
    #df_dig[col] = np.int64(df_dig[col])
    
display(features.head(5))


# In[24]:


#Create one-hot-encoding for certain classes and replace the original class
#onehotlabels = pd.get_dummies(df_dig.iloc[:,1])

#Add one-hot-encondig columns to the dataset
#for i, name in enumerate(onehotlabels.columns):
#    df_dig.insert(i+1, column='Cylinder' + str(name), value=onehotlabels.loc[:,name])

#Remove the original columns
#df_dig.drop(columns=['cylinders'], inplace=True)


# ## Prepare the Outcomes

# In[25]:


# Replace classes with digital values
outcomes = outcomes_raw.copy()
outcomes = outcomes.astype(int)
print("Outcome types")
print(outcomes.dtypes)


# ### Binarize Multiclass Dataset

# In[26]:


#If the binaryize setting is used, then binarize the class of the outcome.
if conf['binarize_labels']==True:
    binarized_outcome = (outcomes[conf['class_name']] == conf['class_number']).astype(np.int_)
    y = binarized_outcome.values.flatten()
    print("y was binarized. Classes before: {}. Classes after: {}".format(np.unique(outcomes[conf['class_name']]), np.unique(y)))
    
    #Redefine class labels
    class_labels = {
        0 : conf['binary_0_label'],
        1 : conf['binary_1_label']
    }
    
    print("Class labels redefined to: {}".format(class_labels))
else:
    y = outcomes[conf['class_name']].values.flatten()
    print("No binarization was made. Classes: {}".format(np.unique(y)))


# In[27]:


#y = outcomes[class_name].values.flatten()
#y_labels = class_labels
#class_labels_inverse = sup.inverse_dict(class_labels)

print("y shape: {}".format(y.shape))
print("y labels: {}".format(class_labels))
print("y unique classes: {}".format(np.unique(y, axis=0)))


# ## Determine Missing Data
# Missing data is only visualized here as it is handled in the training algorithm in S40.

# In[28]:


# Check if there are any nulls in the data
print("Missing data in the features: ", features.isnull().values.sum())
features[features.isna().any(axis=1)]


# In[29]:


#Missing data part
print("Number of missing values per feature")
missingValueShare = []
for col in features.columns:
    #if is_string_dtype(df_dig[col]):
    missingValueShare.append(sum(features[col].isna())/numSamples)

#Print missing value graph
vis.paintBarChartForMissingValues(features.columns, missingValueShare)


# In[30]:


#Visualize missing data with missingno
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(features)


# In[31]:


if features.isnull().values.sum()>0:
    msno.heatmap(features)


# #### View Prepared Binary Features
# 
# We need some more plots for the binary data types.

# In[32]:


#vis.plotBinaryValues(df_dig, df_dig.columns) #0:-1
#plt.savefig(image_save_directory + "/BinaryFeatures.png", dpi=70)


# ## Feature Visualization

# ### Auto Correlations of Time Dependent Variables

# Source: https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
# 
# #### Autoregression Intuition
# 
# Consider a time series that was generated by an autoregression (AR) process with a lag of k. We know that the ACF describes the autocorrelation between an observation and another observation at a prior time step that includes direct and indirect dependence information. This means we would expect the ACF for the AR(k) time series to be strong to a lag of k and the inertia of that relationship would carry on to subsequent lag values, trailing off at some point as the effect was weakened. We know that the PACF only describes the direct relationship between an observation and its lag. This would suggest that there would be no correlation for lag values beyond k. This is exactly the expectation of the ACF and PACF plots for an AR(k) process.
# 
# #### Moving Average Intuition
# 
# Consider a time series that was generated by a moving average (MA) process with a lag of k. Remember that the moving average process is an autoregression model of the time series of residual errors from prior predictions. Another way to think about the moving average model is that it corrects future forecasts based on errors made on recent forecasts. We would expect the ACF for the MA(k) process to show a strong correlation with recent values up to the lag of k, then a sharp decline to low or no correlation. By definition, this is how the process was generated. For the PACF, we would expect the plot to show a strong relationship to the lag and a trailing off of correlation from the lag onwards. Again, this is exactly the expectation of the ACF and PACF plots for an MA(k) process.

# if the autocorrelation function has a very long tail, then it is no stationary process

# In[33]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\n\nm.rc_file_defaults() #Reset sns\n\n#Here, the time graph is selected\nprint("Plot the total autocorrelation of the price. The dark blue values are the correlation of the price with the lag. "+\n      "The light blue cone is the confidence interval. If the correlation is > cone, the value is significant.")\nplot_acf(np.log(source[\'Close\']))\nplt.title("Autocorrelation function of the OMXS30 price")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.show()\n\nplot_acf(np.log(source[\'Close\']))\nplt.title("Autocorrelation function of the OMXS30 price 600 first values")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.xlim([0,700])\nplt.show()\n\nprint("Ljung-Box statistics: p-value=", acorr_ljungbox(np.log(source[\'Close\']), lags=None, boxpierce="Ljung-Box")[1])\nprint("If p values > 0.05 then there are significant autocorrelations.")\n\nplot_pacf(np.log(source[\'Close\']), lags=200)\nplt.title("Partial Autocorrelation function of the OMXS30")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\n#plt.xlim([0,700])\nplt.show()\n\nplot_pacf(np.log(source[\'Close\']), lags=50)\nplt.title("Partial Autocorrelation function of the OMXS30 first 50 Values")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.xlim([0,10])\nplt.show()')


# In[34]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', '#Here, the time graph is selected\nprint("Plot the total autocorrelation of the price. The dark blue values are the correlation of the price with the lag. "+\n      "The light blue cone is the confidence interval. If the correlation is > cone, the value is significant.")\nplot_acf(features.MA200Norm)\nplt.title("Autocorrelation function of the MA200 price")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.show()\n\nplot_acf(features.MA200Norm)\nplt.title("Autocorrelation function of the MA200 price 600 first values")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.xlim([0,200])\nplt.show()\n\nprint("Ljung-Box statistics: p-value=", acorr_ljungbox(features.MA200Norm, lags=None, boxpierce="Ljung-Box")[1])\nprint("If p values > 0.05 then there are significant autocorrelations.")\n\nplot_pacf(features.MA200Norm, lags=200)\nplt.title("Partial Autocorrelation function of the OMXS30")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\n#plt.xlim([0,700])\nplt.show()\n\nplot_pacf(features.MA200Norm, lags=50)\nplt.title("Partial Autocorrelation function of the OMXS30 first 50 Values")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.xlim([0,10])\nplt.show()')


# In[35]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "#Plot difference between time values to see if the differences are stationary\ndiff = pd.DataFrame(data=np.divide(source['Close'] - source['Close'].shift(1), source['Close'])).set_index(source['Date'])\ndiff=diff.iloc[1:,:]\nfig = plt.figure(figsize= (15, 4))\nplt.plot(source['Date'].iloc[1:], diff)\nplt.grid()")


# In[36]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\n\n#Here, the time graph is selected\nprint("Plot the total autocorrelation of the price. The dark blue values are the correlation of the price with the lag. "+\n      "The light blue cone is the confidence interval. If the correlation is > cone, the value is significant.")\n\nplot_acf(diff)\nplt.title("Autocorrelation function of the price difference")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.ylim([-0.2, 0.2])\nplt.xlim([0,50])\nplt.show()\n\nplot_pacf(diff, lags=100)\nplt.title("Partial Autocorrelation function of the OMXS30 price difference")\nplt.xlabel("Lag")\nplt.ylabel("Correlation")\nplt.xlim([0,50])\nplt.ylim([-0.2, 0.2])\nplt.show()\n\nprint("Ljung-Box statistics: p-value=", acorr_ljungbox(diff, lags=None, boxpierce="Ljung-Box")[1])\nprint("If p values > 0.05 then there are significant autocorrelations.")')


# 

# # Feature Visualization
# Here, feature selection and visulization of datasets is performed
# Methods
# - Feature visualization through t-SNE
# - Feature visualization and analysis through PCA

# ### Standardize Data for Feature Selection and Visualization
# Z-Normalize the data around zero and divided by standard deviation. Fit the normalizer on the training data and transform the training and the test data. The reason is that the scaler only must depend on the training data, in order to prevent leakage of information from the test data.

# In[37]:


from sklearn import preprocessing

#=== Select the best type of scaler ===#
scaler = preprocessing.StandardScaler() #Because normal distribution. Don't use minmax scaler for PCA or unsupervised learning
# as the axis shall be centered and not shifted.


scaler.fit(features)
#Use this scaler also for the test data at the end
X_scaled = pd.DataFrame(data=scaler.transform(features), index = features.index, columns=features.columns)
print("Unscaled values")
display(features.iloc[0:2,:])
print("Scaled values")
display(X_scaled.iloc[0:2,:])

scaler.fit(y.reshape(-1, 1))
y_scaled = pd.DataFrame(data=scaler.transform(y.reshape(-1, 1)), index = features.index, columns=[conf['class_name']])
print("Unscaled values")
display(y[0:10])
print("Scaled values")
display(y_scaled.iloc[0:10,:])


# ### Feature and Outcomes Correlation Matrix

# In[38]:


total_values = X_scaled.join(y_scaled)
print("Merged features and outcomes to use in correlation matrix")


# In[39]:


#Select column values to use in the correlation plot
feature_plot=list(range(0,10,1))
#Select outcomes to show
feature_plot.extend([-4, -3, -2, -1])

print(feature_plot)
print(total_values.columns[feature_plot])


# In[40]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "#http://benalexkeen.com/correlation-in-python/\n#https://stackoverflow.com/questions/26975089/making-the-labels-of-the-scatterplot-vertical-and-horizontal-in-pandas\nfrom matplotlib.artist import setp\n\nm.rc_file_defaults() #Reset sns\n\naxs = pd.plotting.scatter_matrix(total_values.iloc[:,feature_plot], figsize=(15, 15), alpha=0.2, diagonal='kde')\nn = len(features.iloc[:,feature_plot].columns)\nfor i in range(n):\n    for j in range(n):\n        # to get the axis of subplots\n        ax = axs[i, j]\n        # to make x axis name vertical  \n        ax.xaxis.label.set_rotation(90)\n        # to make y axis name horizontal \n        ax.yaxis.label.set_rotation(0)\n        # to make sure y axis names are outside the plot area\n        ax.yaxis.labelpad = 50\n#plt.yticks(rotation=90)\nplt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_Scatter-Matrix', dpi=300)\nplt.show()")


# In[41]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'import seaborn as sns\n\n# https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6\n\nfeature_plot=list(range(0,10,1))\nfeature_plot.extend([-1])\n\ng = sns.pairplot(total_values.iloc[0:1000,feature_plot], hue=conf[\'class_name\'], diag_kind="hist")\n#total_values.columns[-1]\ng.map_upper(sns.regplot) \ng.map_lower(sns.residplot) \ng.map_diag(plt.hist) \nfor ax in g.axes.flat: \n    plt.setp(ax.get_xticklabels(), rotation=45) \ng.add_legend() \ng.set(alpha=0.5)\n\nplt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_Pairplot\', dpi=300)\nplt.show()')


# In[42]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "#http://benalexkeen.com/correlation-in-python/\nmatfig = plt.figure(figsize=(20, 20))\nplt.matshow(total_values.corr(method='spearman'), fignum=1, cmap=plt.get_cmap('coolwarm')) #Use spearman correlation instead of pearson to have a robust correlation\nplt.xticks(range(len(total_values.columns)), total_values.columns)\nplt.yticks(range(len(total_values.columns)), total_values.columns)\nplt.xticks(rotation=90)\nplt.colorbar()\n\nplt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_Spearman_Correlation_Plot', dpi=300)\n\nplt.show()")


# In[43]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "subx = X_scaled.iloc[1000:2000, 0:4]\nsuby = y_scaled.iloc[1000:2000]\n\ndisplay(subx)\ndisplay(suby)\n\ntype(suby)\nsubx.corrwith(suby[conf['class_name']], drop=True)")


# In[44]:


#import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display

from scipy.cluster import hierarchy
from scipy.spatial import distance

m.rc_file_defaults() #Reset sns

corr = X_scaled.corrwith(y_scaled[conf['class_name']], axis = 0)
corr.sort_values().plot.barh(color = 'blue',title = 'Strength of Correlation', figsize=(10,25))

plt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_Correlation_Strength', dpi=300)

display(corr)


# In[45]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "\n################\n## VERBESSERUNG\n################\n\n#Only for time series continuous correlation\n#Check correlations for the following attributes\nfor col in X_scaled.columns:\n    #cols = ['MA100Norm', 'RSI20']\n    #col_index = [X_scaled.columns.get_loc(c) for c in cols if c in X_scaled]\n    col_index = [X_scaled.columns.get_loc(col)]\n\n    tmp = X_scaled.iloc[:,col_index].join(source).join(y_scaled).reset_index().set_index('Date').drop(columns=['id', 'Open', 'High', 'Low', 'Close'])\n    #display(tmp)\n    tmp.dropna().resample('Q').apply(lambda x: x.corr()).iloc[:,-1].unstack().iloc[:,:-1].plot(title='Correlation of Features to Outcome', figsize=(8,2))\n    plt.ylim(-1, 1)\n    plt.hlines([-0.2, 0.2], xmin=tmp.index[0], xmax=tmp.index[-1], colors='r')\n    \n    plt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_Temporal_Correlation_to_Outcome_{}'.format(col), dpi=300)")


# In[46]:


#tmp = X_scaled.iloc[:,[0]].join(df_timegraph).join(y_scaled).reset_index().set_index('Time').drop(columns=['id', 'High', 'Low', 'Close'])
#tmp
#tmp.dropna().resample('M').apply(lambda x: x.corr()).iloc[:,-1].unstack().iloc[:,:-1]
#fig = plt.Figure()
#plt.scatter(tmp.iloc[:,1], tmp.iloc[:,0])


# In[47]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "corr_matrix = X_scaled.corr()\ncorrelations_array = np.asarray(corr_matrix)\n\nlinkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')\n\ng = sns.clustermap(corr_matrix,row_linkage=linkage,col_linkage=linkage, row_cluster=True,col_cluster=True,figsize=(30,30),cmap=plt.get_cmap('coolwarm'))\nplt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)\n\nplt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_Hierarchical_Linkage', dpi=300)\n\nplt.show()\n\nlabel_order = corr_matrix.iloc[:,g.dendrogram_row.reordered_ind].columns")


# ### Feature visualization with Parallel Coordinates

# In[48]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', '#Select a random subset to visualize\nimport random\n\ntotal_values = features.join(outcomes)\nprint("Merged features and outcomes to use in correlation matrix")\n\n#Reduce the training set with the number of samples randomly chosen\nX_train_index_subset = sup.get_random_data_subset_index(1000, features)\n\n#Select column values to use in the correlation plot\nfeature_plot=list(range(0,10,1))\n#cols = [\'MA2Norm\', \'MA50Norm\', \'MA200Norm\', \'MA400Norm\', \'MA200NormDiff\', \'MA400NormDiff\']\ncols = total_values.columns[feature_plot]\nprint(feature_plot)\nprint(cols)\n\ncomparison_name = conf[\'class_name\']\nprint("Class name: ", comparison_name)\n\ndf_fv = total_values.iloc[X_train_index_subset, :]')


# In[49]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', '# Use parallel coordinates to visualize the classes and all features for plotting\n#https://plot.ly/python/parallel-coordinates-plot/\n#http://benalexkeen.com/parallel-coordinates-in-matplotlib/\nfrom matplotlib import ticker\n\ndef plotParallelCoordinates(df, cols, colours, comparison_name):\n    x = [i for i, _ in enumerate(cols)]\n\n    # create dict of categories: colours\n    colours = {df[comparison_name].astype(\'category\').cat.categories[i]: colours[i] \n               for i, _ in enumerate(df[comparison_name].astype(\'category\').cat.categories)}\n\n    # Create (X-1) sublots along x axis\n    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))\n\n    # Get min, max and range for each column\n    # Normalize the data for each column\n    min_max_range = {}\n    for col in cols:\n        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]\n        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))\n\n    # Plot each row\n    for i, ax in enumerate(axes):\n        for idx in df.index:\n            mpg_category = df.loc[idx, comparison_name]\n            ax.plot(x, df.loc[idx, cols], colours[mpg_category])\n        ax.set_xlim([x[i], x[i+1]])\n    \n    # Set the tick positions and labels on y axis for each plot\n    # Tick positions based on normalised data\n    # Tick labels are based on original data\n    def set_ticks_for_axis(dim, ax, ticks):\n        min_val, max_val, val_range = min_max_range[cols[dim]]\n        step = val_range / float(ticks-1)\n        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]\n        norm_min = df[cols[dim]].min()\n        norm_range = np.ptp(df[cols[dim]])\n        norm_step = norm_range / float(ticks-1)\n        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]\n        ax.yaxis.set_ticks(ticks)\n        ax.set_yticklabels(tick_labels)\n\n    for dim, ax in enumerate(axes):\n        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))\n        set_ticks_for_axis(dim, ax, ticks=6)\n        ax.set_xticklabels([cols[dim]])\n    \n\n    # Move the final axis\' ticks to the right-hand side\n    ax = plt.twinx(axes[-1])\n    dim = len(axes)\n    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))\n    set_ticks_for_axis(dim, ax, ticks=6)\n    ax.set_xticklabels([cols[-2], cols[-1]])\n\n    # Remove space between subplots\n    plt.subplots_adjust(wspace=0)\n\n    # Add legend to plot\n    plt.legend(\n        [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df[comparison_name].astype(\'category\').cat.categories],\n        df[comparison_name].astype(\'category\').cat.categories,\n        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)\n\n    plt.title("Values of car attributes by LongTrend category")\n    \n    plt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_Parallel_Coordinates\', dpi=300)\n\n    plt.show()')


# In[50]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "m.rc_file_defaults() #Reset sns\n\ncolors = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']\nplotParallelCoordinates(df_fv, cols, colors, comparison_name)")


# ### Visualize Data with t-SNE

# In[51]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', '#Select a random subset to visualize\nimport random\n\n#Reduce the training set with the number of samples randomly chosen\nX_train_index_subset = sup.get_random_data_subset_index(1000, X_scaled)')


# In[52]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'from sklearn.manifold import TSNE\n#%matplotlib notebook\n#%matplotlib inline\n\nnp.random.seed(0)\n#X_embedded = TSNE(n_components=2, perplexity=5.0, early_exaggeration=12.0, n_iter=5000, \n#                  n_iter_without_progress=1000, learning_rate=10).fit_transform(embedded)\nX_embedded = TSNE(n_components=2, perplexity=10.0, early_exaggeration=100.0, n_iter=5000, \n                  n_iter_without_progress=1000, learning_rate=10).fit_transform(X_scaled.iloc[X_train_index_subset,:])')


# #### Plot t-SNE with best parameters

# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
m.rc_file_defaults() #Reset sns


# In[54]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', '#Plot with texts added to the graphs\n\n#from adjustText import adjust_text\ntargets = np.array(y[X_train_index_subset]).flatten()\n\nplt.figure(figsize=(10,10))\ntexts = []\nfor i, t in enumerate(set(targets)):\n    idx = targets == t\n    #for x, y in zip(X_embedded[idx, 0], X_embedded[idx, 1]):\n        #texts.append(plt.text(x, y, t))\n    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=class_labels[t])   \n\n#adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1,1), expand_text=(1,1), arrowprops=dict(arrowstyle="-", color=\'black\', lw=0.5)) \n\nplt.legend(bbox_to_anchor=(1, 1));\nplt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_T-SNE_Plot\', dpi=300)')


# #### t-SNE Parameter Grid Search

# In[55]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'from IPython.display import clear_output\n\n#Optimize t-sne plot\ntne_gridsearch = False\n\n#Create a TSNE grid search with two variables\nperplex = [5, 10, 30, 50, 100]\nexaggregation = [5, 12, 20, 50, 100]\n#learning_rate = [10, 50, 200]\n\nfig, axarr = plt.subplots(len(perplex), len(exaggregation), figsize=(15,15))\n\nif tne_gridsearch == True:\n    #for m,l in enumerate(learning_rate):\n    for k,p in enumerate(perplex):\n        #print("i {}, p {}".format(i, p))\n        for j,e in enumerate(exaggregation):\n            #print("j {}, e {}".format(j, e))\n            X_embedded = TSNE(n_components=2, perplexity=p, early_exaggeration=e, n_iter=5000, \n                              n_iter_without_progress=1000, learning_rate=10).fit_transform(X_scaled.iloc[X_train_index_subset,:])\n\n            for i, t in enumerate(set(targets)):\n                idx = targets == t\n                axarr[k,j].scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=class_labels[t])\n\n            axarr[k,j].set_title("p={}, e={}".format(p, e))\n        \n            clear_output(wait=True)\n            print(\'perplex paramater={}/{}, exaggregation parameterj={}/{}\'.format(k, len(perplex), j, len(exaggregation)))\n        \nfig.subplots_adjust(hspace=0.3)')


# ### UMAP Cluster Analysis
# Use a supervised/unsupervised analysis to make the clusters

# In[56]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "import numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set(style='white', context='poster')\n\n#import umap\nimport umap.umap_ as umap  #Work around from https://github.com/lmcinnes/umap/issues/24\n\n#%time #Time of the whole cell\nembeddingUnsupervised = umap.UMAP(n_neighbors=5).fit_transform(X_scaled)\n#%time #Time of the whole cell\nembeddingSupervised = umap.UMAP(n_neighbors=5).fit_transform(X_scaled, y=y)")


# In[57]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', "vis.plotUmap(embeddingUnsupervised, y, list(class_labels.values()), 'Dataset unsupervised clustering', cmapString='RdYlGn')\nplt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_UMAP_Unsupervised', dpi=300)\nvis.plotUmap(embeddingSupervised, y, list(class_labels.values()), 'Dataset supervised clustering')\nplt.savefig(image_save_directory + '/' + conf['dataset_name'] + '_UMAP_Supervised', dpi=300)")


# ### PCA Analysis

# In[58]:


get_ipython().run_cell_magic('skip', '$skip_feature_analysis', 'import sklearn.datasets as ds\nfrom sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\nimport seaborn as sns\n\nm.rc_file_defaults() #Reset sns\n\npca_trafo = PCA().fit(X_scaled);\npca_values = pca_trafo.transform(X_scaled)\n#from adjustText import adjust_text\ntargets = np.array(y).flatten()\n\nfig, ax1 = plt.subplots(figsize=(10, 8))\nplt.semilogy(pca_trafo.explained_variance_ratio_, \'--o\');\nax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis\nplt.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), \'--o\', color=\'green\');\nplt.xlabel("Principal Component")\nplt.ylabel("Explained variance")\nplt.xticks(np.arange(0, len(pca_trafo.explained_variance_ratio_)))\nplt.hlines(0.95, 0, len(pca_trafo.explained_variance_ratio_.cumsum()), colors=\'red\', linestyles=\'solid\', label=\'95% variance covered\')\nplt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_PCA_Variance_Coverage\', dpi=300)\n\n\nfig = plt.figure()\nsns.heatmap(np.log(pca_trafo.inverse_transform(np.eye(X_scaled.shape[1]))), cmap="hot", cbar=True)\n\nnecessary_components = pca_trafo.explained_variance_ratio_.cumsum()[pca_trafo.explained_variance_ratio_.cumsum()<0.95]\nprint("95% variance covered with the {} first components. Values={}". format(len(necessary_components), necessary_components))\n\nplt.figure(figsize=(10,10))\n#plt.scatter(pca_values[:,0], pca_values[:,1], c=targets, edgecolor=\'none\', label=class_labels.values(), alpha=0.5)\nfor i, t in enumerate(set(targets)):\n    idx = targets == t\n    plt.scatter(pca_values[idx, 0], pca_values[idx, 1], label=class_labels[t], edgecolor=\'none\', alpha=0.5)  \n\nplt.legend(labels=class_labels.values(), bbox_to_anchor=(1, 1));\nplt.xlabel(\'Component 1\')\nplt.ylabel(\'Component 2\')\n\nplt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_PCA_Plot\', dpi=300)\n\nplt.show()')


# ## Feature Selection

# In[59]:


#Data has already been scaled


# In[60]:


#Select a random subset to visualize
import random

#Reduce the training set with the number of samples randomly chosen
X_train_index_subset = sup.get_random_data_subset_index(1000, X_scaled)

relevantFeatureList = []
selected_feature_list = pd.DataFrame()


# In[61]:


#Predict with logistic regression
from sklearn.linear_model import LogisticRegression

def predict_features_simple(X, y):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    return clf.score(X, y)


# ### Lasso Feature Selection

# In[62]:


get_ipython().run_cell_magic('skip', '$skip_feature_selection', '#%matplotlib inline\nfrom sklearn.linear_model import LassoCV\n\nm.rc_file_defaults() #Reset sns\n\ndef execute_lasso_feature_selection(X_scaled, y):\n    reg = LassoCV(cv=10, max_iter = 100000)\n    reg.fit(X_scaled, y)\n    coef = pd.Series(reg.coef_, index = X_scaled.columns)\n    print("Best alpha using built-in LassoCV: %f" %reg.alpha_)\n    print("Best score using built-in LassoCV: %f" %reg.score(X_scaled,y))\n    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")\n    imp_coef = coef.sort_values()\n    coefList = list(imp_coef[imp_coef!=0].index)\n    print(coefList)\n\n    #plt.figure()\n    m.rcParams[\'figure.figsize\'] = (8.0, 10.0)\n    imp_coef.plot(kind = "barh")\n    plt.title("Feature importance using Lasso Model")\n    \n    plt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_Lasso_Model_Weights\', dpi=300)\n    \n    plt.show()\n    \n    return coefList\n    \n#if do_feature_analysis==True:\ncoefList = execute_lasso_feature_selection(X_scaled, y)\nselected_feature_list = selected_feature_list.append(pd.Series(name=\'Lasso\', data=coefList))\nrelevantFeatureList.extend(coefList)\n\nprint("Prediction of training data with logistic regression: {0:.2f}".format(predict_features_simple(X_scaled[coefList], y)))')


# ### Tree based feature selection

# In[63]:


get_ipython().run_cell_magic('skip', '$skip_feature_selection', 'from sklearn.ensemble import ExtraTreesClassifier\nfrom sklearn.feature_selection import SelectFromModel\n\ndef execute_treebased_feature_selection(X_scaled, y):\n    clf = ExtraTreesClassifier(n_estimators=50)\n    clf = clf.fit(X_scaled, y)\n    print(clf.feature_importances_)\n    print("Best score: %f" %clf.score(X_scaled, y))\n    model = SelectFromModel(clf, prefit=True)\n    X_new = model.transform(X_scaled)\n    X_new.shape\n\n    threshold = 0.010\n    tree_coef = pd.Series(clf.feature_importances_, index = X_scaled.columns)\n\n    print("Tree search picked " + str(sum(tree_coef >= threshold)) + " variables and eliminated the other " +  str(sum(tree_coef < threshold)) + " variables")\n    imp_treecoef = tree_coef.sort_values()\n    treecoefList = list(imp_treecoef[imp_treecoef>threshold].index)\n    print(treecoefList)\n\n    plt.figure()\n    m.rcParams[\'figure.figsize\'] = (8.0, 10.0)\n    imp_treecoef.plot(kind = "barh")\n    plt.title("Feature importance using Tree Search Model")\n    plt.vlines(threshold, 0, len(X_scaled.columns), color=\'red\')\n    \n    plt.savefig(image_save_directory + \'/\' + conf[\'dataset_name\'] + \'_Tree_Based_Importance\', dpi=300)\n    \n    plt.show()\n    \n    return treecoefList\n\n#if do_feature_analysis==True:\ntreecoefList = execute_treebased_feature_selection(X_scaled, y)\nselected_feature_list = selected_feature_list.append(pd.Series(name=\'Tree\', data=treecoefList))\nrelevantFeatureList.extend(treecoefList)\n\nprint("Prediction of training data with logistic regression: {0:.2f}".format(predict_features_simple(X_scaled[treecoefList], y)))')


# ### Backward Elimination

# In[64]:


get_ipython().run_cell_magic('skip', '$skip_feature_selection', '#Backward Elimination - Wrapper method\nimport statsmodels.api as sm\n\ndef execute_backwardelimination_feature_selection(X_scaled, y):\n    cols = list(X_scaled.columns)\n    pmax = 1\n    while (len(cols)>0):\n        p= []\n        X_1 = X_scaled[cols]\n        X_1 = sm.add_constant(X_1)\n        model = sm.OLS(y,X_1).fit()\n        p = pd.Series(model.pvalues.values[1:],index = cols)      \n        pmax = max(p)\n        feature_with_p_max = p.idxmax()\n        if(pmax>0.05):\n            cols.remove(feature_with_p_max)\n        else:\n            break\n    selected_features_BE = cols\n\n    print(selected_features_BE)\n    print("\\nNumber of features={}. Original number of features={}\\n".format(len(selected_features_BE), len(X_scaled.columns)))\n    [print("column {} removed".format(x)) for x in X_scaled.columns if x not in selected_features_BE]\n    print("Finished")\n    \n    return selected_features_BE\n\n#if do_feature_analysis==True:\nselected_features_BE = execute_backwardelimination_feature_selection(X_scaled, y)\nrelevantFeatureList.extend(selected_features_BE)\nselected_feature_list = selected_feature_list.append(pd.Series(name=\'Backward_Elimination\', data=selected_features_BE))\n\nprint("Prediction of training data with logistic regression: {0:.2f}".format(predict_features_simple(X_scaled[selected_features_BE], y)))')


# ### Recursive Elimination with Logistic Regression

# In[65]:


get_ipython().run_cell_magic('skip', '$skip_feature_selection', '#Recursive Elimination - Wrapper method, Feature ranking with recursive feature elimination\nfrom sklearn.linear_model import LogisticRegressionCV\nfrom sklearn.feature_selection import RFE\n\ndef execute_recursive_elimination_feature_selection(X_scaled, y):\n    model = LogisticRegressionCV(solver=\'liblinear\', cv=3)\n    print("Start Recursive Elimination. Fit model with {} examples.".format(X_scaled.shape[0]))\n    #Initializing RFE model, 3 features selected\n    rfe = RFE(model, 1) #It has to be one to get a unique index\n    #Transforming data using RFE\n    X_rfe = rfe.fit_transform(X_scaled,y)\n    #Fitting the data to model\n    model.fit(X_rfe,y)\n\n    print("Best accuracy score using built-in Logistic Regression: ", model.score(X_rfe, y))\n    print("Ranking")\n    rfe_coef = pd.Series(X_scaled.columns, index = rfe.ranking_-1).sort_index()\n    print(rfe_coef)\n    print("Select columns")\n\n    \n    print(X_scaled.columns[rfe.support_].values)\n    \n    return X_scaled.columns[rfe.support_].values, rfe_coef\n\n#if do_feature_analysis==True:\nrelevant_features, rfe_coef = execute_recursive_elimination_feature_selection(X_scaled.iloc[X_train_index_subset], y[X_train_index_subset])\nrelevantFeatureList.extend(relevant_features)\n\nstep_size = np.round(len(X_scaled.columns)/4,0).astype("int")\nfor i in range(step_size, len(X_scaled.columns), step_size):\n    selected_feature_list = selected_feature_list.append(pd.Series(name=\'RecursiveTop\' + str(i), data=rfe_coef.loc[0:i-1]))\n    print(\'Created RecursiveTop{}\'.format(str(i)))')


# In[66]:


#rfe_coef.loc[0:10].values


# In[ ]:





# ### Weighted values

# In[67]:


#Weights
values, counts = np.unique(relevantFeatureList, return_counts=True)
s = pd.Series(index=values, data=counts).sort_values(ascending=False)
print(s)


# ### Add Manually Selected Subset

# In[68]:


#print subset
newval = [x for x, c in zip(values, counts) if c>1]
subsetColumns = newval#X.columns[rfe.support_].values #list(values)
display(subsetColumns)
selected_feature_list = selected_feature_list.append(pd.Series(name='Manual', data=subsetColumns))


# ### Add all columns

# In[69]:


selected_feature_list = selected_feature_list.append(pd.Series(name='All', data=X_scaled.columns))


# In[70]:


#subsetColumns = X.columns[rfe.support_].values
#X_subset = X_raw[subsetColumns]
#display(X_subset.head(5))
selected_feature_list.transpose()


# ## Save Subset

# In[71]:


import csv

#=== Save features to a csv file ===#
print("Features shape {}".format(features.shape))
features.to_csv(model_features_filename, sep=';', index=True)
#np.savetxt(filenameprefix + "_X.csv", X, delimiter=";", fmt='%s')
print("Saved features to " + model_features_filename)

#=== Save the selected outcome to a csv file ===#
print("outcome shape {}".format(y.shape))
y_true = pd.DataFrame(y, columns=[conf['class_name']], index=outcomes.index)
y_true.to_csv(model_outcomes_filename, sep=';', index=True, header=True)
print("Saved features to " + model_outcomes_filename)

#=== Save new y labels to a csv file ===#
print("Class labels length {}".format(len(class_labels)))
with open(model_labels_filename, 'w') as f:
    for key in class_labels.keys():
        f.write("%s;%s\n"%(class_labels[key], key))  #Classes are saved inverse to the labelling in the file, i.e. first value, then key
print("Saved class names and id to " + model_labels_filename)

#=== Save x subset column names to a csv file as a list ===#
selected_feature_list.transpose().to_csv(selected_feature_columns_filename, sep=';', index=False, header=True)
print("Saved selected feature columns to " + selected_feature_columns_filename)

print("=== Data for {} prepared to be trained ===". format(conf['dataset_name']))


# ## Debug and Experiment

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# 
# from sklearn import datasets
# from sklearn.decomposition import PCA
# from sklearn.linear_model import SGDClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# 
# 
# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
#                          max_iter=10000, tol=1e-5, random_state=0)
# pca = PCA()
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
# 
# digits = datasets.load_digits()
# X_digits = digits.data
# y_digits = digits.target
# 
# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#     'pca__n_components': [5, 20, 30, 40, 50, 64],
#     'logistic__alpha': np.logspace(-4, 4, 5),
# }
# search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
# search.fit(X_digits, y_digits)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)
# 
# # Plot the PCA spectrum
# pca.fit(X_digits)
# 
# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(pca.explained_variance_ratio_, linewidth=2)
# ax0.set_ylabel('PCA explained variance')
# 
# ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
#             linestyle=':', label='n_components chosen')

# from tqdm import tqdm
# import time
# %matplotlib inline
# 
# pbar = tqdm(total=100)
# for i in range(10):
#     time.sleep(0.1)
#     pbar.update(10)
# pbar.close()

# from statsmodels import robust
# 
# a = np.matrix( [
#     [ 80, 76, 77, 78, 79, 81, 76, 77, 79, 84, 75, 79, 76, 78 ],
#     [ 66, 69, 76, 72, 79, 77, 74, 77, 71, 79, 74, 66, 67, 73 ]
#     ], dtype=float )
# robust.mad(a, axis=1)

# from pandas.plotting import autocorrelation_plot
# 
# autocorrelation_plot(df_timegraph['Close'])
# plt.show()

# In[ ]:




