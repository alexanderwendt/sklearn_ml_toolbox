# The SKLearn Machine Learning Toolbox
The Machine Learning Toolbox represents a complete tool-chain of data preparation, data analysis, model training and prediction of unknown data. 
The machine learning toolbox is supposed to be used for primary small and middle sized, structured datasets. 

In this toolbox, a stock index prediction example is provided to show how to use the tools. The tools contains several methods collected from 
different sources like stack overflow. The purpose is to have a comprehensive set of tools to handle many types of data. Methods that have been 
copied are referenced by the url. 

Machine Learning algorithms from the Scikit Learn library are used in the implementations.

## Setup
In a python 3.6+ environment, run 
```shell
pip install -r requirements.txt
```

In case all libraries cannot be setup correctly, they can be installed manually based on the requirements.txt

## Setup a Project
The toolbox contains several python scripts that are divided into steps, e.g. "step20_". In the following, each step will be described. The file structure 
with scripts is supposed to be used as a template to adapt to an arbitrary dataset, as scripts almost always have to be adapted to their specific use case.  

Toolbox default directories:
- (root): all python scripts used in the process
- samples: Debug and test projects, on which the scripts have been tested.
- doc: documentation

For individual projects, it is recommended to use the directory structure in the sample/debug_omxs30 project. These directories are defined in the project
configuration file and the naming can be changed. Project specifix directories:
- annotations: Place to put the class labels
- data_raw: Put your raw data here, i.e. tables with training and test data as well as inference data
- data_prepared: Prepared training, validation and inference data with all features and handled NaNs. The scripts process the raw data and then puts new 
tables into the prepared data folder. 
- models: Stored, trained models
- results: Results of training and prediction
- config: This is the location, where the configuration files are located. In the script arguments, this path is provided.
- (root): Shell or bat scripts that are used to run the pipeline. They refer to the location of the scripts.


## Machine Learning Toolbox Process
The process and the scripts will be described with a example. To demonstrate the machine learning toolbox, the problem of classifying the 
trend of the swedish OMXS30 stock index was selected. From the raw data, future samples were used to detemine the current trend. The challenge is 
to recognize the correct trend, which is simple when looking backward, but hard in advance or in the current moment. The trend is classified 
in a positive and a negative trend.

The raw input data has the following format
Date | Open | High | Low | Close 
--- | --- | --- | --- |--- |
1986-09-30 | 125 | 125 | 125 | 125
1986-10-01 | 125 | 125.93 | 125.93 | 125.93
1986-10-02 | 125.93 | 126.25 | 126.25 | 126.25
1986-10-03 | 126.25 | 126.68 | 126.68 | 126.68


### Feature and Outcomes Generation Step 2X
For raw data, sometimes, it is necessary to generate features or outcomes. In step 2X, Feature generation as well as outcome generation is applied.
In the data preparation, the y values are generated if applicable. 


#### step20_generate_groundtruth_stockmarket.py
In the OMXS30 example, the positive trend is illustrated as orange and the negative 
trend is blue. The long-term trend has been defined automatically by combining a top-bottom recognition algorithm with Lowess.

<img src="doc/saved_images/S20_omxs30_tb_Groud_Truth_LongTrend_two_class_graph.png" width="900">

#### step20_generate_groundtruth_stockmarket_from_annotation.py
If the outcomes are not automatically generated, they can be loaded from a csv file instead.

#### step21_generate_features.py
Features are generated based on the raw X data. In the example, technical indicators like moving average, RSI and Stochastics are used to generate features. 
Any technical indicators can be added here.

#### step22_adapt_dimensions.py
In the generation of outcomes, different averering methods are used, which use future data. To get correct labeling, future data is removed at the end of the generation, e.g. last 50 values.
In the generation of features, moving averages 200 are used. Therefore, the 200 first values are cut off to get a correct feature representation. Both 
features and outcomes have to have the same dimensions. This adaption is automatically done in with this script.

This step is very individual for each problem. Only step22_adapt_dimensions.py could remain the same in another project.

### Data Preparation, Analysis and Feature Selection 3X
In this process step, the following processing is done:
- Dataset loading
- Dataset cleaning
- Dataset analysis
- Feature selection for the machine learning algorithms

#### step30_clean_raw_data.py
Prepared features and labels are loaded from files. The following cleaning steps are applied:
- Column names are stripped 
- " " are replaced by "_" to get unified naming
- "/" are replaxed by "-"
- All string values "?" or empty values are replaced by numpy "NaN"

In this module, each feature is plotted and visualized with median and means like below.

<img src="doc/saved_images/omxs30_lt_25-MA200Norm.png" width="400">
<img src="doc/saved_images/omxs30_lt_3-LongTrend.png" width="400">

The outcome in this example is "LongTrend". From the graph, it is visible that the classes are skewed. There are twice as many values of "positive trend" as 
"negative trend".

Usually, this script has to adapted to an input dataset. However, if several similar datasets are analyzed or inference data is continually used, this script
can remain unchanged.

#### step31_adapt_features.py
After the raw data has been processed in a first pass, features may have to be adapted for a machine learning algorithm. An example is the one-hot-encoding,
where nominal values are replaced by binary values for each value type.

This script does the following:
- one-hot encoding
- Making all columns numeric values to int or float
- Outcomes as int for classification problems
- Binarize labels: For some problems, only one class might be interesting. This is set in the configuration file. In this step, the selected class gets
the number 1 and all other classes get the number 0. In that way, a class is binarized. In our example, we are only interested in the positive trend. 

Finally, the share of missing values are shown in the following graph.

<img src="doc/saved_images/_missing_numbers_matrix.png" width="700">

In our example, there are no missing values (NaN).

#### step32_search_hyperparameters.py
In the next step, the data will be analyzed to get an overview of the distribution and possibilites to group it. Before that, some hyperparameters 
are search for T-SNE. The result of the hyperparameter search for T-SNE looks like this

<img src="doc/saved_images/omxs30_lt_TSNE_Calibration_Plot.png" width="700">

In this case, probably the standard parameters can be used.

#### step33_analyze_data.py
Several tools and graphs analyze and visualize different charactersics of the data to create an understanding of its structure.

Spearman Correlation

<img src="doc/saved_images/Spearman_Correlation_Plot.png" width="700">

Scatterplot Matrix

<img src="doc/saved_images/Scatter-Matrix.png" width="700">

Pairplot

<img src="doc/saved_images/Pairplot.png" width="700">

Correlation strength between features and outcome

<img src="doc/saved_images/omxs30_ltred_Correlation_Strength.png" width="400">

Hierarchical Linkage

<img src="doc/saved_images/Hierarchical_Linkage.png" width="600">

Parallel coordinates of selected features

<img src="doc/saved_images/Parallel_Coordinates.png" width="700">

T-SNE unsupervised grouping of data

<img src="doc/saved_images/T-SNE_Plot.png" width="600">

UMAP unsupervised clustering

<img src="doc/saved_images/UMAP_Unsupervised.png" width="600">

UMAP supervised clustering

<img src="doc/saved_images/UMAP_Supervised.png" width="600">

PCA Plot

<img src="doc/saved_images/PCA_Plot.png" width="600">

PCA Variance Coverage. Plots the number of composed features that are necessary to cover 95% of the variance of the data. 
It shows that it is possible to reduce the number of features from 37 to 9.

<img src="doc/saved_images/PCA_Variance_Coverage.png" width="600">

#### step34_analyze_temporal_data.py
For time series, chart tools are available for plotting auto correlations of the source data.


#### step35_perform_feature_selection.py
Feature selection is done by using several different methods to get the most significant features and adding them to a list of features. This list is then tested in the model optimization step. The following feature selection methods are used:
- Logistic regression with lasso (L1) regulaization 

<img src="doc/saved_images/omxs30_ltred_Lasso_Model_Weights.png" width="400">

- Tree based feature selection 

<img src="doc/saved_images/omxs30_ltred_Tree_Based_Importance.png" width="400">

- Backward Elimination
- Recursive Elimination with Logistic Regression

All extracted features are merged into a data frame, similar to the following image for each feature extraction method.

<img src="doc/saved_images/Significant_Features_OMXS30.jpg" width="600">

Finally, the prepared dataset and the extracted features are stored.

#### step36_split_training_validation.py
The training data is then split into a training and a validation set. Often, the split is 80% for training data and 20% for validation data. The share of validation data
is defined in the configuration file. Also if data shall be shuffled is set in the configuration file.

### Model Training 4X
The model used is a Support Vector Machine. In the model optimization the following process steps are done.

Training is done in the following steps:
- Estimation of baseline that the classifier should beat
- Wide hyperparameter search to determine discrete values like scaler, features to use and sampler.
- Narrow hyperparameter search to determine the continuous values
- For multiclass problems, determine the precision/recall coefficient
- Train a model with the hyperparameters


#### step42_analyze_training_time_svm.py
With this script, it is possible to make estimations of dummy classifiers, i.e. the majority classifier to have as a baseline for the best guess. Further,
it is analyzed how the training duration and F1 score increases with increasing number of samples.

Baseline prediction based on majority class and stratified class prediction to detemine, whether the classifier provides signifiant results. In this case, as 2/3 of the trend
is positive, the accuracy of the majority classifier would be around 0.66.


Test training durations are tested to get an overview how the algorithm scales, i.e. training time of the classifier as a function of number of samples to be able 
to estimate the effort of training. In the case of an Support Vector Machine, it scales with O(n^2)

<img src="doc/saved_images/SVM_Duration_Samples.png" width="300">

Test training results as a function of the number of samples to determine the minimum training size. The question to be answer is how much data is needed to flatten the
learning curve.

<img src="doc/saved_images/SVM_F1_Samples.png" width="300">

#### step43_wide_hyperparameter_search_svm.py
A hyperparameter search is performed for the machine learning algorithm. For the Support Vector Machine, it is possible to narrow the search space with less
data and then make a fine grained search with more data. In the first search, discrete values are selected. 

Selection of Scaler, Sampler Feature Subset and Kernel. In the first run, the goal is to select the best scaler, sampler, kernel and feature subset. 
The hyper parameters C and gamma are used in a small range around the default value. Only a subset of the data is used, e.g. 20% of the samples. 
In a grid search, the best parameter of the following is determined and fixed.

Scalers:
- StandardScaler()
- RobustScaler()
- QuantileTransformer()
- Normalizer()

Proportion of the different scalers in the top 10% of the results

<img src="doc/saved_images/_scaler_categorical_bar.png" width="400">

Statistical difference significance matrix (0 for the same distribution, 1 for different distribution)

<img src="doc/saved_images/_scaler_significance_matrix.png" width="500">

Distributions of the scalers for the result range (f1) 

<img src="doc/saved_images/_scaler_overlayed_histograms.png" width="500">

Samples:
- Nosampler()
- SMOTE()
- SMOTEENN()
- SMOTETomek()
- ADASYN()

Proportion of the different scalers in the top 10% of the results

<img src="doc/saved_images/_sampling_categorical_bar.png" width="400">

Statistical difference significance matrix (0 for the same distribution, 1 for different distribution)

<img src="doc/saved_images/_sampling_significance_matrix.png" width="500">

Distributions of the samplers for the result range (f1) 

<img src="doc/saved_images/_sampling_overlayed_histograms.png" width="500">

Kernels:
- linear
- polynomial with degree 2, 3 or 4
- rbf
- sigmoid

Proportion of the different scalers in the top 10% of the results

<img src="doc/saved_images/_svm__kernel_categorical_bar.png" width="400">

Statistical difference significance matrix (0 for the same distribution, 1 for different distribution)

<img src="doc/saved_images/_svm__kernel_significance_matrix.png" width="500">

Distributions of the kernels for the result range (f1) 

<img src="doc/saved_images/_svm__kernel_overlayed_histograms.png" width="500">

Feature Selection
- Lasso	
- Tree based selection
- Backward elimination
- Recursive elimination top 1/3
- Recursive elimination top 2/3
- Recursive elimination top 3/3
- Manual feature selection
- All features

Proportion of the different scalers in the top 10% of the results

<img src="doc/saved_images/_feat__cols_categorical_bar.png" width="400">

Statistical difference significance matrix (0 for the same distribution, 1 for different distribution)

<img src="doc/saved_images/_feat__cols_significance_matrix.png" width="500">

Distributions of the feature selections for the result range (f1) 

<img src="doc/saved_images/_feat__cols_overlayed_histograms.png" width="500">

In out case, the algorithm selected the following pipline:
- Best scaler:  QuantileTransformer()
- Best sampler:  ADASYN()
- Best feature selection:  Tree
- Best kernel:  rbf

#### step44_narrow_hyperparameter_search_svm.py
Exhaustive parameter selection through narrow grid search. Only continuous parameters are optimized here. The basic parameters have been set. Now make an exhaustive parameter 
search for tuning parameters. Only a few samples are used and low kfold just to find the parameter limits. The parameters of C and gamma are selected wide.


To focus the search on hyper parameters C and gamma with a wide range. The optimization is divided into several runs where the top results are selected after each run with
more data.

For gamma and C the range was selected [1e-5, 1e5]. In the first run 1337 (40%) samples were run 4000 iterations and 3 folds with the random search. The results show the top 10% 
of the runs. The 100 best results are selected for the next run. The range is reduced significantly.

<img src="doc/saved_images/_run2_subrun_0_samples1337_fold3_iter4000_sel100.png" width="400">

The second run with 2674 samples, 2000 iterations and 5 folds. The 50 best results are selected for the next run. The range is further.

<img src="doc/saved_images/_run2_subrun_1_samples2674_fold5_iter2000_sel50.png" width="400">

The third run with 6685 samples, 100 iterations and 5 folds. The 20 best results are selected for the next run. The range is further.

<img src="doc/saved_images/_run2_subrun_2_samples6685_fold5_iter100_sel20.png" width="400">

From the narrow search, the following pipeline produces the best results:

In out case, the algorithm selected the following pipline:
- Best scaler:  QuantileTransformer()
- Best sampler:  ADASYN()
- Best feature selection: Tree
- Best kernel:  rbf
- C=2786.3123139377217
- gamma=0.6169115526350926


#### step45_define_precision_recall.py
The training data is split in a new training set and a cross validation set (shuffled and stratified). The training set is trained with the optimal 
parameters. On the cross validation data, the precision/recall curve is optimized as seen below. The decision threshold is moved in the optimal 
direction.

Precision/Recall curve for validation data

<img src="doc/saved_images/Val__pr_curve.png" width="500">

ROC curve

<img src="doc/saved_images/Val__roc_curve.png" width="500">

Precision/Recall threshold for validation data

<img src="doc/saved_images/Val__pr_scores_of_decision_threshold.png" width="500">

In our example with binary classes, the precision/recall threshold was -0.14

### Train model 5X
As all hyperparameters have been set, the next step is to train a model with complete pipline.

#### step50_train_model_from_pipe.py
The script takes a pipe from the hyperparameter optimization and trains it on the data. For the complete process, two models will be needed: the first model is
trained on the training data only. It will be used for validation with the validation data that was split in step 3X. The second model will be used for inference
and is trained on all available data. It is to expect that the more data is used for training, the better the model gets.

### Validation and Evaluation 6X
The model is trained with the complete training data and the optimal parameters.

#### step60_evaluate_model.py
Then, the model is evaluated on the training and validation data. The results are shown in a confusion matrix. The folloing image shows the comfusion matrix. 
The model almost perfectly fits the data.

<img src="doc/saved_images/Train__confusion_matrix.png" width="400">

The confusion matrix for the test data.

<img src="doc/saved_images/Val__confusion_matrix.png" width="400">

From the confusion matrices, it is visible that the classifier overfits the data

Decision boundary plot to see how well the classifier fits the data.

<img src="doc/saved_images/Val__decision_boundary_plot.png" width="700">


#### step61_evaluate_model_temporal_data.py
For temporal data like stock market data, this script offers the possibility to validate the predictions on the chart.

This graph shows the ground thuth of the training data, followed by the prediction of the training data. The prediction almost perfectly fits the 
training data.

<img src="doc/saved_images/Train_GT_omxs30_ltred_3class.png" width="700">

Prediction on training data.

<img src="doc/saved_images/Train_Pred_omxs30_ltred_3class.png" width="700">

The validation ground truth is shown here.

<img src="doc/saved_images/Val_GT_omxs30_ltred_3class.png" width="700">

The prediction on the validation data confirms the overfitting shown in the confusion matrix. 

<img src="doc/saved_images/Val_Pred_omxs30_ltred_3class.png" width="700">

With the system trained like this, is it possible to make money with this system? A filter that removes single outlier signals could help to get smoother
trends. It is expected that in a negative trend up to 26% values are predicted positive, where they should be negative. In the other way, 11% of the 
values are predicted negative, where the long trend is positive.

### Prediction 7X
In the prediction, the model from the training phase is loaded and used for prediction of the unknown data. 

#### step70_predict_temporal_data.py
This function is similar to the evaluation function and provides the inference for a chart. Looking at the last date 2021-01-26, the trend is predicted
to be positive.

<img src="doc/saved_images/InferenceInference_omxs30_ltred_3class.png" width="700">


## Future Work
The following items will be implemented to complete the tool chain
- Random Forest Class Classifier, KNN and multi layer perceptons
- Metrics and visualations for regression problems
 

## Issues
If there are any issues or suggestions for improvements, please add an issue to github's bug tracking system or please send a mail 
to [Alexander Wendt](mailto:alexander.wendt@tuwien.ac.at)



