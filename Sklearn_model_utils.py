from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class Nosampler(BaseEstimator, TransformerMixin):
    '''
    The nosampler class do not do any type of sampling. It shall be used to compare with common over, under and
    combined samplers

    '''

    #def __init__(self):
        #self.cols = cols

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        return self

class ColumnExtractor(BaseEstimator, TransformerMixin):
    '''
    Column extractor method to extract selected columns from a list. This is used as a feature selector. Source
    https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline,
    http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/.

    '''

    def __init__(self, cols=[0]):
        self.cols = cols

    def transform(self, X):
        col_list = []
        if self.cols is not None:
            return X[:,self.cols]
        else:
            return X
        #for c in self.cols:
        #    col_list.append(X[:, c:c+1])
        #return np.concatenate(col_list, axis=1)
        #return X[self.cols].values

    def fit(self, X, y=None):
        return self

def extract_data_subset(X_train, y_train, number_of_samples, shuffled=True):
    '''
    Extract subset of a dataset with X and y. The subset size is set and if the data shall be shuffled

    '''

    print("Original size X: ", X_train.shape)
    print("Original size y: ", y_train.shape)
    if number_of_samples < X_train.shape[0]:
        print("Quota of samples used in the optimization: {0:.2f}".format(number_of_samples / X_train.shape[0]))
        _, X_train_subset, _, y_train_subset = train_test_split(X_train, y_train, random_state=0,
                                                                test_size=number_of_samples / X_train.shape[0],
                                                                shuffle=shuffled, stratify=y_train)
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    print("Subset size X: ", X_train_subset.shape)
    print("Subset size y: ", y_train_subset.shape)

    a, b = np.unique(y_train_subset, return_counts=True)
    print("Classes {}, counts {}".format(a, b))


    return X_train_subset, y_train_subset

def generate_result_table(gridsearch_run, params_run, refit_scorer_name):
    '''
    Generate a result table from a sklearn model run.
    gridsearch_run: the run
    params_run: parameters for the grid search
    refit_scorer_name: refit scorer name

    '''

    #merge all parameters to get the keys of all runs
    merged_params = {}
    # Check if parameters are a list or a dict. If they are a list of dicts, they have to be merged to get all
    # values within a table. Else, nothing has to be done.
    if isinstance(params_run, list):
        for d in params_run:
            merged_params.update(d)
    else:
        merged_params = params_run


    if isinstance(merged_params, list):
        table_parameters = ['param_' + x for x in merged_params[0].keys()]
    else:
        table_parameters = ['param_' + x for x in merged_params.keys()]

    metric_parameters = ['mean_test_' + refit_scorer_name, 'std_test_' + refit_scorer_name]
    result_columns = metric_parameters
    result_columns.extend(table_parameters)

    results = pd.DataFrame(gridsearch_run.cv_results_)
    results = results.sort_values(by='mean_test_' + refit_scorer_name, ascending=False)
    results[result_columns].round(3).head(20)

    return results[result_columns]


def create_parameter_grid_for_svm(results, top_results=None):
    '''Get the top x results from the result list and create a parameter list from it

    :args:
        results: Result table
        top_results: Pick the top results to create a new list of parameters for the random search. Default=None
           means that all values are picked.

    :return:
        params_new: New parameter list from the result values
    '''

    if top_results is None:
        top_results = results.shape[0]

    params_new = []
    for i in range(top_results):
        new_dict = {}
        new_dict['svm__C'] = [results.iloc[i]['param_svm__C']]
        new_dict['svm__gamma'] = [results.iloc[i]['param_svm__gamma']]
        params_new.append(new_dict)

    return params_new