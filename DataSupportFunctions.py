import random
import numpy as np

def inverse_dict(dictionary):
    #Inverse a dictionary
    dictionary_reverse = {}
    for k, v in dictionary.items():
        dictionary_reverse[v] = k
    return dictionary_reverse

def get_data_subset_index(numberOfSamples, X):
    #Get a random subset of data from a set
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
    # Check if y is binarized
    if len(y_classes) == 2 and np.max(list(y_classes.keys())) == 1:
        is_multiclass = False
        print("Classes are binarized, 2 classes.")
    else:
        is_multiclass = True
        print("Classes are not binarized, {} classes with values {}. "
              "For a binarized class, the values should be [0, 1].".format(len(y_classes), list(y_classes.keys())))
    return is_multiclass

class ColumnExtractor(object):
    '''Column extractor method to extract selected columns from a list. This is used as a feature selector. Source
    https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline.'''

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        #for c in self.cols:
        #    col_list.append(X[:, c:c+1])
        #return np.concatenate(col_list, axis=1)
        return X[self.cols]

    def fit(self, X, y=None):
        return self
