#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KNN Model setup
License_info: ISC
ISC License

Copyright (c) 2020, Alexander Wendt

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import os

# Libs
import argparse
import warnings

from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from pandas.plotting import register_matplotlib_converters
import pickle
from pickle import dump

# from IPython.core.display import display
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import utils.sklearn_utils as modelutil
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

# Own modules
from models.model_param import ModelParamInterface

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'


class ModelParam(ModelParamInterface):
    """
    SVM Default parameters

    To change the setup, modify the use_parameters method.

    """

    def __init__(self):
        print("in init")

    def get_model_type(self):
        return 'knn'

    def use_parameters(self, X_train, selected_features):
        """
        Default Parameter

        """

        test_scaler = [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()]
        test_sampling = [modelutil.Nosampler(),
                         ClusterCentroids(),
                         RandomUnderSampler(),
                         # NearMiss(version=1),
                         # EditedNearestNeighbours(),
                         # AllKNN(),
                         # CondensedNearestNeighbour(random_state=0),
                         # InstanceHardnessThreshold(random_state=0,
                         #                          estimator=LogisticRegression(solver='lbfgs', multi_class='auto')),
                         RandomOverSampler(random_state=0),
                         SMOTE(),
                         BorderlineSMOTE(),
                         SMOTEENN(),
                         SMOTETomek(),
                         ADASYN()]
        #test_C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        #test_C_linear = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

        # gamma default parameters
        #param_scale = 1 / (X_train.shape[1] * np.mean(X_train.var()))

        #parameters = [
        #    {
        #        'scaler': test_scaler,
        #        'sampling': test_sampling,
        #        'feat__cols': selected_features,
        #        'model__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        #        'model__weights': ['uniform', 'distance']
        #    }]

        parameters = [
            {
                'scaler': test_scaler,
                'sampling': test_sampling,
                'feat__cols': selected_features,
                'model__n_neighbors': [13, 15, 21, 25],
                'model__weights': ['uniform', 'distance']
            }]

        # If no missing values, only one imputer strategy shall be used
        if X_train.isna().sum().sum() > 0:
            parameters['imputer__strategy'] = ['mean', 'median', 'most_frequent']
            print("Missing values used. Test different imputer strategies")
        else:
            print("No missing values. No imputer necessary")

            print("Selected Parameters: ", parameters)
        # else:
        print("Parameters defined in the input: ", parameters)

        return parameters


    def get_categorical_parameters(self):
        return [
            'scaler',
            'sampling',
            'model__n_neighbors',
            'model__weights',
            'feat__cols',
        ]

    def use_debug_parameters(self, reduced_selected_features):
        # Define parameters as an array of dicts in case different parameters are used for different optimizations
        params_debug = [{'scaler': [StandardScaler()],
                         'sampling': [modelutil.Nosampler(), SMOTE(), SMOTEENN(), ADASYN()],
                         'feat__cols': reduced_selected_features[0:2],
                         'model__n_neighbors': [3, 5],
                         'model__weights': ['uniform', 'distance']
                         }]

        return params_debug

    def create_pipeline(self):
        pipe_run = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('sampling', modelutil.Nosampler()),
            ('feat', modelutil.ColumnExtractor(cols=None)),
            ('model', KNeighborsClassifier())
        ])
        return pipe_run

    def define_best_pipeline(self, best_values_dict, best_columns, models_run1):
        pipe_run_best_first_selection = Pipeline([
            ('scaler', best_values_dict.get('scaler')),
            ('sampling', best_values_dict.get('sampling')),
            ('feat', modelutil.ColumnExtractor(cols=best_columns)),
            ('model', models_run1.set_params(
                n_neighbors=best_values_dict.get('model__n_neighbors'),
                weights=best_values_dict.get('model__weights')
            ))
        ])

        return pipe_run_best_first_selection