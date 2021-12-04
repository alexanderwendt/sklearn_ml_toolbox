#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4X Training: Train wide Search for XGBoost
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

from pandas.plotting import register_matplotlib_converters
import pickle
from pickle import dump

# from IPython.core.display import display
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import utils.sklearn_utils as modelutil
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier

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
    XGBoost default parameters

    To change the setup, modify the use_parameters method.

    """

    def get_model_type(self):
        return 'xgboost'

    def use_parameters(self, X_train, selected_features):
        '''


        Returns
        -------

        '''
        test_scaler = [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()]
        test_sampling = [modelutil.Nosampler(),
                         # ClusterCentroids(),
                         # RandomUnderSampler(),
                         # NearMiss(version=1),
                         # EditedNearestNeighbours(),
                         # AllKNN(),
                         # CondensedNearestNeighbour(random_state=0),
                         # InstanceHardnessThreshold(random_state=0,
                         #                          estimator=LogisticRegression(solver='lbfgs', multi_class='auto')),
                         SMOTE(),
                         SMOTEENN(),
                         SMOTETomek(),
                         ADASYN()]

        ### XGBOOST
        parameters = [{
            'scaler': test_scaler,
            'sampling': test_sampling,
            'feat__cols': selected_features,
            'model__objective': ['logloss'],
            'model__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5],  # so called `eta` value
            'model__max_depth': [3, 4, 5],
            'model__min_child_weight': [1, 5, 11, 12, 15],
            'model__silent': [0],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__n_estimators': [5, 50, 100],  # number of trees, change it to 1000 for better results
            'model__missing': [-999],
            'model__gamma': [0.5, 1, 1.5, 2, 5],
            'model__seed': [1337]
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

        ### XGBOOST
        return parameters

    def get_categorical_parameters(self):
        return [
            'scaler',
            'sampling',
            'feat__cols',
            'model__learning_rate',
            'model__max_depth',
            'model__n_estimators'
        ]

    def use_debug_parameters(self, reduced_selected_features):
        ### XGBOOST CODE start
        params_debug = [{
            'scaler': [StandardScaler()],
            'sampling': [modelutil.Nosampler(), SMOTE(), SMOTEENN(), ADASYN()],
            'feat__cols': reduced_selected_features[0:2],
            'model__nthread': [4],  # when use hyperthread, xgboost may become slower
            'model__objective': ['binary:logistic'],
            'model__learning_rate': [0.05, 0.5],  # so called `eta` value
            'model__max_depth': [6, 7, 8],
            'model__min_child_weight': [11],
            'model__silent': [1],
            'model__subsample': [0.8],
            'model__colsample_bytree':[0.7],
            'model__n_estimators': [5, 10],  # number of trees, change it to 1000 for better results
            'model__missing': [-999],
            'model__seed': [1337]
        }]

        return params_debug


    def create_pipeline(self):
        pipe_run = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('sampling', modelutil.Nosampler()),
            ('feat', modelutil.ColumnExtractor(cols=None)),
            ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])
        return pipe_run




    def define_best_pipeline(self, best_values_dict, best_columns, models_run1):
        pipe_run_best_first_selection = Pipeline([
            ('scaler', best_values_dict.get('scaler')),
            ('sampling', best_values_dict.get('sampling')),
            ('feat', modelutil.ColumnExtractor(cols=best_columns)),
            ('model', models_run1.set_params(
                n_estimators=best_values_dict.get('model__n_estimators'),
                n_learning_rate=best_values_dict.get('model__learning_rate'),
                n_max_depth=best_values_dict.get('model__max_depth')
            ))
        ])

        return pipe_run_best_first_selection