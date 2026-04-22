#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Parameters for SVR
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import utils.sklearn_utils as modelutil
from models.model_param import ModelParamInterface as ModelParamBase

class ModelParam(ModelParamBase):
    def __init__(self):
        super().__init__()

    def get_model_type(self):
        return 'svm'

    def create_pipeline(self):
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', StandardScaler()),
            ('sampling', modelutil.Nosampler()), # Not used for regression but kept for structure
            ('feat', modelutil.ColumnExtractor(cols=None)),
            ('model', SVR())
        ])
        return pipe

    def use_parameters(self, X_train, selected_features):
        test_scaler = [StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer()]
        test_C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        
        # gamma default parameters
        param_scale = 1 / (X_train.shape[1] * np.mean(X_train.var()))

        parameters = [
            {
                'scaler': test_scaler,
                'feat__cols': selected_features,
                'model__C': test_C,
                'model__kernel': ['rbf'],
                'model__gamma': [param_scale, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
            },
            {
                'scaler': test_scaler,
                'feat__cols': selected_features,
                'model__C': test_C,
                'model__kernel': ['linear']
            }
        ]
        return parameters

    def use_debug_parameters(self, selected_features):
        parameters = [{
            'scaler': [StandardScaler()],
            'feat__cols': [selected_features[0]],
            'model__C': [1.0],
            'model__kernel': ['rbf'],
            'model__gamma': ['scale']
        }]
        return parameters

    def get_categorical_parameters(self):
        return ['scaler', 'feat__cols', 'model__kernel']

    def define_best_pipeline(self, best_values_dict, best_columns, models_run1):
        pipe_run_best_first_selection = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('scaler', best_values_dict.get('scaler')),
            ('sampling', modelutil.Nosampler()),
            ('feat', modelutil.ColumnExtractor(cols=best_columns)),
            ('model', models_run1.set_params(
                kernel=best_values_dict.get('model__kernel')
            ))
        ])
        return pipe_run_best_first_selection
