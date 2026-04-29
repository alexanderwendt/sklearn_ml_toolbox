import os
import unittest
from unittest.mock import patch, mock_open
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.step50_train_model_from_pipe import train_final_model

class TestTrainModel(unittest.TestCase):

    @patch('utils.data_handling_support_functions.load_config')
    @patch('utils.execution_utils.load_data')
    @patch('joblib.dump')
    def test_train_final_model(self, mock_joblib_dump, mock_load_data, mock_load_config):
        """
        Test that train_final_model can load data and a pipeline, train a model, and save it.
        """
        # Mock the data and pipeline
        X_train = pd.DataFrame({'A': range(10)})
        y_train = pd.Series(range(10))
        pipe = Pipeline([('model', LogisticRegression())])
        
        mock_load_data.return_value = (X_train, y_train, pipe)
        
        # Mock the config
        mock_load_config.return_value = {
            'EvaluationTraining': {
                'model_out': 'model.joblib'
            },
            'Common': {
                'problem_type': 'classification'
            }
        }
        
        # Mock open for loading the pipeline
        m = mock_open(read_data=pickle.dumps(pipe))
        with patch('builtins.open', m):
            train_final_model('dummy_config_path', 'EvaluationTraining')
        
        # Check that the model was saved
        mock_joblib_dump.assert_called_once()

if __name__ == '__main__':
    unittest.main()
