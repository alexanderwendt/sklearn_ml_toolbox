import os
import pandas as pd
import unittest
from unittest.mock import patch, mock_open

from src.data_processing.step36_split_training_validation import split_train_validation_data

class TestSplitData(unittest.TestCase):

    @patch('utils.data_handling_support_functions.load_features')
    @patch('utils.data_handling_support_functions.load_config')
    @patch('pandas.DataFrame.to_csv')
    def test_split_train_validation_data(self, mock_to_csv, mock_load_config, mock_load_features):
        """
        Test that split_train_validation_data correctly splits the data and saves the files.
        """
        # Mock the data that would be loaded
        features = pd.DataFrame({'A': range(10)})
        df_y = pd.DataFrame({'y': range(10)})
        mock_load_features.return_value = (features, None, df_y, None)
        
        # Mock the config
        mock_load_config.return_value = {
            'Preparation': {
                'test_size': '0.2',
                'shuffle_data': 'True',
                'features_out_train': 'train_X.csv',
                'features_out_val': 'val_X.csv',
                'outcomes_out_train': 'train_y.csv',
                'outcomes_out_val': 'val_y.csv'
            }
        }
        
        # Call the function
        split_train_validation_data('dummy_config_path')
        
        # Check that the data was split and saved
        self.assertEqual(mock_to_csv.call_count, 4)
        
        # Further checks could be made on the shapes of the saved DataFrames
        # For example, by inspecting the first argument of the mock_to_csv calls.
        # The first call should be for X_train, which should have 8 rows.
        # The second call should be for X_val, which should have 2 rows.
        
        # Note: This is a simplified test. A more thorough test would inspect
        # the contents of the calls to `to_csv` to verify the shapes and contents
        # of the split data.

if __name__ == '__main__':
    unittest.main()
