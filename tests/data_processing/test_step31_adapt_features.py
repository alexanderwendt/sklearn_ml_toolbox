import numpy as np
import pandas as pd
import unittest
import configparser

from src.data_processing.step31_adapt_features import adapt_features_for_model

class TestAdaptFeatures(unittest.TestCase):

    def test_adapt_features_for_model_numeric_conversion(self):
        """
        Test that adapt_features_for_model correctly converts feature columns to numeric types.
        """
        features_raw = pd.DataFrame({
            'A': ['1', '2', '3'],
            'B': ['4.0', '5.5', '6.1']
        })
        
        # Create a mock config
        config = configparser.ConfigParser()
        config['Common'] = {'problem_type': 'classification'}
        config['Preparation'] = {'binarize_labels': 'False'}
        
        features, _, _ = adapt_features_for_model(features_raw, None, None, None, config)
        
        self.assertTrue(pd.api.types.is_numeric_dtype(features['A']))
        self.assertTrue(pd.api.types.is_numeric_dtype(features['B']))

if __name__ == '__main__':
    unittest.main()
