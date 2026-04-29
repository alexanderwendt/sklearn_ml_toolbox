import numpy as np
import pandas as pd
import unittest

from src.data_processing.step30_clean_raw_data import unique_cols

class TestCleanRawData(unittest.TestCase):

    def test_unique_cols(self):
        """
        Test the unique_cols function to ensure it correctly identifies
        columns with unique and non-unique values.
        """
        # DataFrame with a non-unique column
        data_non_unique = {'A': [1, 1, 1, 1], 'B': [1, 2, 3, 4]}
        df_non_unique = pd.DataFrame(data_non_unique)
        self.assertFalse(unique_cols(df_non_unique[['A']]))
        
        # DataFrame with all unique columns
        data_unique = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
        df_unique = pd.DataFrame(data_unique)
        self.assertTrue(unique_cols(df_unique))

if __name__ == '__main__':
    unittest.main()
