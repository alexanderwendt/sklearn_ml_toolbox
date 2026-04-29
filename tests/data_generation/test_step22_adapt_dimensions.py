import numpy as np
import pandas as pd
import unittest

from src.data_generation.step22_adapt_dimensions import clean_nan

class TestAdaptDimensions(unittest.TestCase):

    def test_clean_nan(self):
        """
        Test the clean_nan function to ensure it correctly removes rows
        with NaN values from a DataFrame.
        """
        data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]}
        df = pd.DataFrame(data)
        
        expected_data = {'A': [1.0, 4.0], 'B': [5.0, 8.0]}
        expected_df = pd.DataFrame(expected_data, index=[0, 3])
        
        result_df = clean_nan(df)
        
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()
