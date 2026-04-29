import numpy as np
import unittest

from src.data_generation.step20_generate_groundtruth_stockmarket import define_tops_bottoms

class TestGenerateGroundtruth(unittest.TestCase):

    def test_define_tops_bottoms(self):
        """
        Test the define_tops_bottoms function to ensure it correctly merges
        top and bottom signals into a single array.
        """
        bottoms = np.array([0, 1, 0, 0, 1, 0])
        tops = np.array([1, 0, 0, 1, 0, 0])
        
        # Expected output: tops are 1, bottoms are 2
        expected_output = np.array([1, 2, 0, 1, 2, 0])
        
        result = define_tops_bottoms(bottoms, tops)
        
        np.testing.assert_array_equal(result, expected_output)

if __name__ == '__main__':
    unittest.main()
