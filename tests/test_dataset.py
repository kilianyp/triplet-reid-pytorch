import sys                                                                                       
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


import unittest
from csv_dataset import *

class TestMarket(unittest.TestCase):
    def test_make_dataset(self):
        csv_file = "~/Projects/cupsizes/data/market1501_train.csv"
        data_dir = "~/Projects/triplet-reid-pytorch/datasets/Market-1501"
        limit = 200
        data = make_dataset(csv_file, data_dir, limit)

        self.assertEqual(len(data), limit)
 
if __name__ == "__main__":
    unittest.main()
