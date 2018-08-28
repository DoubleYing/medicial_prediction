import unittest
from xgboost_sklearn import *


class XgboostSklearnTest(unittest.TestCase):
    def test_vote(self):
        y_preds = np.array([[1, 0, 1], [1, 1, 0]])
        re = vote(y_preds)
        print(re)
