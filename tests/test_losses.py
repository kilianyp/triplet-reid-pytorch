import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import triplet_loss as l
import unittest
import numpy as np
import scipy.spatial as sp
import torch
from torch.autograd import Variable


class TestCdist(unittest.TestCase):
    def test_cdist_different(self):
        a = np.random.randn(10, 20)
        b = np.random.randn(30, 20)

        for metric in ('euclidean', 'sqeuclidean', 'cityblock'):
            D_my = l.calc_cdist(torch.from_numpy(a), torch.from_numpy(b), metric).numpy()
            D_sp = sp.distance.cdist(a, b, metric)
            np.testing.assert_allclose(D_my, D_sp, rtol=1e-5, atol=1e-5)

    def test_cdist_same(self):
        a = np.random.randn(10, 20)
        for metric in ('euclidean', 'sqeuclidean', 'cityblock'):
            D_my = l.calc_cdist(torch.from_numpy(a), torch.from_numpy(a), metric).numpy()
            D_sp = sp.distance.cdist(a, a, metric)
            np.testing.assert_allclose(D_my, D_sp, rtol=1e-5, atol=1e-5)

class TestBatchHard(unittest.TestCase):
    def test_batch_hard(self):
        pids = np.array([0, 0, 1, 0, 1], dtype=np.float32)
        features = np.array([
            [5.0],
            [6.0],
            [1.0],
            [7.0],
            [9.5],
        ], np.float32)

        pids = Variable(torch.from_numpy(pids))
        features = Variable(torch.from_numpy(features))

        loss_fn = l.BatchHard("none")
        loss = loss_fn(l.calc_cdist(features, features), pids)

        result = np.array([2.0 - 4.0, 1.0 - 3.5, 8.5 - 4.0, 2.0 - 2.5, 8.5 - 2.5], dtype=np.float32)
        np.testing.assert_array_equal(result, loss.data.numpy())


if __name__ == '__main__':
    unittest.main()
