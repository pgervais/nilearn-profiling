import os.path
import numpy as np


def cache_array(arr, filename, decimal=7):
    assert filename.endswith(".npy")
    if os.path.isfile(filename):
        cached = np.load(filename)
        np.testing.assert_almost_equal(cached, arr, decimal=decimal)
    else:
        np.save(filename, arr)


def random_spd(n, rand_gen=np.random.RandomState(1)):
    M = 0.1 * rand_gen.randn(n, n)
    return np.dot(M, M.T)
