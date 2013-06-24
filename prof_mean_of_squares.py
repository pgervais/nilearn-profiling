"""Benchmark for nisl implementation of detrend."""
import sys
import time
import numpy as np
import utils
import nisl.signal
import scipy

import utils  # defines profile() if not already defined


def ref_mean_of_squares(series):
    var = np.copy(series)
    var **= 2
    return var.mean(axis=0)


def benchmark(order=None):
    """Run nisl.signal._detrend"""
    shape = (201, 200001)
    print ("Running for %s order..." % order)
    rand_gen = np.random.RandomState(0)
    series = np.ndarray(shape, order=order)
    series[...] = rand_gen.randn(*shape)

    output1 = utils.timeit(profile(nisl.signal._mean_of_squares))(series)
    time.sleep(0.5)  # For memory_profiler
#    del output1

    output2 = utils.timeit(profile(ref_mean_of_squares))(series)
    time.sleep(0.5)  # For memory_profiler
#    del output2
    np.testing.assert_almost_equal(output1, output2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        orders = [sys.argv[-1]]
    else:
        orders = ['C', 'F']

    for order in orders:
        benchmark(order)
