 """Benchmark for nilearn implementation of detrend."""
import sys
import time
import numpy as np
import utils
import nilearn.signal
import scipy

import utils  # defines profile() if not already defined


def benchmark(order=None):
    """Run nilearn.signal._detrend"""
    shape = (201, 200000)
    print ("Running for %s order..." % order)
    rand_gen = np.random.RandomState(0)
    series = np.ndarray(shape, order=order)
    series[...] = rand_gen.randn(*shape)
    output1 = utils.timeit(profile(nilearn.signal._detrend))(series)
    time.sleep(0.5)  # For memory_profiler
    del output1

    output2 = utils.timeit(profile(scipy.signal.detrend))(series, axis=0)
    time.sleep(0.5)  # For memory_profiler
    del output2
#    np.testing.assert_almost_equal(output1, output2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        orders = [sys.argv[-1]]
    else:
        orders = ['C', 'F']

    for order in orders:
        benchmark(order)
