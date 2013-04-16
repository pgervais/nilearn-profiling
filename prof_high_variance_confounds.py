import sys
import time
import numpy as np
import utils
import nisl.signals


# profile() is defined by most profilers, these lines allows running
# the script without any profiler.
try:
    profile
except NameError:
    def profile(func):
        return func


def benchmark(order=None):
    """Run nisl.signals.high_variance_confounds"""
    shape = (201, 200000)
    if order == "C" or order is None:
        print ("Running for C order...")
        rand_gen = np.random.RandomState(0)
        series = np.ndarray(shape, order="C")
        series[...] = rand_gen.randn(*shape)
        output = utils.timeit(profile(nisl.signals.high_variance_confounds)
                              )(series)
        time.sleep(0.5)  # For memory_profiler
        del output

    if order == "F" or order is None:
        print ("Running for F order...")
        rand_gen = np.random.RandomState(0)
        series = np.ndarray(shape, order="F")
        series[...] = rand_gen.randn(*shape)
        output = utils.timeit(profile(nisl.signals.high_variance_confounds)
                              )(series)
        time.sleep(0.5)
        del output

if __name__ == "__main__":
    if len(sys.argv) > 1:
        order = sys.argv[-1]
    else:
        order = None
    benchmark(order)
