"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

import os.path

import numpy as np

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             GroupSparseCovarianceCV)
from nilearn.tests.test_group_sparse_covariance \
        import generate_multi_task_gg_model
import utils  # defines profile() if not already defined


def cache_array(arr, filename, decimal=7):
    assert filename.endswith(".npy")
    if os.path.isfile(filename):
        cached = np.load(filename)
        np.testing.assert_almost_equal(cached, arr, decimal=decimal)
    else:
        np.save(filename, arr)


def benchmark1():
    rho = .2
    tol = 1e-5
    max_iter = 50

    signals, _, _ = generate_multi_task_gg_model(
        n_tasks=40, n_var=30, density=0.15, min_samples=100, max_samples=150,
        rand_gen=np.random.RandomState(0))

    cache_array(signals[0], "tmp/signals_0.npy")

    _, est_precs = utils.timeit(group_sparse_covariance)(
        signals, rho, max_iter=max_iter, tol=tol, verbose=0, debug=False)

    cache_array(est_precs, "tmp/est_precs.npy", decimal=4)


def benchmark2():
    tol = 1e-4
    max_iter = 50
    n_var = 10

    signals, _, _ = generate_multi_task_gg_model(
        n_tasks=40, n_var=n_var, density=0.15,
        min_samples=100, max_samples=150,
        rand_gen=np.random.RandomState(0))

    cache_array(signals[0], "tmp/signals_cv_0_%d.npy" % n_var)

    gsc = GroupSparseCovarianceCV(rhos=4, max_iter=max_iter, tol=tol,
                                  verbose=1, debug=False)
    utils.timeit(gsc.fit)(signals)
    print(gsc.rho_)
    cache_array(gsc.precisions_, "tmp/est_precs_cv_%d.npy" % n_var, decimal=3)

    ## import pylab as pl
    ## pl.matshow(est_precs[..., 0])
    ## pl.show()

if __name__ == "__main__":
    benchmark2()
