"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

import os.path

import numpy as np

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             GroupSparseCovarianceCV)
import nilearn.testing
import utils  # defines profile() if not already defined


def cache_array(arr, filename, decimal=7):
    assert filename.endswith(".npy")
    if os.path.isfile(filename):
        cached = np.load(filename)
        np.testing.assert_almost_equal(cached, arr, decimal=decimal)
    else:
        np.save(filename, arr)


def generate_signals(parameters):
    rand_gen = parameters.get('rand_gen', np.random.RandomState(0))
    min_samples = parameters.get('min_samples', 100)
    max_samples = parameters.get('max_samples', 150)

    # Generate signals
    precisions, topology = \
                nilearn.testing.generate_sparse_precision_matrices(
        n_tasks=parameters["n_tasks"],
        n_var=parameters["n_var"],
        density=parameters["density"], rand_gen=rand_gen)

    signals = nilearn.testing.generate_signals_from_precisions(
        precisions, min_samples=min_samples, max_samples=max_samples,
        rand_gen=rand_gen)

    return signals, precisions, topology


def benchmark1():
    parameters = {'n_tasks': 40, 'n_var': 30, 'density': 0.15,
                  'rho': .2, 'tol': 1e-5, 'max_iter': 50}

    signals, _, _ = generate_signals(parameters)

    cache_array(signals[0], "tmp/signals_0.npy")

    _, est_precs = utils.timeit(group_sparse_covariance)(
        signals, parameters['rho'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], verbose=0, debug=False)

    cache_array(est_precs, "tmp/est_precs.npy", decimal=4)


def benchmark2():
    parameters = {'n_tasks': 40, 'n_var': 10, 'density': 0.15,
                  'rhos': 4, 'tol': 1e-4, 'max_iter': 50}

    signals, _, _ = generate_signals(parameters)

    cache_array(signals[0],
                "tmp/signals_cv_0_{n_var:d}.npy".format(**parameters))

    gsc = GroupSparseCovarianceCV(rhos=parameters['rhos'],
                                  max_iter=parameters['max_iter'],
                                  tol=parameters['tol'],
                                  verbose=1, debug=False)
    utils.timeit(gsc.fit)(signals)
    print(gsc.rho_)
    cache_array(gsc.precisions_,
                "tmp/est_precs_cv_{n_var:d}.npy".format(**parameters),
                decimal=3)

    ## import pylab as pl
    ## pl.matshow(est_precs[..., 0])
    ## pl.show()

if __name__ == "__main__":
    benchmark2()
