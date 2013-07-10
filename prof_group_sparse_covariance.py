"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

import os.path

import numpy as np

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             empirical_covariances,
                                             GroupSparseCovarianceCV)
import nilearn.testing
import utils  # defines profile() if not already defined


def cache_array(arr, filename, decimal=7):
    """Small caching function to check that some array has not changed
    between two invocations."""
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


def lasso_gsc_comparison():
    """Check that graph lasso and group-sparse covariance give the same
    output for a single task."""
    from sklearn.covariance import graph_lasso, empirical_covariance

    parameters = {'n_tasks': 1, 'n_var': 20, 'density': 0.15,
                  'rho': .2, 'tol': 1e-4, 'max_iter': 50}

    signals, _, _ = generate_signals(parameters)

    _, gsc_precision = utils.timeit(group_sparse_covariance)(
        signals, parameters['rho'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], verbose=1, debug=False)

    emp_cov = empirical_covariance(signals[0])
    _, gl_precision = utils.timeit(graph_lasso)(
        emp_cov, parameters['rho'], tol=parameters['tol'],
        max_iter=parameters['max_iter'])

    np.testing.assert_almost_equal(gl_precision, gsc_precision[..., 0],
                                   decimal=3)


def singular_cov_case():
    """Check behaviour of algorithm for singular input matrix."""
    parameters = {'n_tasks': 10, 'n_var': 40, 'density': 0.15,
                  'rho': .1, 'tol': 1e-2, 'max_iter': 50,
                  'min_samples': 10, 'max_samples': 15}

    signals, _, _ = generate_signals(parameters)

    emp_covs, _, _, _ = empirical_covariances(signals)

    # Check that all covariance matrices are singular.
    eps = np.finfo(float).eps
    for k in range(emp_covs.shape[-1]):
        eigvals = np.linalg.eigvalsh(emp_covs[..., k])
        assert(abs(eigvals.min()) <= 50 * eps)

    _, gsc_precisions = utils.timeit(group_sparse_covariance)(
        signals, parameters['rho'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], verbose=1, debug=False)

    print('found sparsity: {0:.3f}'
          ''.format(1. * (gsc_precisions[..., 0] != 0).sum()
                    / gsc_precisions.shape[0] ** 2))

if __name__ == "__main__":
    ## lasso_gsc_comparison()
    singular_cov_case()
