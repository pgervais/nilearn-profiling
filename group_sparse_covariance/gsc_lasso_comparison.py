"""Some test functions for antonio-samaras algorithm implementation."""

import utils  # defines profile() if not already defined

import numpy as np

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             empirical_covariances)
from common import create_signals


def lasso_gsc_comparison():
    """Check that graph lasso and group-sparse covariance give the same
    output for a single task."""
    from sklearn.covariance import graph_lasso, empirical_covariance

    parameters = {'n_tasks': 1, 'n_var': 20, 'density': 0.15,
                  'rho': .2, 'tol': 1e-4, 'max_iter': 50}

    _, _, gt = create_signals(parameters)
    signals = gt["signals"]

    _, gsc_precision = utils.timeit(group_sparse_covariance)(
        signals, parameters['rho'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], verbose=1, debug=False)

    emp_cov = empirical_covariance(signals[0])
    _, gl_precision = utils.timeit(graph_lasso)(
        emp_cov, parameters['rho'], tol=parameters['tol'],
        max_iter=parameters['max_iter'])

    np.testing.assert_almost_equal(gl_precision, gsc_precision[..., 0],
                                   decimal=4)


def singular_cov_case():
    """Check behaviour of algorithm for singular input matrix."""
    parameters = {'n_tasks': 10, 'n_var': 40, 'density': 0.15,
                  'rho': .1, 'tol': 1e-2, 'max_iter': 50,
                  'min_samples': 10, 'max_samples': 15}

    _, _, gt = create_signals(parameters)
    signals = gt["signals"]

    emp_covs, _ = empirical_covariances(signals)

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
    lasso_gsc_comparison()
    singular_cov_case()
