"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices.

Estimation of matrix coefficients uncertainty by random sampling of starting
points.
"""

# Authors: Philippe Gervais
# License: simplified BSD

import joblib

from nilearn.group_sparse_covariance import (compute_alpha_max,
                                             empirical_covariances)
from common import create_signals, save_group_sparse_covariance


def get_series(params, keys):
    ret = [[p[k] for p in params] for k in keys]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def sample_precision_space(parameters, number=100):
    """Launch a large number of times the same estimation, with different
    starting points.

    number: int
        number of samples to generate.
    """
    # Estimation
    max_iter = 200

    # Generate signals
    next_num, cache_dir, gt = create_signals(parameters,
                                             output_dir="gsc_sensitivity")
    precisions, topology, signals = (gt["precisions"], gt["topology"],
                                     gt["signals"])

    emp_covs, n_samples = empirical_covariances(signals)

    print("alpha max: %.3e" % compute_alpha_max(emp_covs, n_samples)[0])

    # Estimate a lot of precision matrices
    parameters = joblib.Parallel(n_jobs=7, verbose=1)(
        joblib.delayed(save_group_sparse_covariance)(
            emp_covs, n_samples, parameters["alpha"], max_iter=max_iter,
            tol=parameters["tol"], cache_dir=cache_dir, num=n)
        for n in xrange(next_num, next_num + number))


if __name__ == "__main__":
    ## sample_precision_space({"n_var": 50, "n_tasks": 40, "density": 0.1,
    ##                         "tol": 1e-2, "alpha": 0.02})
    sample_precision_space({"n_var": 100, "n_tasks": 40, "density": 0.1,
                            "tol": 1e-2, "alpha": 0.02}, number=5)
