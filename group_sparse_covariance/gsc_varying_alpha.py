"""Launch groups_sparse_covariance and record output for varying values of
the regularization parameter alpha.
"""
# Authors: Philippe Gervais
# License: simplified BSD

import itertools

import numpy as np
import joblib

from nilearn.group_sparse_covariance import (compute_alpha_max,
                                             empirical_covariances)

from common import create_signals, save_group_sparse_covariance

output_dir = "gsc_varying_alpha"


def benchmark1():
    parameters = dict(n_var=100,
                      n_tasks=5,
                      density=0.15,

                      tol=1e-2,
                      n_alphas=5,
                      max_iter=50,
                      min_samples=100,
                      max_samples=150)

    next_num, cache_dir, gt = create_signals(parameters, output_dir=output_dir)

    emp_covs, n_samples = empirical_covariances(gt['signals'])
    max_alpha, _ = compute_alpha_max(emp_covs, n_samples)

    min_alpha = max_alpha / 100.
    print(min_alpha, max_alpha)
    alphas = np.logspace(np.log10(min_alpha), np.log10(max_alpha / 50.),
                       parameters['n_alphas'])[::-1]

    joblib.Parallel(n_jobs=1, verbose=1)(
        joblib.delayed(save_group_sparse_covariance)(
            emp_covs, n_samples, alpha, max_iter=parameters['max_iter'],
            tol=parameters['tol'], debug=True, cache_dir=cache_dir, num=num)
        for alpha, num in zip(alphas, itertools.count(next_num)))

if __name__ == '__main__':
    benchmark1()
