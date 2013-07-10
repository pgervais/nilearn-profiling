"""Launch groups_sparse_covariance and record output for varying values of
the regularization parameter rho.
"""
# Authors: Philippe Gervais
# License: simplified BSD

import itertools

import numpy as np

import joblib

from nilearn.group_sparse_covariance import (rho_max, empirical_covariances)

from common import create_signals, save_group_sparse_covariance

output_dir = "gsc_varying_rho"


def benchmark1():
    parameters = dict(n_var=20,
                      n_tasks=5,
                      density=0.15,

                      tol=-1,
                      n_rhos=10,
                      max_iter=5,
                      min_samples=100,
                      max_samples=150)

    next_num, cache_dir, gt = create_signals(parameters, output_dir=output_dir)

    emp_covs, n_samples, _, _ = empirical_covariances(gt['signals'])
    max_rho = rho_max(emp_covs, n_samples)

    rhos = np.logspace(np.log10(max_rho / 1000.), np.log10(max_rho),
                       parameters['n_rhos'])[::-1]
    ## rhos = np.logspace(np.log10(max_rho / 10000.), np.log10(max_rho / 100),
    ##                    parameters['n_rhos'])[::-1]

    joblib.Parallel(n_jobs=2, verbose=1)(
        joblib.delayed(save_group_sparse_covariance)(
            emp_covs, n_samples, rho, max_iter=parameters['max_iter'],
            tol=parameters['tol'],
            cache_dir=cache_dir, num=num)
        for rho, num in zip(rhos, itertools.count(next_num)))

if __name__ == '__main__':
    benchmark1()
