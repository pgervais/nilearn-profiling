"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

#import utils  # defines profile() if not already defined

import numpy as np
import pylab as pl

import joblib

from nilearn.group_sparse_covariance import (_group_sparse_covariance, rho_max,
                                             empirical_covariances,
                                             group_sparse_score)

from nilearn._utils import testing


def group_sparse_covariance(emp_covs, n_samples, rho, max_iter, tol):

    precisions, costs = _group_sparse_covariance(
        emp_covs, n_samples, rho, max_iter=max_iter, tol=tol,
        return_costs=True, verbose=1, debug=False)

    return (dict(n_samples=n_samples, rho=rho, max_iter=max_iter, tol=tol,
                 objective=costs[-1][0], duality_gap=costs[-1][1],
                 precisions=precisions))


def get_series(params, keys):
    ret = [[p[k] for p in params] for k in keys]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def benchmark1():
    """Plot different quantities for varying rho."""
    # Signals
    min_samples, max_samples = 100, 150  # train signals length
    n_var = 50
    n_tasks = 40
    density = 0.1
    rand_gen = np.random.RandomState(0)

    test_samples = 4000  # number of samples for test signals

    # Estimation
    n_rhos = 10
    max_iter = 200
    tol = 1e-3

    # Generate signals
    precisions, topology = testing.generate_sparse_precision_matrices(
        n_tasks=n_tasks, n_var=n_var, density=density, rand_gen=rand_gen)
    signals = testing.generate_signals_from_precisions(
        precisions, min_samples=min_samples, max_samples=max_samples,
        rand_gen=rand_gen)

    emp_covs, n_samples, _, _ = empirical_covariances(signals)

    # Estimate precision matrices
    rho_1 = rho_max(emp_covs, n_samples)
    rho_0 = 1e-2 * rho_1
    ## rho_1 = 0.067
    ## rho_0 = 0.044

    rhos = np.logspace(np.log10(rho_0), np.log10(rho_1), n_rhos)[::-1]

    parameters = joblib.Parallel(n_jobs=7, verbose=1)(
        joblib.delayed(group_sparse_covariance)(emp_covs, n_samples, rho,
                                                max_iter=max_iter, tol=tol)
        for rho in rhos)

    # Compute scores
    test_signals = testing.generate_signals_from_precisions(
        precisions, min_samples=test_samples, max_samples=test_samples + 1,
        rand_gen=rand_gen)

    test_emp_covs, _, _, _ = empirical_covariances(test_signals)
    del test_signals

    for params in parameters:
        params["ll_score"], params["pen_score"] = group_sparse_score(
            params["precisions"], n_samples, test_emp_covs, params["rho"])

    # Plot graphs
    rho, ll_score, pen_score = get_series(
        parameters, ("rho", "ll_score", "pen_score"))
    non_zero = [(p["precisions"][..., 0] != 0).sum() for p in parameters]

    pl.figure()
    pl.semilogx(rho, ll_score, "-+", label="log-likelihood")
    pl.semilogx(rho, pen_score, "-+", label="penalized LL")
    pl.xlabel("rho")
    pl.ylabel("score")
    pl.grid()

    pl.figure()
    pl.semilogx(rho, non_zero, "-+")
    pl.xlabel("rho")
    pl.ylabel("non_zero")
    pl.grid()

    pl.figure()
    pl.loglog(rho, non_zero, "-+")
    pl.xlabel("rho")
    pl.ylabel("non_zero")
    pl.grid()

    pl.figure()
    pl.imshow(topology, interpolation="nearest")
    pl.title("true topology")

    ## precisions = get_series(parameters, ("precisions", ))
    ## for prec, rho in zip(precisions, rho):
    ##     pl.figure()
    ##     pl.imshow(prec[..., 0] != 0, interpolation="nearest")
    ##     pl.title(rho)

    pl.show()

if __name__ == "__main__":
    benchmark1()
