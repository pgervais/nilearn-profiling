"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

#import utils  # defines profile() if not already defined

import numpy as np
import pylab as pl

import joblib

from nilearn.group_sparse_covariance import (_group_sparse_covariance,
                                             compute_alpha_max,
                                             empirical_covariances,
                                             group_sparse_scores)

from nilearn._utils import testing


def group_sparse_covariance(emp_covs, n_samples, alpha, max_iter, tol):
    precisions = _group_sparse_covariance(
        emp_covs, n_samples, alpha, max_iter=max_iter, tol=tol,
        verbose=1, debug=False)

    return (dict(n_samples=n_samples, alpha=alpha, max_iter=max_iter, tol=tol,
                 precisions=precisions))


def get_series(params, keys):
    ret = [[p[k] for p in params] for k in keys]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def benchmark1():
    """Plot different quantities for varying alpha."""
    # Signals
    min_samples, max_samples = 100, 150  # train signals length
    n_var = 50
    n_tasks = 40
    density = 0.1
    random_state = np.random.RandomState(0)

    test_samples = 4000  # number of samples for test signals

    # Estimation
    n_alphas = 10
    max_iter = 200
    tol = 1e-3

    # Generate signals
    signals, precisions, topology = \
             testing.generate_group_sparse_gaussian_graphs(
        n_subjects=n_tasks, n_features=n_var, density=density,
        random_state=random_state, min_n_samples=min_samples,
        max_n_samples=max_samples)

    emp_covs, n_samples = empirical_covariances(signals)

    # Estimate precision matrices
    alpha_1, _ = compute_alpha_max(emp_covs, n_samples)
    alpha_0 = 1e-2 * alpha_1
    ## alpha_1 = 0.067
    ## alpha_0 = 0.044

    alphas = np.logspace(np.log10(alpha_0), np.log10(alpha_1), n_alphas)[::-1]

    parameters = joblib.Parallel(n_jobs=7, verbose=1)(
        joblib.delayed(group_sparse_covariance)(emp_covs, n_samples, alpha,
                                                max_iter=max_iter, tol=tol)
        for alpha in alphas)

    # Compute scores
    test_signals = testing.generate_signals_from_precisions(
        precisions, min_samples=test_samples, max_samples=test_samples + 1,
        rand_gen=random_state)

    test_emp_covs, _ = empirical_covariances(test_signals)
    del test_signals

    for params in parameters:
        params["ll_score"], params["pen_score"] = group_sparse_scores(
            params["precisions"], n_samples, test_emp_covs, params["alpha"])

    # Plot graphs
    alpha, ll_score, pen_score = get_series(
        parameters, ("alpha", "ll_score", "pen_score"))
    non_zero = [(p["precisions"][..., 0] != 0).sum() for p in parameters]

    pl.figure()
    pl.semilogx(alpha, ll_score, "-+", label="log-likelihood")
    pl.semilogx(alpha, pen_score, "-+", label="penalized LL")
    pl.xlabel("alpha")
    pl.ylabel("score")
    pl.grid()

    pl.figure()
    pl.semilogx(alpha, non_zero, "-+")
    pl.xlabel("alpha")
    pl.ylabel("non_zero")
    pl.grid()

    pl.figure()
    pl.loglog(alpha, non_zero, "-+")
    pl.xlabel("alpha")
    pl.ylabel("non_zero")
    pl.grid()

    pl.figure()
    pl.imshow(topology, interpolation="nearest")
    pl.title("true topology")

    ## precisions = get_series(parameters, ("precisions", ))
    ## for prec, alpha in zip(precisions, alpha):
    ##     pl.figure()
    ##     pl.imshow(prec[..., 0] != 0, interpolation="nearest")
    ##     pl.title(alpha)

    pl.show()

if __name__ == "__main__":
    benchmark1()
