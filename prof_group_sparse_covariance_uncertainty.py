"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

import os.path
import time
import numpy as np
import sys
import joblib
import pylab as pl

import nilearn
from nilearn.group_sparse_covariance import (_group_sparse_covariance, rho_max,
                                             empirical_covariances,
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
    tol = 1e-4
    max_iter = 50
    n_var = 10
    rho = 0.01
    n_tols = 8
    tols_m = 1e-2
    trials = 10

    rand_gen = np.random.RandomState(0)

    mem = joblib.Memory(cachedir="gsc")

    signals, _, _ = mem.cache(generate_multi_task_gg_model)(
        n_tasks=40, n_var=n_var, density=0.15,
        min_samples=100, max_samples=150,
        rand_gen=rand_gen)

    emp_covs, n_samples, _, _ = empirical_covariances(signals)

    tols = np.logspace(np.log10(tols_m), np.log10(tols_m / 10000.),
                       n_tols)[::-1]

    non_zero = []
    uncertainty = []
    all_costs = []
    for tol in tols:
        precisions_init = None
        all_precisions = np.empty(emp_covs.shape + (trials, ))

        trial_costs = []
        for n in xrange(trials):
            precisions, costs = mem.cache(_group_sparse_covariance)(
                emp_covs, n_samples, rho, max_iter=max_iter, tol=tol,
                return_costs=True, verbose=1, debug=False,
                precisions_init=precisions_init)
            non_zero.append((precisions[..., 0] != 0).sum())
            all_precisions[..., n] = precisions
            trial_costs.append(costs[-1][-1])  # append final duality gap

            if precisions_init is None:
                precisions_init = precisions.copy()

            for k in xrange(precisions_init.shape[-1]):
                noise = np.triu(
                    0.1 * (rand_gen.rand(*precisions.shape[:-1]) - 0.5), k=1)
                precisions_init[..., k] = precisions[..., k] + noise + noise.T

                assert nilearn.testing.is_spd(precisions_init[..., k])

        all_costs.append(np.mean(trial_costs))
        all_var = all_precisions.var(axis=-1)
        all_mean = all_precisions.mean(axis=-1)

        uncertainty2 = 0
        for k in xrange(all_mean.shape[-1]):
            inv_prec = np.linalg.inv(all_mean[..., k])
            uncertainty2 += n_samples[k] * np.trace(np.dot(
                np.dot(
                    np.dot(inv_prec, all_var[..., k]),
                    inv_prec),
                all_var[..., k]))
        uncertainty2 /= 2
        uncertainty2 = np.sqrt(uncertainty2)
        uncertainty.append(uncertainty2)

    costs = all_costs
    print (costs)
    print("rho\testimated uncertainty\ttolerance")

    for uncert, tol in zip(uncertainty, tols):
        print("%e\t%e\t%e" % (rho, uncert, tol))

    pl.figure()
    pl.plot(tols, tols, label="tolerance")
    pl.plot(tols, uncertainty, label="estimated")
    pl.plot(tols, costs, label="duality gap")
    pl.xlabel("tols")
    pl.ylabel("uncertainty")
    pl.legend(loc=0)
    pl.grid()

    pl.figure()
    pl.loglog(tols, tols, label="tolerance")
    pl.loglog(tols, uncertainty, label="estimated")
    pl.loglog(tols, costs, label="duality gap")
    pl.xlabel("tols")
    pl.ylabel("uncertainty")
    pl.legend(loc=0)
    pl.grid()

    pl.show()

if __name__ == "__main__":
    benchmark1()
