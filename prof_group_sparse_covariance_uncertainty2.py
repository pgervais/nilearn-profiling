"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

import utils  # defines profile() if not already defined
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
    n_rhos = 16

    signals, _, _ = generate_multi_task_gg_model(
        n_tasks=40, n_var=n_var, density=0.15,
        min_samples=100, max_samples=150,
        rand_gen=np.random.RandomState(0))

    emp_covs, n_samples, _, _ = empirical_covariances(signals)
    rho_m = rho_max(emp_covs, n_samples)
    rhos = np.logspace(np.log10(rho_m), np.log10(rho_m / 100.),
                       n_rhos)[::-1]

    exec_time = []
    non_zero = []
    for rho in rhos:
        t0 = time.time()
        precisions, costs = utils.timeit(_group_sparse_covariance)(
            emp_covs, n_samples, rho, max_iter=max_iter, tol=tol,
            return_costs=True, verbose=1, debug=False)
        exec_time.append(time.time() - t0)
        non_zero.append((precisions[..., 0] != 0).sum())

    pl.figure()
    pl.plot(non_zero, exec_time, "-+")
    pl.grid()
    pl.xlabel("number of non-zero elements")
    pl.ylabel("execution time [s]")

    pl.figure()
    pl.loglog(rhos, exec_time, "-+")
    pl.grid()
    pl.xlabel("regularization parameter")
    pl.ylabel("execution time [s]")

    pl.figure()
    pl.loglog(rhos, non_zero, "-+")
    pl.grid()
    pl.xlabel("regularization parameter")
    pl.ylabel("number of non-zero elements")

    pl.show()


def benchmark2():
    tol = 1e-4
    max_iter = 50
    n_var = 10

    signals, _, topology = generate_multi_task_gg_model(
        n_tasks=40, n_var=n_var, density=0.15,
        min_samples=100, max_samples=150,
        rand_gen=np.random.RandomState(0))

    gsc = GroupSparseCovarianceCV(rhos=4, max_iter=max_iter, tol=tol,
                                  verbose=1, debug=False, n_jobs=3)
    utils.timeit(gsc.fit)(signals)
    print("selected rho: %f" % gsc.rho_)
    print("number of non-zero elements: %d" %
          (gsc.precisions_[..., 0] != 0).sum())
    print("True number of non-zero elements: %d" % topology.sum())


def benchmark3():
    tol = 1e-4
    max_iter = 50
    n_var = 10
    n_rhos = 15
    trials = 10

    rand_gen = np.random.RandomState(0)

    signals, _, _ = generate_multi_task_gg_model(
        n_tasks=40, n_var=n_var, density=0.15,
        min_samples=100, max_samples=150,
        rand_gen=rand_gen)

    emp_covs, n_samples, _, _ = empirical_covariances(signals)
    rho_m = rho_max(emp_covs, n_samples)
    rhos = np.logspace(np.log10(rho_m), np.log10(rho_m / 100.),
                       n_rhos)[::-1]

    mem = joblib.Memory(cachedir="gsc")

    non_zero = []
    uncertainty = []
    for rho in rhos:
        precisions_init = None
        all_precisions = np.empty(emp_covs.shape + (trials, ))

        for n in xrange(trials):
            precisions, costs = mem.cache(_group_sparse_covariance)(
                emp_covs, n_samples, rho, max_iter=max_iter, tol=tol,
                return_costs=True, verbose=1, debug=False,
                precisions_init=precisions_init)
            non_zero.append((precisions[..., 0] != 0).sum())
            all_precisions[..., n] = precisions

            if precisions_init is None:
                precisions_init = precisions.copy()

            for k in xrange(precisions_init.shape[-1]):
                noise = np.triu(
                    0.1 * (rand_gen.rand(*precisions.shape[:-1]) - 0.5), k=1)
                precisions_init[..., k] = precisions[..., k] + noise + noise.T

                assert nilearn.testing.is_spd(precisions_init[..., k])

        all_std = all_precisions.std(axis=-1)
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

    print("rho\testimated uncertainty\ttolerance")

    for uncert, rho in zip(uncertainty, rhos):
        print("%e\t%e\t%e" % (rho, uncert, tol))

    pl.figure()
    pl.plot(rhos, uncertainty)
    pl.xlabel("rhos")
    pl.ylabel("estimated uncertainty")
    pl.grid()

    pl.figure()
    pl.semilogy(rhos, uncertainty)
    pl.xlabel("rhos")
    pl.ylabel("estimated uncertainty")
    pl.grid()

    pl.show()
    sys.exit(0)

    for k in xrange(3):
        pl.matshow(all_std[..., k])
        pl.title("std deviations %d" % k)
        pl.colorbar()

        pl.matshow(all_precisions[..., k, 0])
        pl.title("precision %d" % k)
        pl.colorbar()

    print("mean std: " + str(np.mean(all_std)))
    print("median std: " + str(np.median(all_std)))
    print("max std: " + str(np.max(all_std)))
    print("min std: " + str(np.min(all_std)))

    pl.show()

if __name__ == "__main__":
    benchmark3()
