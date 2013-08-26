"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

import utils  # defines profile() if not already defined

import os.path
import time

import numpy as np
import pylab as pl

import joblib
from sklearn.covariance import ledoit_wolf

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             group_sparse_score,
                                             empirical_covariances, rho_max,
                                             GroupSparseCovarianceCV)
import nilearn._utils.testing as testing


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
    precisions, topology = testing.generate_sparse_precision_matrices(
        n_tasks=parameters["n_tasks"],
        n_var=parameters["n_var"],
        density=parameters["density"], rand_gen=rand_gen)

    signals = testing.generate_signals_from_precisions(
        precisions, min_samples=min_samples, max_samples=max_samples,
        rand_gen=rand_gen)

    return signals, precisions, topology


def benchmark1():
    parameters = {'n_tasks': 40, 'n_var': 30, 'density': 0.15,
                  'rho': .01, 'tol': 1e-4, 'max_iter': 40}

    signals, _, _ = generate_signals(parameters)
    cache_array(signals[0], "tmp/benchmark1_signals_0.npy")

    _, est_precs = utils.timeit(group_sparse_covariance)(
        signals, parameters['rho'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], verbose=1, debug=False)

    cache_array(est_precs, "tmp/benchmark1_est_precs.npy", decimal=4)


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


class ScoreProbe(object):
    def __init__(self, comment=""):
        self.comment = comment
        self.score = []
        self.objective = []
        self.timings = []
        self.start_time = 0
        self.max_norm = []
        self.l1_norm = []

    def __call__(self, emp_covs, n_samples, rho, max_iter, tol, n, omega,
                 omega_diff):
        """Probe for group_sparse_covariance that returns times and scores"""
        if n == -1:
            print("\n-- probe: starting '{0}' --".format(str(self.comment)))
            self.start_time = time.time()
            self.timings.append(0)
        else:
            self.timings.append(time.time() - self.start_time)
            self.max_norm.append(abs(omega_diff).max())
            self.l1_norm.append(abs(omega_diff).sum() / omega.size)

        score, objective = group_sparse_score(omega, n_samples, emp_covs, rho)
        self.score.append(score)
        self.objective.append(objective)
        ## if n == 5:
        ##     return True

    def plot(self):
        pl.figure()
        pl.plot(self.timings, self.score, "+-", label="score")
        pl.plot(self.timings, self.objective, "+-", label="objective")
        pl.xlabel("Time [s]")
        pl.grid()
        pl.legend(loc="best")
        pl.title(str(self.comment))

        pl.figure()
        pl.semilogy(self.timings[1:], self.max_norm, "+-", label="max norm")
        pl.semilogy(self.timings[1:], self.l1_norm, "+-", label="l1 norm")
        pl.xlabel("Time [s]")
        pl.ylabel("norm of difference")
        pl.grid()
        pl.legend(loc="best")
        pl.title(str(self.comment))


class MemProbe(object):
    def __init__(self, comment=""):
        self.comment = comment
        self.timings = []
        self.start_time = 0
        self.precisions = []

    def __call__(self, emp_covs, n_samples, rho, max_iter, tol, n, omega,
                 omega_diff):
        """Probe for group_sparse_covariance that returns times and scores"""
        if n == -1:
            print("\n-- probe: starting '{0}' --".format(str(self.comment)))
            self.start_time = time.time()
            self.timings.append(0)
        else:
            self.timings.append(time.time() - self.start_time)

        self.precisions.append(omega.copy())


def modified_gsc(signals, parameters, probe=None):
    """Modified group_sparse_covariance, just for joblib wrapping.
    """

    _, est_precs = utils.timeit(group_sparse_covariance)(
        signals, parameters['rho'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], probe_function=probe,
        precisions_init=parameters.get("precisions_init", None),
        verbose=1, debug=False)

    return est_precs, probe


def benchmark3():
    ## parameters = {'n_tasks': 10, 'n_var': 50, 'density': 0.15,
    ##               'rho': .001, 'tol': 1e-2, 'max_iter': 100}
    parameters = {'n_tasks': 10, 'n_var': 50, 'density': 0.15,
                  'rho': .001, 'tol': 1e-2, 'max_iter': 100}

    mem = joblib.Memory(".")

    signals, _, _ = generate_signals(parameters)
    # cache_array(signals[0], "tmp/benchmark3_signals_0.npy")

    emp_covs, n_samples, _, _ = empirical_covariances(signals)
    print("rho_max: " + str(rho_max(emp_covs, n_samples)))

    # With diagonal elements initialization
    probe1 = ScoreProbe()
    est_precs1, probe1 = mem.cache(modified_gsc)(signals, parameters, probe1)
    probe1.comment = "diagonal"  # for joblib to ignore this value
    probe1.plot()

    # With Ledoit-Wolf initialization
    ld = np.empty(emp_covs.shape)
    for k in range(emp_covs.shape[-1]):
        ld[..., k] = np.linalg.inv(ledoit_wolf(signals[k])[0])

    probe1 = ScoreProbe()
    est_precs1, probe1 = utils.timeit(mem.cache(modified_gsc))(
        signals, parameters, probe=probe1)
    probe1.comment = "diagonal"  # for joblib to ignore this value

    probe2 = ScoreProbe()
    parameters["precisions_init"] = ld
    est_precs2, probe2 = utils.timeit(mem.cache(modified_gsc))(
        signals, parameters, probe=probe2)
    probe2.comment = "ledoit-wolf"

    print(abs(est_precs1 - est_precs2).max())

    ## probe1.plot()
    ## probe2.plot()

    pl.figure()
    pl.semilogy(probe1.timings[1:], probe1.max_norm,
                "+-", label=probe1.comment)
    pl.semilogy(probe2.timings[1:], probe2.max_norm,
                "+-", label=probe2.comment)
    pl.xlabel("Time [s]")
    pl.ylabel("Max norm")
    pl.grid()
    pl.legend(loc="best")

    pl.figure()
    pl.plot(probe1.timings, probe1.objective,
                "+-", label=probe1.comment)
    pl.plot(probe2.timings, probe2.objective,
                "+-", label=probe2.comment)
    pl.xlabel("Time [s]")
    pl.ylabel("objective")
    pl.grid()
    pl.legend(loc="best")

    pl.show()


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
    benchmark1()
    ## lasso_gsc_comparison()
    ## singular_cov_case()
