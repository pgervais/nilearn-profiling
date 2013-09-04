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
                                             group_sparse_scores,
                                             empirical_covariances,
                                             compute_alpha_max,
                                             GroupSparseCovarianceCV)

from common import create_signals


class ScoreProbe(object):
    def __init__(self, comment=""):
        self.comment = comment
        self.score = []
        self.objective = []
        self.timings = []
        self.start_time = 0
        self.max_norm = []
        self.l1_norm = []

    def __call__(self, emp_covs, n_samples, alpha, max_iter, tol, n, omega,
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

        score, objective = group_sparse_scores(omega, n_samples, emp_covs,
                                               alpha)
        self.score.append(score)
        self.objective.append(objective)

    def plot(self):
        pl.figure()
        pl.plot(self.timings, -np.asarray(self.score), "+-", label="score")
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


def modified_gsc(signals, parameters, probe=None):
    """Modified group_sparse_covariance, just for joblib wrapping.
    """

    _, est_precs = utils.timeit(group_sparse_covariance)(
        signals, parameters['alpha'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], probe_function=probe,
        precisions_init=parameters.get("precisions_init", None),
        verbose=1, debug=False)

    return est_precs, probe


def benchmark1():
    """Run group_sparse_covariance on a simple case, for benchmarking."""
    parameters = {'n_tasks': 40, 'n_var': 30, 'density': 0.15,
                  'alpha': .01, 'tol': 1e-4, 'max_iter': 50}

    _, _, gt = create_signals(parameters,
                              output_dir="prof_group_sparse_covariance")

    _, est_precs = utils.timeit(group_sparse_covariance)(
        gt["signals"], parameters['alpha'], max_iter=parameters['max_iter'],
        tol=parameters['tol'], verbose=1, debug=False)

    # Check that output doesn't change between invocations.
    utils.cache_array(est_precs, os.path.join("prof_group_sparse_covariance",
                                              "benchmark1_est_precs.npy"),
                      decimal=4)


def benchmark2():
    """Run GroupSparseCovarianceCV on a simple case, for benchmarking."""
    parameters = {'n_tasks': 40, 'n_var': 10, 'density': 0.15,
                  'alphas': 4, 'tol': 1e-4, 'max_iter': 50}

    _, _, gt = create_signals(parameters,
                              output_dir="prof_group_sparse_covariance")

    gsc = GroupSparseCovarianceCV(alphas=parameters['alphas'],
                                  max_iter=parameters['max_iter'],
                                  tol=parameters['tol'],
                                  verbose=1, debug=False,
                                  early_stopping=True)
    utils.timeit(gsc.fit)(gt["signals"])
    print(gsc.alpha_)
    utils.cache_array(gsc.precisions_,
                      os.path.join("prof_group_sparse_covariance",
                      "est_precs_cv_{n_var:d}.npy".format(**parameters)),
                      decimal=3)


def benchmark3():
    """Compare group_sparse_covariance result for different initializations.
    """
    ## parameters = {'n_tasks': 10, 'n_var': 50, 'density': 0.15,
    ##               'alpha': .001, 'tol': 1e-2, 'max_iter': 100}
    parameters = {'n_var': 40, 'n_tasks': 10, 'density': 0.15,
                  'alpha': .01, 'tol': 1e-3, 'max_iter': 100}

    mem = joblib.Memory(".")

    _, _, gt = create_signals(parameters,
                              output_dir="prof_group_sparse_covariance")
    signals = gt["signals"]

    emp_covs, n_samples = empirical_covariances(signals)
    print("alpha max: " + str(compute_alpha_max(emp_covs, n_samples)))

    # With diagonal elements initialization
    probe1 = ScoreProbe()
    est_precs1, probe1 = mem.cache(modified_gsc)(signals, parameters, probe1)
    probe1.comment = "diagonal"  # set after execution for joblib not to see it
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

    print("difference between final estimates (max norm) %.2e"
          % abs(est_precs1 - est_precs2).max())

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


if __name__ == "__main__":
    benchmark3()
