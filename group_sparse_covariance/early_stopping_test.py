"""Some functions for helping developing the early stopping criterion
of group_sparse_covariance() """

import utils
import time
import cPickle as pickle

import numpy as np
import pylab as pl

import joblib

import nilearn.image
import nilearn.input_data as input_data

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             empirical_covariances, rho_max,
                                             group_sparse_score,
                                             GroupSparseCovarianceCV)
import nilearn._utils.testing as testing


class CostProbe(object):
    """Probe function for group_sparse_covariance computing various scores and
    quantities during optimization."""

    def __init__(self, test_emp_covs):
        self.test_emp_covs = test_emp_covs
        self.score = []  # score on train set
        self.test_score = []  # score on test set
        self.initial_test_score = None
        self.wall_clock = []
        self.start_time = None

    def __call__(self, emp_covs, n_samples, rho, max_iter, tol,
                 iter_n, omega, prev_omega):
        if iter_n == -1:
            self.start_time = time.time()
            self.wall_clock.append(0)
        else:
            self.wall_clock.append(time.time() - self.start_time)
        self.score.append(group_sparse_score(omega, n_samples, emp_covs,
                                             rho)[0])
        self.test_score.append(group_sparse_score(omega, n_samples,
                                                  self.test_emp_covs,
                                                  rho)[0])
        if self.test_score[-1] - self.test_score[0] > 0:
            print("test_score increasing")


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


def region_signals(subject_n):
    """Extract signals from brain regions. Uses the adhd database."""
    dataset = nilearn.datasets.fetch_adhd()
    filename = dataset["func"][subject_n]
    confound_file = dataset["confounds"][subject_n]

    print("Processing file %s" % filename)

    print("-- Loading raw data ({0:d}) and masking ...".format(subject_n))
    msdl_atlas = nilearn.datasets.fetch_msdl_atlas()
    niimgs = filename

    print("-- Computing confounds ...")
    hv_confounds = nilearn.image.high_variance_confounds(niimgs)

    print("-- Computing region signals ...")
    masker = input_data.NiftiMapsMasker(msdl_atlas["maps"],
                                     resampling_target="maps",
                                     low_pass=None, high_pass=0.01, t_r=2.5,
                                     standardize=True,
                                     verbose=1)
    region_ts = masker.fit_transform(niimgs,
                                     confounds=[hv_confounds, confound_file])

    return region_ts


def plot_probe_results(rhos, cost_probes):
    pl.figure()
    for rho, probe in zip(rhos, cost_probes):
        pl.plot(probe.wall_clock, probe.test_score, '+-', label="%.2e" % rho)
    pl.xlabel("time [s]")
    pl.ylabel('score on test set')
    pl.grid()
    pl.legend()

    pl.figure()
    for rho, probe in zip(rhos, cost_probes):
        pl.plot(probe.wall_clock, probe.score, '+-', label="%.2e" % rho)
    pl.xlabel("time [s]")
    pl.ylabel('score on train set')
    pl.grid()
    pl.legend()


def split_signals(signals):
    """Split signals into train and test sets."""
    # Smallest signal is 77-sample long.
    # Keep first 50 samples for train set, everything else for test set.
    #    train_test = [(s[:50, ...], s[50:, ...]) for s in signals]
    # Keep first two-third for train set, everything else for test set
    train_test = [(s[:2 * s.shape[0] // 3, ...], s[2 * s.shape[0] // 3:, ...])
                  for s in signals]
    signals, test_signals = zip(*train_test)

    emp_covs, n_samples, _, _ = empirical_covariances(signals)
    test_emp_covs, test_n_samples, _, _ = empirical_covariances(test_signals)

    n_samples_norm = n_samples.copy()
    n_samples_norm /= n_samples_norm.sum()

    return signals, test_signals, emp_covs, test_emp_covs, n_samples_norm


def brute_force_study():
    """Loop through many values of rho, and run a full gsc for each.

    Record information for each iteration using CostProbe, store the
    obtained values on disk.

    Plot scores on train and test results versus time.
    """
    parameters = {'n_tasks': 10, 'tol': 1e-3, 'max_iter': 50}
    mem = joblib.Memory(".")

    print("-- Extracting signals ...")
    signals = []
    for n in range(parameters["n_tasks"]):
        signals.append(mem.cache(region_signals)(n))

    signals, test_signals, emp_covs, test_emp_covs, n_samples_norm = \
             split_signals(signals)

    print("-- Optimizing --")
    rho_mx, _ = rho_max(emp_covs, n_samples_norm)
#    rhos = np.logspace(-3, -1, 10)
    rhos = np.logspace(np.log10(rho_mx / 500), np.log10(rho_mx), 100)
    cost_probes = []
    t0 = time.time()
    for rho in rhos:
        # Honorio-Samaras
        cost_probes.append(CostProbe(test_emp_covs))
        _, est_precs = utils.timeit(group_sparse_covariance)(
            signals, rho, max_iter=parameters['max_iter'],
            tol=parameters['tol'], verbose=1, debug=False,
            probe_function=cost_probes[-1])
    t1 = time.time()
    print ('Time spent in loop: %.2fs' % (t1 - t0))

    pickle.dump([rhos, cost_probes], open('cost_probes.pickle', "w"))
    plot_probe_results(rhos, cost_probes)


def cv_object_study():
    """Convenience function for running GroupSparseCovarianceCV. """
    parameters = {'n_tasks': 10, 'tol': 1e-3, 'max_iter': 50}
    synthetic = False

    print("-- Getting signals")
    if synthetic:
        parameters["n_var"] = 50
        parameters["density"] = 0.2
        signals, _, _ = generate_signals(parameters)
    else:
        mem = joblib.Memory(".")
        signals = []
        for n in range(parameters["n_tasks"]):
            signals.append(mem.cache(region_signals)(n))

    print("-- Optimizing")
    gsc = GroupSparseCovarianceCV(early_stopping=False)
    t0 = time.time()
    gsc.fit(signals)
    t1 = time.time()
    print("\nTime spent in fit(): %.1f s" % (t1 - t0))
    print("\n-- selected rho: %.3e" % gsc.rho_)
    print("-- cv_rhos: ")
    print(gsc.cv_rhos)
    print("-- cv_scores: ")
    print(gsc.cv_scores)

    pickle.dump([gsc.rho_, gsc.cv_rhos, gsc.cv_scores, gsc.covariances_,
                 gsc.precisions_],
                open("early_stopping_test_gsc.pickle", "wb"))

if __name__ == "__main__":
    #    brute_force_study(); pl.show()
    cv_object_study()

