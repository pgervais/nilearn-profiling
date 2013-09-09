"""Some functions for helping developing the early stopping criterion
of group_sparse_covariance() """

import utils
import time
import os.path
import cPickle as pickle

import numpy as np
import pylab as pl

import joblib
from sklearn.cross_validation import KFold

import nilearn.image
import nilearn.input_data as input_data

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             empirical_covariances,
                                             compute_alpha_max,
                                             group_sparse_scores,
                                             GroupSparseCovarianceCV)
import nilearn._utils.testing as testing
import common


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
        self.first_max_ind = None

    def __call__(self, emp_covs, n_samples, alpha, max_iter, tol,
                 iter_n, omega, prev_omega):
        if iter_n == -1:
            self.start_time = time.time()
            self.wall_clock.append(0)
        else:
            self.wall_clock.append(time.time() - self.start_time)
        self.score.append(group_sparse_scores(omega, n_samples, emp_covs,
                                              alpha)[0])
        self.test_score.append(group_sparse_scores(omega, n_samples,
                                                   self.test_emp_covs,
                                                   alpha)[0])
        if iter_n > -1 and self.test_score[-2] > self.test_score[-1]:
            if self.first_max_ind is None:
                print("score decreasing")
                self.first_max_ind = iter_n


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


# Useless?
def plot_probe_results(alphas, cost_probes,
                       min_alpha=-float("inf"), max_alpha=float("inf")):
    ind = np.where(np.logical_and(alphas >= min_alpha, alphas <= max_alpha))[0]
    cost_probes = [cost_probes[i] for i in ind]
    alphas = alphas[ind]
    pl.figure()
    for alpha, probe in zip(alphas, cost_probes):
        pl.plot(probe.wall_clock, probe.test_score, '+-', label="%.2e" % alpha)
    pl.xlabel("time [s]")
    pl.ylabel('score on test set')
    pl.grid()
    pl.legend()

    pl.figure()
    for alpha, probe in zip(alphas, cost_probes):
        pl.plot(probe.wall_clock, probe.score, '+-', label="%.2e" % alpha)
    pl.xlabel("time [s]")
    pl.ylabel('score on train set')
    pl.grid()
    pl.legend()


def split_signals(signals, fold_n=0):
    """Split signals into train and test sets."""
    # Smallest signal is 77-sample long.
    # Keep first 50 samples for train set, everything else for test set.
    #    train_test = [(s[:50, ...], s[50:, ...]) for s in signals]
    # Keep first two-third for train set, everything else for test set

    folds = [tuple(KFold(s.shape[0], 3)) for s in signals]
    train_test = [(s[fold[fold_n][0], ...], s[fold[fold_n][1], ...])
                  for s, fold in zip(signals, folds)]
    signals, test_signals = zip(*train_test)

    emp_covs, n_samples = empirical_covariances(signals)
    test_emp_covs, test_n_samples = empirical_covariances(test_signals)

    n_samples_norm = n_samples.copy()
    n_samples_norm /= n_samples_norm.sum()

    return signals, test_signals, emp_covs, test_emp_covs, n_samples_norm


def brute_force_study(output_dir="_early_stopping"):
    """Loop through many values of alpha, and run a full gsc for each.

    Record information for each iteration using CostProbe, store the
    obtained values on disk.

    Plot scores on train and test sets versus wall-clock time.
    """
    parameters = {'n_tasks': 10, 'tol': 1e-3, 'max_iter': 50, "fold_n": 2,
                  "n_alphas": 20}
    mem = joblib.Memory(".")

    print("-- Extracting signals ...")
    signals = []
    for n in range(parameters["n_tasks"]):
        signals.append(mem.cache(region_signals)(n))

    signals, test_signals, emp_covs, test_emp_covs, n_samples_norm = \
             split_signals(signals, fold_n=parameters["fold_n"])

    print("-- Optimizing --")
    alpha_mx, _ = compute_alpha_max(emp_covs, n_samples_norm)
#    alphas = np.logspace(-3, -1, 10)
    alphas = np.logspace(np.log10(alpha_mx / 500), np.log10(alpha_mx),
                       parameters["n_alphas"])
    cost_probes = []
    t0 = time.time()
    for alpha in alphas:
        # Honorio-Samaras
        cost_probes.append(CostProbe(test_emp_covs))
        _, est_precs = utils.timeit(group_sparse_covariance)(
            signals, alpha, max_iter=parameters['max_iter'],
            tol=parameters['tol'], verbose=1, debug=False,
            probe_function=cost_probes[-1])
    t1 = time.time()
    print ('Time spent in loop: %.2fs' % (t1 - t0))

    out_filename = os.path.join(output_dir, 'brute_force_study.pickle')
    pickle.dump([alphas, cost_probes], open(out_filename, "wb"))
    print("Use plot_early_stopping.py to analyze the generated file:\n"
          "%s" % out_filename)


def cv_object_study(early_stopping=True, output_dir="_early_stopping"):
    """Convenience function for running GroupSparseCovarianceCV. """
    parameters = {'n_tasks': 10, 'tol': 1e-3, 'max_iter': 50, "n_jobs": 7,
                  "cv": 4}
    parameters["tol_cv"] = parameters["tol"]
    parameters["max_iter_cv"] = parameters["max_iter"]

    synthetic = False

    print("-- Getting signals")
    if synthetic:
        parameters["n_features"] = 50
        parameters["density"] = 0.2
        signals, _, _ = testing.generate_group_sparse_gaussian_graphs(
            n_subjects=parameters["n_tasks"],
            n_features=parameters["n_features"],
            min_n_samples=100, max_n_samples=150,
            density=parameters["density"])
    else:
        mem = joblib.Memory(".")
        signals = []
        for n in range(parameters["n_tasks"]):
            signals.append(mem.cache(region_signals)(n))

    print("-- Optimizing")
    gsc = GroupSparseCovarianceCV(early_stopping=early_stopping,
                                  cv=parameters["cv"],
                                  n_jobs=parameters["n_jobs"],
                                  tol=parameters["tol"],
                                  tol_cv=parameters["tol_cv"],
                                  max_iter=parameters["max_iter"],
                                  max_iter_cv=parameters["max_iter_cv"],
                                  verbose=1)
    t0 = time.time()
    gsc.fit(signals)
    t1 = time.time()
    print("\nTime spent in fit(): %.1f s" % (t1 - t0))
    print("\n-- selected alpha: %.3e" % gsc.alpha_)
    print("-- cv_alphas_:")
    print(repr(np.asarray(gsc.cv_alphas_)))
    print("-- cv_scores_:")
    print(repr(np.asarray(gsc.cv_scores_)))

    out_filename = os.path.join(output_dir, "cv_object_study.pickle")
    pickle.dump([gsc.alpha_, gsc.cv_alphas_, gsc.cv_scores_, gsc.covariances_,
                 gsc.precisions_], open(out_filename, "wb"))

if __name__ == "__main__":
    output_dir = "_early_stopping"
    common.makedirs(output_dir)
#    brute_force_study(output_dir=output_dir)
    cv_object_study(early_stopping=True, output_dir=output_dir)
    cv_object_study(early_stopping=False, output_dir=output_dir)
