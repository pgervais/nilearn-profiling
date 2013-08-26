"""Some functions for helping developing the early stopping criterion
of group_sparse_covariance() """

import utils
import time

import numpy as np
import pylab as pl

import joblib

import nilearn.image
import nilearn.input_data as input_data
import nibabel

from nilearn.group_sparse_covariance import (group_sparse_covariance,
                                             empirical_covariances, rho_max,
                                             group_sparse_score)
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

    def plot_test_score(self, *args, **kwargs):
        pl.plot(self.wall_clock, self.test_score, *args,
                label=kwargs.get("label", None))

    def plot_score(self, *args, **kwargs):
        pl.plot(self.wall_clock, self.score, *args,
                label=kwargs.get("label", None))


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
    niimgs = nibabel.load(filename)

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


def loop():
    parameters = {'n_tasks': 10, 'tol': 1e-3, 'max_iter': 5}
    mem = joblib.Memory(".")

    print("-- Extracting signals ...")
    signals = []
    for n in range(parameters["n_tasks"]):
        signals.append(mem.cache(region_signals)(n))

    # Smallest signal is 77-sample long.
    # Keep first 50 samples for train set, everything else for test set.
    # Having the same number of samples is important to get similar results
    # from the two algorithms.
    train_test = [(s[:50, ...], s[50:, ...]) for s in signals]
    signals, test_signals = zip(*train_test)

    emp_covs, n_samples, _, _ = empirical_covariances(signals)
    test_emp_covs, test_n_samples, _, _ = empirical_covariances(test_signals)

    n_samples_norm = n_samples.copy()
    n_samples_norm /= n_samples_norm.sum()

    print("-- Optimizing --")

    cost_probes = []
    rhos = np.logspace(-3, -1, 10)
    for rho in rhos:
        # Honorio-Samaras
        cost_probes.append(CostProbe(test_emp_covs))
        _, est_precs = utils.timeit(
#            mem.cache(group_sparse_covariance, ignore=["debug", "verbose"])
            group_sparse_covariance
            )(signals, rho, max_iter=parameters['max_iter'],
              tol=parameters['tol'] * 100., verbose=1, debug=False,
              probe_function=cost_probes[-1])

        ## Compute values of interest.

        # Norms of difference with last value
        ## frobenius.append([np.sqrt(((prec2 - prec1) ** 2).sum())
        ##                   / est_precs.shape[0] ** 2
        ##                   for (_, _, _, prec2, _), (_, _, _, prec1, _)
        ##                   in zip(costs[1:], costs[:-1])])

        ## l1.append([abs(prec2 - prec1).sum()
        ##            / est_precs.shape[0] ** 2
        ##            for (_, _, _, prec2, _), (_, _, _, prec1, _)
        ##            in zip(costs[1:], costs[:-1])])

        ## l_max.append([abs(prec2 - prec1).max()
        ##               for (_, _, _, prec2, _), (_, _, _, prec1, _)
        ##               in zip(costs[1:], costs[:-1])])

    ## min_running_score = [min(r) for r in running_score]
    ## min_running_score_v = [min(r) for r in running_score_v]

    ## argmin_running_score = [np.argmin(r) for r in running_score]
    ## argmin_running_score_v = [np.argmin(r) for r in running_score_v]

    ## time_min_running_score = [w[np.argmin(r)] - w[0]
    ##                           for (r, w) in zip(running_score, wall_clock)]
    ## time_min_running_score_v = [w[np.argmin(r)] - w[0]
    ##                           for (r, w) in zip(running_score_v, wall_clock_v)]

    ## pl.figure()
    ## pl.semilogx(rhos, min_running_score, '+-', label='HS')
    ## pl.semilogx(rhos, min_running_score_v, '+-', label='V')
    ## pl.ylabel('minimum score')
    ## pl.xlabel('rho')
    ## pl.grid()
    ## pl.legend()

    ## pl.figure()
    ## pl.semilogx(rhos, argmin_running_score, '+-', label='HS')
    ## pl.semilogx(rhos, argmin_running_score_v, '+-', label='V')
    ## pl.ylabel('iteration @ minimum score')
    ## pl.xlabel('rho')
    ## pl.grid()
    ## pl.legend()

    ## pl.figure()
    ## pl.semilogx(rhos, time_min_running_score, '+-', label='HS')
    ## pl.semilogx(rhos, time_min_running_score_v, '+-', label='V')
    ## pl.ylabel('time @ minimum score [s]')
    ## pl.xlabel('rho')
    ## pl.grid()
    ## pl.legend()

    pl.figure()
    for rho, probe in zip(rhos[::2], cost_probes[::2]):
        probe.plot_test_score('+-', label="%.2e" % rho)
    pl.xlabel("time [s]")
    pl.ylabel('score on test set')
    pl.grid()
    pl.legend()

    pl.figure()
    for rho, probe in zip(rhos[::2], cost_probes[::2]):
        probe.plot_score('+-', label="%.2e" % rho)
    pl.xlabel("time [s]")
    pl.ylabel('score on train set')
    pl.grid()
    pl.legend()


def benchmark():
    ## parameters = {'n_tasks': 10, 'n_var': 30, 'density': 0.15,
    ##               'rho': .01, 'tol': 1e-5, 'max_iter': 100,
    ##               'min_samples': 100, 'max_samples': 101}
    parameters = {'n_tasks': 10, 'rho': .001, 'tol': 1e-5, 'max_iter': 100}

    mem = joblib.Memory(".")

    print("-- Extracting signals ...")
    signals = []
    for n in range(parameters["n_tasks"]):
        signals.append(mem.cache(region_signals)(n))

    mem = joblib.Memory(None)

    # Smallest signal is 77-sample long.
    # Keep first 50 samples for train set, everything else for test set.
    # Having the same number of samples is important to get similar results
    # from the two algorithms.
    train_test = [(s[:50, ...], s[50:, ...]) for s in signals]
    signals, test_signals = zip(*train_test)

    ## signals, _, _ = generate_signals(parameters)
    # Covariances
    emp_covs, n_samples, _, _ = empirical_covariances(signals)
    print("rho_max: " + str(rho_max(emp_covs, n_samples)))

    # Honorio-Samaras
    _, est_precs, costs = utils.timeit(
        mem.cache(group_sparse_covariance, ignore=["debug", "verbose"])
        )(signals, parameters['rho'], max_iter=parameters['max_iter'],
          tol=parameters['tol'], verbose=1, debug=False, return_costs=True)

    # Results
    test_emp_covs, test_n_samples, _, _ = empirical_covariances(test_signals)

    n_samples_norm = n_samples.copy()
    n_samples_norm /= n_samples_norm.sum()
    score, objective = group_sparse_score(est_precs, n_samples_norm,
                                          test_emp_covs,
                                          parameters["rho"])

    # Scores for different iterations
    running_score = [- group_sparse_score(prec, n_samples_norm, test_emp_covs,
                                          parameters["rho"])[0]
                     for _, _, prec, _ in costs]
    wall_clock = np.asarray(zip(*costs)[3])   # Extract fourth column

    print("\n-- Final values --")
    print("Cost. %.4e" % (-objective, ))
    print("Score: %.4e" % (-score, ))

    # Figures
    pl.figure()
    pl.plot(wall_clock - wall_clock[0], running_score, "+-", label="HS")
    pl.ylabel("score on test set")
    pl.xlabel("time [s]")
    pl.title('rho: %.3e' % parameters['rho'])
    pl.legend()
    pl.grid()

    pl.figure()
    pl.imshow(est_precs[..., 0], interpolation="nearest")
    pl.title("Honorio-Samaras")
    pl.colorbar()


if __name__ == "__main__":
    #    benchmark()
    loop()
    pl.show()
