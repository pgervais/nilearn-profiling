"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices.

Estimation of matrix coefficients uncertainty by random sampling of starting
points.
"""

# Authors: Philippe Gervais
# License: simplified BSD

import os.path
import os
import glob
import time

import cPickle as pickle
import itertools

import numpy as np

import joblib

import nilearn
from nilearn.group_sparse_covariance import (_group_sparse_covariance, rho_max,
                                             empirical_covariances)
from common import random_spd


def group_sparse_covariance(emp_covs, n_samples, rho, max_iter, tol,
                            cache_dir, num=0):
    rand_gen = np.random.RandomState(int(1000000 * time.time()))
    precisions_init = np.empty(emp_covs.shape)
    for k in xrange(emp_covs.shape[-1]):
        precisions_init[..., k] = random_spd(emp_covs.shape[0],
                                             rand_gen=rand_gen)

    precisions, costs = _group_sparse_covariance(
        emp_covs, n_samples, rho, max_iter=max_iter, tol=tol,
        return_costs=True, verbose=1, debug=False,
        precisions_init=precisions_init)

    output_fname = os.path.join(cache_dir,
                                "precisions_{num:d}.pickle".format(num=num))
    pickle.dump(dict(n_samples=n_samples, rho=rho, max_iter=max_iter, tol=tol,
                     objective=costs[-1][0], duality_gap=costs[-1][1],
                     precisions=precisions, precisions_init=precisions_init),
                open(output_fname, "wb"))


def get_series(params, keys):
    ret = [[p[k] for p in params] for k in keys]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def get_cache_dir(parameters, output_dir):
    return os.path.join(output_dir,
                        "case_{n_var:d}_{n_tasks:d}_"
                        "{density:.2f}_{rho:.3f}_{tol:.4f}".format(
                            n_var=parameters["n_var"],
                            n_tasks=parameters["n_tasks"],
                            density=parameters["density"],
                            tol=parameters["tol"],
                            rho=parameters["rho"])
                        )


def setup(parameters, output_dir="sensitivity"):
    """Simple cache system.

    parameters: dict
        keys: n_var, n_tasks, density
    """
    cache_dir = get_cache_dir(parameters, output_dir)

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
        next_num = 0

    else:
        filenames = glob.glob(os.path.join(cache_dir, "precisions_*.pickle"))
        numbers = [int(os.path.basename(fname).rsplit(".")[0].split("_")[1])
                   for fname in filenames]

        if len(numbers) > 0:
            next_num = max(numbers) + 1
        else:
            next_num = 0

    # Look for/create true precisions, topology and signals
    ground_truth_fname = os.path.join(cache_dir, "ground_truth.pickle")
    if not os.path.isfile(ground_truth_fname):
        rand_gen = np.random.RandomState(0)
        min_samples = 100
        max_samples = 150
        # Generate signals
        precisions, topology = \
                    nilearn.testing.generate_sparse_precision_matrices(
                        n_tasks=parameters["n_tasks"],
                        n_var=parameters["n_var"],
                        density=parameters["density"], rand_gen=rand_gen)
        signals = nilearn.testing.generate_signals_from_precisions(
            precisions, min_samples=min_samples, max_samples=max_samples,
            rand_gen=rand_gen)
        pickle.dump({"precisions": precisions, "topology": topology,
                     "signals": signals}, open(ground_truth_fname, "wb"))

    gt = pickle.load(open(ground_truth_fname, "rb"))

    return next_num, cache_dir, gt


def sample_precision_space(parameters, number=100):
    """Launch a large number of times the same estimation, with different
    starting points.

    number: int
        number of samples to generate.
    """
    # Signals
    next_num, cache_dir, gt = setup(parameters)
    min_samples, max_samples = 100, 150  # train signals length

    # Estimation
    max_iter = 200
    rho = parameters["rho"]
    tol = parameters["tol"]

    # Generate signals
    precisions, topology, signals = (gt["precisions"], gt["topology"],
                                     gt["signals"])

    emp_covs, n_samples, _, _ = empirical_covariances(signals)

    print("rho max: %.3e" % rho_max(emp_covs, n_samples))

    # Estimate a lot of precision matrices
    parameters = joblib.Parallel(n_jobs=7, verbose=1)(
        joblib.delayed(group_sparse_covariance)(emp_covs, n_samples, rho,
                                                max_iter=max_iter, tol=tol,
                                                cache_dir=cache_dir, num=n)
        for n in xrange(next_num, next_num + number))

    ## group_sparse_covariance(emp_covs, n_samples, rho,
    ##                         max_iter=max_iter, tol=tol,
    ##                         cache_dir=cache_dir, num=next_num)


if __name__ == "__main__":
    ## sample_precision_space({"n_var": 50, "n_tasks": 40, "density": 0.1,
    ##                         "tol": 1e-2, "rho": 0.02})
    sample_precision_space({"n_var": 100, "n_tasks": 40, "density": 0.1,
                            "tol": 1., "rho": 0.02}, number=1)
