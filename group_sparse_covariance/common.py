import math
import os.path
import glob
import time
import cPickle as pickle

import numpy as np

import nilearn
from nilearn.group_sparse_covariance import _group_sparse_covariance


def cache_array(arr, filename, decimal=7):
    assert filename.endswith(".npy")
    if os.path.isfile(filename):
        cached = np.load(filename)
        np.testing.assert_almost_equal(cached, arr, decimal=decimal)
    else:
        np.save(filename, arr)


def random_spd(n, rand_gen=np.random.RandomState(1)):
    M = 0.1 * rand_gen.randn(n, n)
    return np.dot(M, M.T)


def get_cache_dir(parameters, output_dir):
    basename = ("case_{n_var:d}_{n_tasks:d}_"
                "{density:.2f}".format(**parameters))
    if 'rho' in parameters:
        basename += "_{rho:.3f}".format(**parameters)

    basename += (("_{tol:.4f}_{min_samples:d}_{max_samples:d}"
                  "").format(**parameters))

    return os.path.join(output_dir, basename)


def get_ground_truth(cache_dir):
    """Return a dictionary containing the ground truth values. """
    ground_truth_fname = os.path.join(cache_dir, "ground_truth.pickle")
    return pickle.load(open(ground_truth_fname, "rb"))


def iter_outputs(cache_dir):
    filenames = glob.glob(os.path.join(cache_dir, "precisions_*.pickle"))
    for fname in filenames:
        yield pickle.load(open(fname, 'rb'))


def create_signals(parameters, output_dir="sensitivity"):
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


def save_group_sparse_covariance(emp_covs, n_samples, rho, max_iter, tol,
                                 cache_dir, num=0, random_init=True):
    if random_init:
        rand_gen = np.random.RandomState(
            int(int(1000000 * time.time()) % 100000000))
        precisions_init = np.empty(emp_covs.shape)
        for k in xrange(emp_covs.shape[-1]):
            precisions_init[..., k] = random_spd(emp_covs.shape[0],
                                                 rand_gen=rand_gen)
    else:
        precisions_init = None

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
