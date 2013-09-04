import os.path
import glob
import time
import cPickle as pickle

import numpy as np

import nilearn._utils.testing as testing
from nilearn.group_sparse_covariance import (_group_sparse_covariance,
                                             group_sparse_scores)


class ScoreProbe(object):
    """Probe function for group_sparse_covariance computing various scores and
    quantities during optimization."""

    def __init__(self):
        self.log_lik = []  # log_likelihood on train set
        self.objective = []  # objective minimized by the algorithm
        self.wall_clock = []  # End time of each iteration
        self.start_time = None

    def __call__(self, emp_covs, n_samples, alpha, max_iter, tol,
                 iter_n, omega, prev_omega):
        if iter_n == -1:
            self.start_time = time.time()
            self.wall_clock.append(0)
        else:
            self.wall_clock.append(time.time() - self.start_time)
        log_lik, objective = group_sparse_scores(omega, n_samples, emp_covs,
                                                 alpha)
        self.log_lik.append(log_lik)
        self.objective.append(objective)


def makedirs(directory):
    """Create directory if it does not already exist."""
    if not os.path.isdir(directory):
        os.makedirs(directory)


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
    if 'alpha' in parameters:
        basename += "_{alpha:.3f}".format(**parameters)

    if "tol" in parameters:
        basename += ("_{tol:.4f}").format(**parameters)
    if "min_samples" in parameters:
        basename += "_{min_samples:d}_{max_samples:d}".format(**parameters)

    return os.path.join(output_dir, basename)


def get_ground_truth(cache_dir):
    """Return a dictionary containing the ground truth values. """
    ground_truth_fname = os.path.join(cache_dir, "ground_truth.pickle")
    return pickle.load(open(ground_truth_fname, "rb"))


def iter_outputs(cache_dir):
    filenames = glob.glob(os.path.join(cache_dir, "precisions_*.pickle"))
    for fname in filenames:
        yield pickle.load(open(fname, 'rb'))


def create_signals(parameters, output_dir="tmp_signals"):
    """Simple cache system.

    parameters: dict
        keys: n_var, n_tasks, density, (mandatory)
        min_samples, max_samples (optional)
        normalize (optional, default True)
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
        min_samples = parameters.get("min_samples", 100)
        max_samples = parameters.get("max_samples", 150)
        # Generate signals
        signals, precisions, topology = \
                 testing.generate_group_sparse_gaussian_graphs(
            n_subjects=parameters["n_tasks"], n_features=parameters["n_var"],
            density=parameters["density"], random_state=rand_gen,
            min_n_samples=min_samples, max_n_samples=max_samples)

        if parameters.get("normalize", True):
            for signal in signals:
                signal /= signal.std(axis=0)
        pickle.dump({"precisions": precisions, "topology": topology,
                     "signals": signals}, open(ground_truth_fname, "wb"))

    gt = pickle.load(open(ground_truth_fname, "rb"))

    return next_num, cache_dir, gt


def save_group_sparse_covariance(emp_covs, n_samples, alpha, max_iter, tol,
                                 cache_dir, num=0, random_init=True,
                                 debug=False):
    if random_init:
        rand_gen = np.random.RandomState(
            int(int(1000000 * time.time()) % 100000000))
        precisions_init = np.empty(emp_covs.shape)
        for k in xrange(emp_covs.shape[-1]):
            precisions_init[..., k] = random_spd(emp_covs.shape[0],
                                                 rand_gen=rand_gen)
    else:
        precisions_init = None

    probe = ScoreProbe()
    precisions = _group_sparse_covariance(
        emp_covs, n_samples, alpha, max_iter=max_iter, tol=tol,
        verbose=1, debug=debug, probe_function=probe,
        precisions_init=precisions_init)

    output_fname = os.path.join(cache_dir,
                                "precisions_{num:d}.pickle".format(num=num))
    pickle.dump(dict(n_samples=n_samples, alpha=alpha, max_iter=max_iter,
                     tol=tol, objective=probe.objective,
                     log_lik=probe.log_lik, wall_clock=probe.wall_clock,
                     precisions=precisions, precisions_init=precisions_init),
                open(output_fname, "wb"))
