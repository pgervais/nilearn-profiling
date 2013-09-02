"""Plot various quantities versus iteration step."""
# Does not work: must be adapted to probe system.
import pylab as pl
import numpy as np

from nilearn.group_sparse_covariance import (empirical_covariances,
                                             compute_alpha_max,
                                             _group_sparse_covariance)

import joblib
from common import create_signals


def benchmark(parameters, output_d="convergence"):
    _, _, gt = create_signals(parameters, output_dir=output_d)

    emp_covs, n_samples = empirical_covariances(gt["signals"])
    print("alpha_max: %.3e, %.3e" % compute_alpha_max(emp_covs, n_samples))
    _, costs = _group_sparse_covariance(
        emp_covs, n_samples, alpha=parameters["alpha"], tol=parameters["tol"],
        max_iter=parameters["max_iter"], return_costs=True, verbose=1)

    costs = zip(*costs)
    return costs, gt


def benchmark2(parameters, output_d="convergence"):
    _, _, gt = create_signals(parameters, output_dir=output_d)

    # Normalize to unit variance
    for s in gt["signals"]:
        s /= np.std(s, axis=0)

    emp_covs, n_samples = empirical_covariances(gt["signals"])
    print("alpha_max: %.3e, %.3e" % compute_alpha_max(emp_covs, n_samples))
    _, costs = _group_sparse_covariance(
        emp_covs, n_samples, alpha=parameters["alpha"], tol=parameters["tol"],
        max_iter=parameters["max_iter"], return_costs=True, verbose=1)

    costs = zip(*costs)
    return costs, gt


if __name__ == "__main__":

# For benchmark()
#    parameters = dict(n_var=40, n_tasks=10, density=0.15, max_iter=50, alpha=0.1, tol=-1)
#    parameters = dict(n_var=40, n_tasks=10, density=0.15, max_iter=80, alpha=0.1, tol=-1)
#    parameters = dict(n_var=40, n_tasks=10, density=0.15, max_iter=150, alpha=0.1, tol=-1)
#    parameters = dict(n_var=40, n_tasks=10, density=0.15, max_iter=300, alpha=0.1, tol=-1)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=0.1, tol=1.)

# Loop on alpha
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=0.1, tol=-1.)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=16, tol=-1.)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=8, tol=5e-10)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=40, alpha=8, tol=-1)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=4, tol=1e-11)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=2., tol=1e-11)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=1., tol=1e-11)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=.5, tol=1e-11)
#    parameters = dict(n_var=80, n_tasks=10, density=0.15, max_iter=300, alpha=.25, tol=1e-11)

#    parameters = dict(n_var=300, n_tasks=10, density=0.15, max_iter=5, alpha=0.1, tol=-1.)

# For benchmark2()
    parameters = dict(n_var=300, n_tasks=10, density=0.15, max_iter=500, alpha=0.1, tol=1e-5)

    output_dir = "convergence"
    mem = joblib.Memory(output_dir)

    ## costs, gt = mem.cache(benchmark, ignore=("output_d",))(
    ##     parameters, output_d=output_dir)
    costs, gt = mem.cache(benchmark2, ignore=("output_d",))(
        parameters, output_d=output_dir)

    # distance to ground truth
    true_precisions = np.dstack(gt["precisions"])
    l2_dist = [((prec - true_precisions) ** 2).mean() for prec in costs[2]]

    # proportion of zeros in estimated precision matrices
    n_zeros = [1. * (prec[..., 0] == 0).sum() / (prec.shape[0] ** 2)
               for prec in costs[2]]

    pl.figure()
    pl.loglog(costs[1], "-+")
    pl.xlabel("iteration")
    pl.ylabel("duality gap")
    pl.grid()

    pl.figure()
    pl.loglog((np.asarray(costs[0]) - np.mean(costs[0][-2:])), "-+")
    pl.xlabel("iteration")
    pl.ylabel("difference with final value")
    pl.title("cost function")
    pl.grid()

    pl.figure()
#    pl.loglog((np.asarray(l2_dist) - np.mean(l2_dist[-2:])), "-+")
    pl.semilogx(np.asarray(l2_dist), "-+")
    pl.xlabel("iteration")
    pl.ylabel("difference with final value")
    pl.title("L2 distance to true precision")
    pl.grid()

    pl.figure()
    pl.semilogx(n_zeros, "-+")
    pl.xlabel("iteration")
    pl.ylabel("Number of zeros (normalized)")
    pl.grid()

    pl.show()

