"""Plot various quantities versus iteration step."""
# Does not work: must be adapted to probe system.
import pylab as pl
import numpy as np

from nilearn.group_sparse_covariance import (empirical_covariances,
                                             compute_alpha_max,
                                             _group_sparse_covariance)

import joblib
from common import create_signals, ScoreProbe


def benchmark(parameters, output_d="_convergence"):
    _, _, gt = create_signals(parameters, output_dir=output_d)

    emp_covs, n_samples = empirical_covariances(gt["signals"])
    print("alpha_max: %.3e, %.3e" % compute_alpha_max(emp_covs, n_samples))

    sp = ScoreProbe(duality_gap=True)
    _group_sparse_covariance(
        emp_covs, n_samples, alpha=parameters["alpha"], tol=parameters["tol"],
        max_iter=parameters["max_iter"], probe_function=sp, verbose=1)

    return {"log_lik": np.asarray(sp.log_lik),
            "objective": np.asarray(sp.objective),
            "precisions": np.asarray(sp.precisions),
            "duality_gap": np.asarray(sp.duality_gap),
            "time": np.asarray(sp.wall_clock)}, gt


def benchmark2(parameters, output_d="_convergence"):
    _, _, gt = create_signals(parameters, output_dir=output_d)

    # Normalize to unit variance
    for s in gt["signals"]:
        s /= np.std(s, axis=0)

    emp_covs, n_samples = empirical_covariances(gt["signals"])
    print("alpha_max: %.3e, %.3e" % compute_alpha_max(emp_covs, n_samples))

    sp = ScoreProbe(duality_gap=True)
    _group_sparse_covariance(
        emp_covs, n_samples, alpha=parameters["alpha"], tol=parameters["tol"],
        max_iter=parameters["max_iter"], probe_function=sp, verbose=1)

    return {"log_lik": np.asarray(sp.log_lik),
            "objective": np.asarray(sp.objective),
            "precisions": np.asarray(sp.precisions),
            "duality_gap": np.asarray(sp.duality_gap),
            "time": np.asarray(sp.wall_clock)}, gt


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
#    parameters = dict(n_var=300, n_tasks=10, density=0.15, max_iter=500, alpha=0.1, tol=1e-3)
    parameters = dict(n_var=50, n_tasks=10, density=0.15, max_iter=50, alpha=0.005, tol=1e-3)

    output_dir = "_convergence_graph"
    mem = joblib.Memory(output_dir)

    ## costs, gt = mem.cache(benchmark, ignore=("output_d",))(
    ##     parameters, output_d=output_dir)
    costs, gt = mem.cache(benchmark2, ignore=("output_d",))(
        parameters, output_d=output_dir)

    # proportion of zeros in estimated precision matrices
    n_zeros = [1. * (prec[..., 0] == 0).sum() / (prec.shape[0] ** 2)
               for prec in costs["precisions"]]

    pl.figure()
    pl.plot(costs["time"], costs["objective"], "-+")
    pl.xlabel("time [s]")
    pl.ylabel("objective")
    pl.grid()

    pl.figure()
    pl.loglog(costs["time"],
              (costs["objective"] - np.mean(costs["objective"][-2:])), "-+")
    pl.xlabel("time [s]")
    pl.ylabel("difference with final value")
    pl.title("objective")
    pl.grid()

    pl.figure()
    pl.loglog(costs["time"], costs["duality_gap"], "-+")
    pl.xlabel("time [s]")
    pl.ylabel("duality gap")
    pl.grid()

    pl.figure()
    pl.semilogx(costs["time"], n_zeros, "-+")
    pl.xlabel("time [s]")
    pl.ylabel("Number of zeros (normalized to 1)")
    pl.grid()

    pl.show()
