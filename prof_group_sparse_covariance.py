"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD

from nilearn.group_sparse_covariance import group_sparse_covariance
from nilearn.tests.test_group_sparse_covariance \
        import generate_multi_task_gg_model
import utils  # defines profile() if not already defined


def benchmark():
    rho = .2

    signals, _, _ = generate_multi_task_gg_model(n_tasks=40, n_var=30,
                                              density=0.15,
                                              min_samples=100, max_samples=150)

    emp_covs, est_precs = utils.timeit(group_sparse_covariance)(
        signals, rho, max_iter=20, tol=1e-5, verbose=0, debug=False)

    ## import pylab as pl
    ## pl.matshow(est_precs[..., 0])
    ## pl.show()

if __name__ == "__main__":
    benchmark()
