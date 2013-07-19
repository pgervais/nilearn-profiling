"""Test that the signal generator with a sparse precision matrix works."""
import pylab as pl
import numpy as np
import nilearn
import nilearn.testing
import nilearn.group_sparse_covariance


def generate_signals(n_var=20, n_tasks=2, density=0.15,
                     min_samples=100, max_samples=150,
                     rand_gen=np.random.RandomState(0)):
    precisions, topology = nilearn.testing.generate_sparse_precision_matrices(
        n_tasks=n_tasks, n_var=n_var, density=density, rand_gen=rand_gen)

    signals = nilearn.testing.generate_signals_from_precisions(
        precisions, min_samples=min_samples, max_samples=max_samples,
        rand_gen=rand_gen)

    emp_covs, _, _, _ = \
              nilearn.group_sparse_covariance.empirical_covariances(signals)

    covariances = []
    for precision in precisions:
        covariances.append(np.linalg.inv(precision))

    return signals, emp_covs, covariances, precisions


def imshow(m, title=None):
    pl.figure()
    pl.imshow(m, interpolation="nearest")
    pl.colorbar()
    if title is not None:
        pl.title(title)


if __name__ == "__main__":
    signals, emp_covs, covariances, precisions = generate_signals(
        min_samples=100000, max_samples=150000)

    # Crude precision estimator. Should work with a lot of data.
    emp_precs = []
    for k in xrange(emp_covs.shape[-1]):
        emp_precs.append(np.linalg.inv(emp_covs[..., k]))

    k = 0
    # Ground truth
    imshow(precisions[k], title="true precision")
    imshow(covariances[k], title="true covariance")
    # Estimated
    imshow(emp_covs[..., k], title="estimated covariance")
    imshow(emp_precs[k], title="estimated precision")
    # These two should take values smaller compared to the two previous.
    imshow(covariances[k] - emp_covs[..., k], title="cov. difference")
    imshow(precisions[k] - emp_precs[k],
           title="prec. difference")
    pl.show()
