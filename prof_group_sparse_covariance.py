"""Benchmark for Honorio & Samaras algorithm for group-sparse estimation of
precision matrices."""

# Authors: Philippe Gervais
# License: simplified BSD
import numpy as np
from scipy import linalg

from nilearn.group_sparse_covariance import group_sparse_covariance
import utils  # defines profile() if not already defined


def generate_sparse_spd_matrix(n_var=1, alpha=0.95,
                               rand_gen=np.random.RandomState(0),
                               sparsity_pattern=None):
    """
    Generate a sparse symmetric definite positive matrix with the given
    dimension.

    sparsity_pattern: numpy.ndarray
        only lower triangular half is used.

    Returns
    -------
    prec: array of shape(n_var, n_var)
    """
    # Compute sparsity pattern
    if sparsity_pattern is None:
        aux = rand_gen.rand(n_var, n_var)
        aux[aux < alpha] = 0
        aux[aux >= alpha] = 1
        aux = np.tril(aux, k=-1)
        assert(np.all(np.triu(aux, k=1) == 0))
        aux = aux + aux.T
        # Permute the variables: we don't want to have any asymmetry in the
        # sparsity matrix. aux must be symmetric
        permutation = rand_gen.permutation(n_var)
        aux = aux[permutation].T[permutation]
        sparsity_pattern = aux
        np.testing.assert_almost_equal(aux, aux.T)

    # Compute sparse SDP matrix
    aux = np.zeros((n_var, n_var))
    # non-zero in sparsity_pattern means non-zero value in precision matrix
    mask = sparsity_pattern != 0
    aux[np.logical_not(mask)] = 0
    aux[mask] = .9 * rand_gen.rand(np.sum(mask))
    aux = np.tril(aux, k=-1)

    chol = -np.eye(n_var)
    chol += aux
    prec = np.dot(chol.T, chol)
    return prec, sparsity_pattern


def generate_standard_sparse_mvn(n_tasks=5, n_samples=50, n_var=10, alpha=.95,
                                 rand_gen=np.random.RandomState(0)):
    """ Generate a multivariate normal samples with sparse precision, zero
        mean and covariance diagonal equal to one.

        Parameters
        ----------
        alpha: float
            percentile of zero values in precision matrix. Values between 0 and
            1. 0 gives a full matrix, 1 gives a identity precision matrix.

        Returns
        -------
        x, array of shape (n_samples, n_var):
            samples
        prec, array of shape (n_var, n_var):
            theoretical precision
    """
    precisions = []  # List of precision matrices
    signals = []
    sparsity_pattern = None
    for n in xrange(n_tasks):
        prec, sparsity_pattern = generate_sparse_spd_matrix(
            n_var=n_var, alpha=alpha, rand_gen=rand_gen,
            sparsity_pattern=sparsity_pattern)

        # Inversion for SPD matrices
        vals, vecs = linalg.eigh(prec)
        cov = np.dot(vecs / vals, vecs.T)

        # normalize covariance (and precision)
        # in order to have unit values on the diagonal
        idelta = np.diag(np.sqrt(np.diag(cov)))
        delta = np.diag(1. / np.sqrt(np.diag(cov)))
        cov = np.dot(np.dot(delta, cov), delta)
        prec = np.dot(np.dot(idelta, prec), idelta)

        # generate samples
        x = rand_gen.multivariate_normal(mean=np.zeros(n_var), cov=cov,
                                         size=(n_samples,))
        signals.append(x)
        precisions.append(prec)

    return signals, precisions, sparsity_pattern


def benchmark():
    display = False
    rho = 1.
    signals, precisions, sparsity_pattern = generate_standard_sparse_mvn(
        n_samples=150, n_var=30, n_tasks=40)

    emp_covs = [np.dot(signal.T, signal) / signal.shape[0]
                for signal in signals]
    n_samples = [signal.shape[0] for signal in signals]

    est_precs, all_crit = utils.timeit(group_sparse_covariance)(emp_covs, rho,
                                                        n_samples, n_iter=4,
                                                        verbose=0, debug=False)
    if display:
        import pylab as pl
        for n in xrange(min(est_precs.shape[-1], 5)):
            pl.matshow(est_precs[..., n] != 0)
        pl.matshow(sparsity_pattern + np.eye(sparsity_pattern.shape[0]))
        pl.title("true sparsity pattern")
        pl.show()

if __name__ == "__main__":
    benchmark()
