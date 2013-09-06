import pylab as pl
import numpy as np

from common import get_cache_dir, get_ground_truth, iter_outputs
from nilearn.group_sparse_covariance import (empirical_covariances,
                                             group_sparse_scores)

output_dir = "_gsc_varying_alpha"


def distance(m, p):
    """Distance between two matrices"""
    return np.sqrt(((m - p) ** 2).sum() / m.size)


def plot(x, y, label="", title=None, new_figure=True):
    ind = np.argsort(x)
    ind_max = np.argmax(y)
    ind_min = np.argmin(y)

    if new_figure:
        pl.figure()
    pl.semilogx(np.asarray(x)[ind], np.asarray(y)[ind], '+-', label=label)
    pl.plot(x[ind_max], y[ind_max], 'ro')
    pl.plot(x[ind_min], y[ind_min], 'go')
    pl.grid(True)
    pl.xlabel('alpha')
    if not new_figure:
        pl.ylabel('')
        pl.legend(loc="best")
    else:
        pl.ylabel(label)
    if title is not None:
        pl.title(title)


def plot_benchmark1():
    """Plot various quantities obtained for varying values of alpha."""
    parameters = dict(n_var=200,
                      n_tasks=5,
                      density=0.15,

                      tol=1e-2,
#                      max_iter=50,
                      min_samples=100,
                      max_samples=150)

    cache_dir = get_cache_dir(parameters, output_dir=output_dir)
    gt = get_ground_truth(cache_dir)
    gt['precisions'] = np.dstack(gt['precisions'])

    emp_covs, n_samples = empirical_covariances(gt['signals'])
    n_samples /= n_samples.sum()

    alpha = []
    objective = []
    log_likelihood = []
    ll_penalized = []
    sparsity = []
    kl = []

    true_covs = np.empty(gt['precisions'].shape)
    for k in range(gt['precisions'].shape[-1]):
        true_covs[..., k] = np.linalg.inv(gt['precisions'][..., k])

    for out in iter_outputs(cache_dir):
        alpha.append(out['alpha'])
        objective.append(- out['objective'][-1])
        ll, llpen = group_sparse_scores(out['precisions'],
                                       n_samples, true_covs, out['alpha'])
        log_likelihood.append(ll)
        ll_penalized.append(llpen)
        sparsity.append(1. * (out['precisions'][..., 0] != 0).sum()
                        / out['precisions'].shape[0] ** 2)
        kl.append(distance(out['precisions'], gt['precisions']))

    gt["true_sparsity"] = (1. * (gt['precisions'][..., 0] != 0).sum()
                           / gt['precisions'].shape[0] ** 2)
    title = (("n_var: {n_var}, n_tasks: {n_tasks}, "
             + "true sparsity: {true_sparsity:.2f} "
             + "\ntol: {tol:.2e} samples: {min_samples}-{max_samples}").format(
                 true_sparsity=gt["true_sparsity"],
                 **parameters))

    plot(alpha, objective, label="objective", title=title)
    plot(alpha, log_likelihood, label="log-likelihood", new_figure=False)
    plot(alpha, ll_penalized, label="penalized L-L", new_figure=False)

    plot(alpha, sparsity, label="sparsity", title=title)
    pl.hlines(gt["true_sparsity"], min(alpha), max(alpha))

    plot(alpha, kl, label="distance", title=title)
    pl.show()


if __name__ == "__main__":
    plot_benchmark1()
