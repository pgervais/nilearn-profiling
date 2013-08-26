import pylab as pl
import numpy as np

from common import get_cache_dir, get_ground_truth, iter_outputs
from nilearn.group_sparse_covariance import (empirical_covariances,
                                              group_sparse_score)

output_dir = "gsc_varying_rho"


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
    pl.xlabel('rho')
    if not new_figure:
        pl.ylabel('')
        pl.legend(loc="best")
    else:
        pl.ylabel(label)
    if title is not None:
        pl.title(title)


def plot_benchmark1():
    """Plot various quantities obtained for varying values of rho."""
    parameters = dict(n_var=100,
                      n_tasks=5,
                      density=0.15,

                      tol=50,
#                      max_iter=500,
                      min_samples=100,
                      max_samples=150)

    cache_dir = get_cache_dir(parameters, output_dir=output_dir)
    gt = get_ground_truth(cache_dir)
    gt['precisions'] = np.dstack(gt['precisions'])

    emp_covs, n_samples, _, _ = empirical_covariances(gt['signals'])
    n_samples /= n_samples.sum()

    rho = []
    objective = []
    log_likelihood = []
    ll_penalized = []
    sparsity = []
    duality_gap = []
    kl = []

    true_covs = np.empty(gt['precisions'].shape)
    for k in range(gt['precisions'].shape[-1]):
        true_covs[..., k] = np.linalg.inv(gt['precisions'][..., k])

    for out in iter_outputs(cache_dir):
        rho.append(out['rho'])
        objective.append(- out['objective'])
        ll, llpen = group_sparse_score(out['precisions'],
                                       n_samples, true_covs, out['rho'])
        log_likelihood.append(ll)
        ll_penalized.append(llpen)
        sparsity.append(1. * (out['precisions'][..., 0] != 0).sum()
                        / out['precisions'].shape[0] ** 2)
        kl.append(distance(out['precisions'], gt['precisions']))
        duality_gap.append(out['duality_gap'])

    gt["true_sparsity"] = (1. * (gt['precisions'][..., 0] != 0).sum()
                           / gt['precisions'].shape[0] ** 2)
    title = (("n_var: {n_var}, n_tasks: {n_tasks}, "
             + "true sparsity: {true_sparsity:.2f} "
             + "\ntol: {tol:.2e} samples: {min_samples}-{max_samples}").format(
                 true_sparsity=gt["true_sparsity"],
                 **parameters))

    plot(rho, objective, label="objective", title=title)
    plot(rho, log_likelihood, label="log-likelihood", new_figure=False)
    plot(rho, ll_penalized, label="penalized L-L", new_figure=False)

    plot(rho, sparsity, label="sparsity", title=title)
    pl.hlines(gt["true_sparsity"], min(rho), max(rho))

    plot(rho, kl, label="distance", title=title)
    plot(rho, duality_gap, label='duality gap', title=title)
    pl.show()

if __name__ == "__main__":
    plot_benchmark1()
