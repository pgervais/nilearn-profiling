
import cPickle as pickle
import glob
import os.path

import pylab as pl
import numpy as np

from nilearn.group_sparse_covariance import empirical_covariances
from common import get_cache_dir


def matshow(m, title=""):
    pl.figure()
    pl.imshow(m, interpolation="nearest")
    pl.title(title)
    pl.colorbar()


def compute_stats(cache_dir):
    l = 0  # task to plot

    data_fnames = glob.glob(os.path.join(cache_dir, "precisions_*.pickle"))
    d = pickle.load(open(data_fnames[0], "rb"))
    mean = np.zeros(d["precisions"].shape)
    acc2 = mean.copy()
##    topo_count = mean.copy()
    minimum = float("inf") * np.ones(mean.shape)
    maximum = float("-inf") * np.ones(mean.shape)

    data_fnames = data_fnames[:5]

    for data_fname in data_fnames:
        print(data_fname)
        d = pickle.load(open(data_fname, "rb"))
        mean += d["precisions"]
        acc2 += d["precisions"] ** 2
        minimum = np.where(d["precisions"] < minimum, d["precisions"], minimum)
        maximum = np.where(d["precisions"] > maximum, d["precisions"], maximum)
##        topo_count += d["precisions"] != 0

    mean /= len(data_fnames)
    acc2 /= len(data_fnames)
    var = acc2 - mean ** 2
    assert var.min() >= -1e-13
    var[var < 0] = 0  # remove very small negative values
    matshow(var[..., l], title="variance")

    std = np.sqrt(var)
    assert np.all(np.isreal(std))
    matshow(std[..., l], title="std")
    matshow(mean[..., l], title="mean")

#    matshow(mean[..., l] != 0, title="topology")
#    matshow(maximum[..., l], title="maximum")
#    matshow(minimum[..., l], title="minimum")
    matshow(maximum[..., l] - minimum[..., 0], title="ptp")

    mean_no_diag = mean.copy()
    for k in range(mean_no_diag.shape[-1]):
        mean_no_diag[..., k].flat[::mean_no_diag.shape[0] + 1] = 0

    matshow(mean_no_diag[..., l], title="mean without diagonal")

    ratio = (std / abs(mean))[..., 0]
    ratio[np.logical_not(np.isfinite(ratio))] = 0
    matshow(ratio, title="ratio")

    # load estimated covariance
    gt = pickle.load(
        open(os.path.join(cache_dir, "ground_truth.pickle"), "rb"))

    emp_covs, n_samples = empirical_covariances(gt["signals"])
    rho = 0.02

    # Estimate first-order sensitivity
    n_samples /= n_samples.sum()
    last_m = d["precisions"]
    last_m_inv = np.empty(last_m.shape)
    for k in range(last_m.shape[-1]):
        last_m_inv[..., k] = np.linalg.inv(last_m[..., k])

    norms = np.sqrt(np.sum(last_m ** 2, axis=-1))
    last_m_normed = np.empty(last_m.shape)
    for k in range(last_m.shape[-1]):
        last_m_normed[..., k] = last_m[..., k] / norms
        # set diagonal to zero
        last_m_normed[..., k].flat[::last_m_normed.shape[0] + 1] = 0

    derivative = (n_samples * (last_m_inv - emp_covs) - rho * last_m_normed)
    derivative[np.logical_not(np.isfinite(derivative))] = 0
    derivative = derivative ** 2

    # estimate second-order sensibility
    sens2 = np.empty(last_m.shape)
    for k in range(last_m.shape[-1]):
        sens2[..., k] = n_samples[k] * (
            np.dot(
                np.dot(last_m_inv[..., k],
                       derivative[..., k]),
                last_m_inv[..., k])
            )

    sens2 = np.abs(sens2 / 2.)

    matshow(np.sqrt(derivative[..., l]), title="sensitivity 1")
    matshow(np.sqrt(sens2[..., l]), title="sensitivity 2")
    matshow(np.sqrt(sens2[..., l] + derivative[..., l]),
            title="sensitivity 1+2")
    ## matshow((last_m - mean)[..., l], title="difference with mean")
    ## matshow(topo_count[..., l], title="non-zero count")
    pl.show()

if __name__ == "__main__":
    output_dir = "gsc_sensitivity"
    ## parameters = {"n_var": 10, "n_tasks": 40, "density": 0.1,
    ##                "tol": 1e-4, "alpha": 0.02}
    parameters = {"n_var": 100, "n_tasks": 40, "density": 0.1,
                  "tol": 1e-2, "alpha": 0.02}

    cache_dir = get_cache_dir(parameters, output_dir)
    compute_stats(cache_dir)
