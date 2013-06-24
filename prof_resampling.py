"""Benchmark for resample_img()
"""

import time

import numpy as np

import nibabel
import nilearn.utils
import nilearn.datasets
import nilearn.resampling
import nilearn.resampling_orig

import utils  # defines profile() if not already defined


def benchmark():
    check = True
    shape = (40, 41, 42, 150)
    affine = np.eye(4)
    data = np.ndarray(shape, order="F", dtype=np.float32)
    with profile.timestamp("Data_generation"):
        data[...] = np.random.standard_normal(data.shape)
    target_shape = tuple([s * 1.26 for s in shape[:3]])
    target_affine = affine
    img = nibabel.Nifti1Image(data, affine)

    # Resample one 4D image
    if check:
        print("Resampling (original)...")
        data_orig = utils.timeit(profile(nilearn.resampling_orig.resample_img)
                            )(img, target_shape=target_shape,
                              target_affine=target_affine,
                              interpolation="continuous")

    print("Resampling (new)...")
    data = utils.timeit(profile(nilearn.resampling.resample_img)
                        )(img, target_shape=target_shape,
                          target_affine=target_affine,
                          interpolation="continuous")
    time.sleep(0.5)
    del img
    time.sleep(0.5)
    if check:
        np.testing.assert_almost_equal(data_orig.get_data(), data.get_data())
    del data
    time.sleep(0.5)


if __name__ == "__main__":
    benchmark()
