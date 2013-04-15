"""Benchmark for concat_niimgs()

"""

import sys
sys.path.append("..")

import os
import os.path as osp
import glob

import nibabel
import nisl.utils
import nisl.datasets

import utils

# profile() is defined by most profilers, these lines allows running
# the script without any profiler.
try:
    profile
except NameError:
    def profile(func):
        return func


def get_filenames():
    adhd = nisl.datasets.fetch_adhd(n_subjects=1)
    filename = adhd["func"][0]
    output_dir = "_prof_concat_niimgs"

    # create output directory
    if not osp.isdir(output_dir):
        os.mkdir("_prof_concat_niimgs")

    # list of existing individual images in output_dir
    images = glob.glob(osp.join(output_dir, "*.nii.gz"))

    return filename, output_dir, images


def init():
    filename, output_dir, _ = get_filenames()

    print ("Creating nii files...")
    fmri = nibabel.load(filename)

    data = fmri.get_data()
    affine = fmri.get_affine()
    img_number = data.shape[3]
    for n in xrange(img_number):
        print("image %d / %d" % (n + 1, img_number))
        img = nibabel.Nifti1Image(data[..., n], affine)
        nibabel.save(img, osp.join(output_dir, "image_%.3d.nii.gz" % n))


def benchmark():
    # Concatenate all individual images, time the operation
    _, _, images = get_filenames()
    if utils.cache_tools_available:
        print("Invalidating cache...")
        utils.dontneed(images)
        print("Concatenating images...")
        data = utils.timeit(profile(nisl.utils.concat_niimgs))(images)
        assert(data.shape[3] == len(images))

    print("Concatenating images...")
    data = utils.timeit(profile(nisl.utils.concat_niimgs))(images)
    assert(data.shape[3] == len(images))


if __name__ == "__main__":
    kind = None
    if len(sys.argv) == 2:
        kind = sys.argv[1]

    if kind == "init":
        init()
    else:
        benchmark()
