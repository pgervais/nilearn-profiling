import nisl.datasets
import nisl.testing
import nisl.region

import utils

# profile() is defined by most profilers, these lines allows running
# the script without any profiler.
try:
    profile
except NameError:
    def profile(func):
        return func


def benchmark():
    """ """
    n_regions = 1500

    print("Loading data ...")
    adhd = nisl.datasets.fetch_adhd()
    filename = adhd["func"][0]
    img = nisl.utils.check_niimg(filename)
    shape = img.shape[:3]
    affine = img.get_affine()
    _ = img.get_data()  # Preload data
    print("Generating regions ...")
    regions = nisl.testing.generate_labeled_regions_large(shape, n_regions,
                                                          affine=affine)
    signals, labels = utils.timeit(profile(nisl.region.img_to_signals_labels)
                                   )(img, regions)
    img_r = utils.timeit(profile(nisl.region.signals_to_img_labels)
                         )(signals, regions, order='C')


if __name__ == "__main__":
    benchmark()
