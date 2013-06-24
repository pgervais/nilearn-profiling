import nilearn.datasets
import nilearn.testing
import nilearn.region

import utils


def benchmark():
    """ """
    n_regions = 1500

    print("Loading data ...")
    adhd = nilearn.datasets.fetch_adhd()
    filename = adhd["func"][0]
    img = nilearn.utils.check_niimg(filename)
    shape = img.shape[:3]
    affine = img.get_affine()
    _ = img.get_data()  # Preload data
    print("Generating regions ...")
    regions = nilearn.testing.generate_labeled_regions_large(shape, n_regions,
                                                          affine=affine)
    signals, labels = utils.timeit(profile(nilearn.region.img_to_signals_labels)
                                   )(img, regions)
    img_r = utils.timeit(profile(nilearn.region.signals_to_img_labels)
                         )(signals, regions, order='C')


if __name__ == "__main__":
    benchmark()
