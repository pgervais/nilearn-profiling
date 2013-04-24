import numpy as np
from scipy import ndimage


def random_matrix(shape, order="C"):
    series = np.ndarray(shape, order=order)
    series[...] = np.random.randn(*shape)
    return series


def random_mask(shape, order="C"):
    # Make random boolean array, with 2/3 of True and 1/3 of False.
    mask = np.ndarray(shape, order=order, dtype=np.bool)
    mask[...] = np.random.randint(3, size=shape) < 2
    return mask


def order_preserving_copy(data):
    if data.flags["F_CONTIGUOUS"]:
        data2 = data.T.copy().T
    else:
        data2 = data.copy()
    return data2


## @profile
def gaussian_filter():
    shape = (40, 41, 42, 100)
    smooth_sigma = np.asarray((1., 2, 3.))

    data1 = random_matrix(shape, order='F')
    data2 = order_preserving_copy(data1)

    for img in np.rollaxis(data1, -1):
        img[...] = ndimage.gaussian_filter(img, smooth_sigma)

    for n, s in enumerate(smooth_sigma):
        ndimage.gaussian_filter1d(data2, s, output=data2, axis=n)

    np.testing.assert_almost_equal(data1, data2)


## @profile
def mask():
    order = "F"
    data = random_matrix((200, 400000), order=order)
    data2 = order_preserving_copy(data)
    data[data == 2] = 0        # C order
    data2.T[data2.T == 2] = 0  # Fortran order
    assert (not(data == 2).any())
    np.testing.assert_almost_equal(data, data2)


#@profile
def copy():
    series = random_matrix((200, 400000), order="C")
    series2 = series.copy()        # best for C
    series3 = (series.T).copy().T  # best for F
    np.testing.assert_almost_equal(series2, series3, decimal=13)


@profile
def mean_square_6(series, chunk_size_row=7, chunk_size_column=1):
    # Process array chunk-by-chunk.
    # Faster than mean_square_4(), if chunk_size_row and chunk_size_column are
    # small enough. Uses almost no extra memory (contrary to mean_square_4())
    square_sum = np.zeros(series.shape[1])

    if series.flags["F_CONTIGUOUS"]:
        # Loop on columns
        chunk_size = chunk_size_column
        n_chunk = series.shape[1] / int(chunk_size)
        for chunk in xrange(n_chunk):
            square_sum[chunk * chunk_size:(chunk + 1) * chunk_size] = (
                series[..., chunk * chunk_size:(chunk + 1) * chunk_size] ** 2
                ).mean(axis=0)

        square_sum[n_chunk * chunk_size:] = (
            series[..., n_chunk * chunk_size:] ** 2).mean(axis=0)
    else:
        # Loop on rows
        chunk_size = chunk_size_row
        n_chunk = series.shape[0] / int(chunk_size)

        for chunk in xrange(n_chunk):
            square_sum += (series[chunk * chunk_size:
                                  (chunk + 1) * chunk_size, ...] ** 2
                           ).sum(axis=0)

        square_sum += (series[n_chunk * chunk_size:, ...] ** 2).sum(axis=0)
        square_sum /= series.shape[0]

    return square_sum


@profile
def mean_square_4(series):
    series4 = np.copy(series)
    series4 **= 2
    series4 = series4.mean(axis=0)
    return series4


## @profile
def mean():
    """Computation of variance on timeseries.
    Three solutions are implemented here. The last has been found to
    perform poorer than the best of the two first, on numpy 1.3 and 1.7.
    """
    series = random_matrix((200, 400000), order="C")
    series -= series.mean(axis=0)

    ## print("2")
    ## series2 = np.mean((series.T ** 2).T, axis=0)
    print("3")
    series3 = np.mean(series ** 2, axis=0)

    print("4")
    series4 = mean_square_4(series)

    print("6")
    series6 = mean_square_6(series)

    ## print("5")
    ## series5 = series.var(axis=0)

    ## np.testing.assert_almost_equal(series3, series2)
    np.testing.assert_almost_equal(series3, series4)
    ## np.testing.assert_almost_equal(series3, series5)
    np.testing.assert_almost_equal(series3, series6)


if __name__ == "__main__":
    mean()
