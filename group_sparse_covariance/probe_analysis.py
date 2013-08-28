import cPickle as pickle
import numpy as np
import pylab as pl

# Used by pickle
from early_stopping_test import CostProbe, plot_probe_results

if __name__ == "__main__":
    rhos, probes = pickle.load(open('cost_probes.pickle', "rb"))

#    plot_probe_results(rhos, probes)

    ind_min_score = np.asarray([np.argmin(p.test_score) for p in probes])
    min_score = np.asarray([np.min(p.test_score) for p in probes])
    end_score = np.asarray([p.test_score[-1] for p in probes])
    len_score = np.asarray([len(p.test_score) for p in probes])

    # Get the value before the first increase in score.
    ind_first_min_score = []
    first_min_score = []
    for n, p in enumerate(probes):
        ind_first_min = np.where(np.diff(p.test_score) > 0)[0]
        if len(ind_first_min) == 0:
            ind_first_min = -1
        else:
            ind_first_min = ind_first_min[0]
        ind_first_min_score.append(ind_first_min)
        first_min_score.append(p.test_score[ind_first_min])

    # Plot minima values
    pl.figure()
    pl.semilogx(rhos, min_score, 'k-+', label="min")
    pl.semilogx(rhos, first_min_score, 'b-+', label="first_min")
    pl.semilogx(rhos, end_score, 'r-+', label="end")
    pl.xlabel("rho")
    pl.ylabel("score on test set")
    pl.grid()
    pl.legend(loc=0)

    # Plot minima as indices
    pl.figure()
    pl.semilogx(rhos, ind_min_score, "k-+", label="min")
    pl.semilogx(rhos, ind_first_min_score, "b-+", label="first_min")
    pl.semilogx(rhos, len_score, "r-+", label="length")
    pl.xlabel("rho")
    pl.ylabel("indices")
    pl.grid()
    pl.legend(loc=0)

    # Plot minima as time
    time_min_score = np.asarray([p.wall_clock[ind]
                                 for p, ind in zip(probes, ind_min_score)])
    time_first_min_score = np.asarray([p.wall_clock[ind]
                                       for p, ind
                                       in zip(probes, ind_first_min_score)])
    time_score = np.asarray([p.wall_clock[-1] for p in probes])

    pl.figure()
    pl.semilogx(rhos, time_min_score, "k-+", label="min")
    pl.semilogx(rhos, time_first_min_score, "b-+", label="first_min")
    pl.semilogx(rhos, time_score, "r-+", label="length")
    pl.xlabel("rho")
    pl.ylabel("time [s]")
    pl.grid()
    pl.legend(loc=0)

    pl.show()
