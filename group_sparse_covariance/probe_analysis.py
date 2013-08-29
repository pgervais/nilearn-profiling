import cPickle as pickle
import numpy as np
import pylab as pl

from early_stopping_test import CostProbe  # Used by pickle

if __name__ == "__main__":
    # Read the file written by early_stopping_test.py
    rhos, probes = pickle.load(open('cost_probes.pickle', "rb"))

    online_ind = np.asarray([p.first_max_ind for p in probes])

    ind_max_score = np.asarray([np.argmax(p.test_score) for p in probes])
    max_score = np.asarray([np.max(p.test_score) for p in probes])
    end_score = np.asarray([p.test_score[-1] for p in probes])
    len_score = np.asarray([len(p.test_score) for p in probes])

    # Get the value before the first increase in score.
    ind_first_max_score = []
    first_max_score = []
    for n, p in enumerate(probes):
        ind_first_max = np.where(np.diff(p.test_score) < 0)[0]
        if len(ind_first_max) == 0:
            ind_first_max = len(p.test_score) - 2
        else:
            ind_first_max = ind_first_max[0]
        ind_first_max += 1
        ind_first_max_score.append(ind_first_max)
        first_max_score.append(p.test_score[ind_first_max])

    # Convert indices to wall-clock time.
    time_max_score = np.asarray([p.wall_clock[ind]
                                 for p, ind in zip(probes, ind_max_score)])
    time_first_max_score = np.asarray([p.wall_clock[ind]
                                       for p, ind
                                       in zip(probes, ind_first_max_score)])
    time_score = np.asarray([p.wall_clock[-1] for p in probes])

    # Plot online and offline indices
    pl.figure()
    pl.semilogx(rhos, online_ind, "+-", label="online")
    pl.semilogx(rhos, ind_first_max_score, "x-", label="offline")
    pl.legend()
    pl.xlabel("rho")
    pl.ylabel("indices")
    pl.grid()

    # Plot maxima values on curves
    pl.figure()
    for rho, probe in zip(rhos, probes):
        pl.plot(probe.wall_clock, probe.test_score, '+-', label="%.2e" % rho)
    pl.plot(time_first_max_score, first_max_score, "ok", label="first max")
    pl.plot(time_max_score, max_score, "or", label="max")
    pl.plot(time_score, end_score, "og", label="end")
    pl.xlabel("time [s]")
    pl.ylabel('score on test set')
    pl.grid()

    # Plot maxima values
    pl.figure()
    pl.semilogx(rhos, max_score, 'k-+', label="max")
    pl.semilogx(rhos, first_max_score, 'b-+', label="first_max")
    pl.semilogx(rhos, end_score, 'r-+', label="end")
    pl.xlabel("rho")
    pl.ylabel("score on test set")
    pl.grid()
    pl.legend(loc=0)

    # Plot maxima as indices
    pl.figure()
    pl.semilogx(rhos, ind_max_score, "k-+", label="max")
    pl.semilogx(rhos, ind_first_max_score, "b-+", label="first_max")
    pl.semilogx(rhos, len_score, "r-+", label="length")
    pl.xlabel("rho")
    pl.ylabel("indices")
    pl.grid()
    pl.legend(loc=0)

    # Plot maxima as time
    pl.figure()
    pl.semilogx(rhos, time_max_score, "k-+", label="max")
    pl.semilogx(rhos, time_first_max_score, "b-+", label="first_max")
    pl.semilogx(rhos, time_score, "r-+", label="length")
    pl.xlabel("rho")
    pl.ylabel("time [s]")
    pl.grid()
    pl.legend(loc=0)

    pl.show()
