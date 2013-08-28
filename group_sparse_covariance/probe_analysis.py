import cPickle as pickle
import numpy as np
import pylab as pl

# Used by pickle
from early_stopping_test import CostProbe

if __name__ == "__main__":
    rhos, probes = pickle.load(open('cost_probes.pickle', "rb"))

    min_score = np.asarray([np.min(p.test_score) for p in probes])
    end_score = np.asarray([p.test_score[-1] for p in probes])

    ind_min_score = np.argsort(min_score)
    ind_end_score = np.argsort(end_score)

    pl.plot(ind_min_score, 'k-+', label="min")
    pl.plot(ind_end_score - 0.2, 'r-+', label="end")
    pl.show()
