import numpy as np


def gap_test_ball(dual_gap, Atu, lbd, n, ind_to_test):
    """
    """
    # -- Radius
    r = np.sqrt(2 * dual_gap) / lbd

    # -- Compute inner product A^t c
    out = np.zeros(n, dtype=int)

    # -- Test: |A^tc| > r || a_i || ?
    out[ind_to_test] += Atu > + r
    out[ind_to_test] -= Atu < - r

    return out