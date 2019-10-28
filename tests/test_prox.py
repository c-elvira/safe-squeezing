import sys
sys.path
sys.path.append('antisparse_screening/')

import pytest
from sinatra_rescaled import _prox

import numpy as np

# --------------------------------
#
#           Testing prox
#
# --------------------------------

def all_good():
    n = 10
    w = 2.

    x = np.random.rand(n)
    (wout, _) = _prox(x, w, 1)

    #assert np.linalg.norm(x, np.Inf) <= wout
    assert wout == w


def test_prox_positive_w():

    n = 10
    w = 0.1

    for i in range(100):
        x = np.random.randn(n)
        (wout, _) = _prox(x, w, 1)
        assert np.linalg.norm(x, np.Inf) <= wout


def test_prox_negative_w():

    n = 10
    w = -10

    x = np.random.randn(n)
    (wout, _) = _prox(x, w, 1)

    print(x)
    print(wout)

    assert np.linalg.norm(x, np.Inf) <= wout