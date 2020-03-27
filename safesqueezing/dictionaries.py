import numpy as np
from scipy.fftpack import dct

def sample_dictionary(dtype, m, n, bnormalize):

    if dtype == 'norm':
        return _sample_norm(m, n, bnormalize)

    elif dtype == 'pos_rand':
        return _sample_pos_rand(m, n, bnormalize)

    elif dtype == 'dct':
        return _sample_sub_dct(m, n, bnormalize)

    elif dtype == 'top':
        return _sample_toplitz(m, n, bnormalize)

    else:
        raise Exception("dictionary type not recognized")


def _sample_norm(m, n, bnormalize):
    # -- generate data
    matA = np.random.randn(m, n)

    if bnormalize:
        for i in range(n):
            matA[:, i] /= np.linalg.norm(matA[:, i], 2)

    return matA

def _sample_pos_rand(m, n, bnormalize):
    # -- generate data
    matA = np.random.rand(m, n)

    if bnormalize:
        for i in range(n):
            matA[:, i] /= np.linalg.norm(matA[:, i], 2)

    return matA


def _sample_sub_dct(m, n, bnormalize):
    matA = dct(np.eye(n)) # Coding matrix known to have good properties
    indices = np.random.permutation(n)
    indices = indices[:m]
    matA = matA[indices,:]

    if bnormalize:
        for i in range(n):
            matA[:, i] /= np.linalg.norm(matA[:, i], 2)

    return matA


def _sample_toplitz(m, n, bnormalize):
    gauss = lambda t: np.exp(-.5 * (t**2))

    ranget = np.linspace(-10, 10, m)
    offset = 3.
    rangemu = np.linspace(np.min(ranget)+offset, np.max(ranget)-offset, n)

    matA = np.zeros((m, n))
    for j in range(n):
        matA[:, j] = gauss(ranget - rangemu[j])

        if bnormalize:
            matA[:, j] /= np.linalg.norm(matA[:, j])

    return matA
