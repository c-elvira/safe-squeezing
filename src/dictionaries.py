import numpy as np
from scipy.fftpack import dct

def sample_dictionary(type, m, n, bnormalize):

    if type == 'norm':
        return _sample_norm(m, n, bnormalize)

    elif type == 'pos_rand':
        return _sample_pos_rand(m, n, bnormalize)

    elif type == 'dct':
        return _sample_sub_dct(m, n, bnormalize)

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


