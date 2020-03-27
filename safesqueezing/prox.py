import numpy as np


def prox_linf(w, lbd):
    """
    """
    n = w.shape[0]
    nb_mult = int(0)

    # -- 2b. N * log(N) method to compute truncation level
    # $$ mult: -wabs: 0
    wabs = np.abs(w)
    wabs[::-1].sort()

    # $$ mult: -ws: 1 + n
    ws = (np.cumsum(wabs) - lbd) / np.linspace(1, n, n)
    alphaopt = np.max(ws)
    nb_mult += 1 + n

    if alphaopt > 0:
        # $$ mult: -xk: n
        xhat = np.minimum(np.abs(w), alphaopt) * np.sign(w) # truncation step
        nb_mult += n
    else:
        xhat = w # do not truncate

    return (xhat, nb_mult)


def prox_scaled_joint_pb(x, w, a):
    """
    """
    #the output of x is given in x
    nb_mult = int(0)

    # $$ mult: -I: 1
    wout = w
    I = np.abs(x) >= w / a
    crit1 = - np.inf
    Isum = np.sum(I)
    nb_mult += 1

    while crit1 != Isum:
        # $$ mult: -wout: 2 + 1 + #I
        # $$ mult: -I: 1
        wout = a / (a**2 + Isum) * (a * w + np.sum(np.abs(x[I])))
        I = np.abs(x) >= wout / a
        nb_mult += 3 + I.shape[0] + 1

        crit1 = Isum
        Isum = np.sum(I)

    if wout < 0:
        wout = 0
        x *= 0
        nb_mult += x.shape[0]
    else:
        x[I] = (wout / a) * np.sign(x[I]) 
        nb_mult += I.shape[0]

    return (wout, nb_mult)
