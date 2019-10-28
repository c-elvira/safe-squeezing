import numpy as np

from utils import TRESHOLD_IT_MONITORING


def fw(matA, vecObs, lbd, stopping):
    """ Frank-Wolfe algorithm for antisparse coding

    Solves:
        xhat = argmin_x { lambda*||x||_inf + .5 * ||A*x-b||^2_2 }
    
    Frank-Wolfe algorithm for approximating the `l_\\infty`-norm regularized 
    least-squares problems.
    The algorithm is described in

    C. Elvira and C. Herzet
    Safe squeezing for antisparse coding, submitted, 2019

    Parameters
    ----------
    matA : np.array
        m x n matrix
    vecObs : np.array
        signal to be represented
    lbd : float
        regularization parameter
    max_iter : int 
        maximum number of iterations
    stopping : dict
        Dictionary containing stopping criteria  

    Returns
    -------
    xhat : np.array
        The output of the algorithm
    monitoring : dict
        A dictionary containing ms

    Raises
    ------

    Example
    ------
    >>> xhat = fw(matA, b, lbd, max_iter, tol)
    """

    try:
        bprint = stopping['bprint']
    except:
        bprint = True

    try:
        max_iter = stopping['max_iter']
    except:
        if bprint:
            print("fw: max_iter not found - default value = 1000 iterations")
        max_iter = 1000

    try:
        gap_tol = stopping['gap_tol']
    except:
        if bprint:
            print("fw: gap tol not found - default value = 1e-3")
        gap_tol = 0.001

    try:
        maxOp = stopping['nbOperation']
    except:
        if bprint:
            print("FW: nbOperation not found - default value = Inf")
        maxOp = np.Inf

    try:
        xinit = stopping['xinit']
    except:
        if bprint:
            print("xinit not found - default value = 0...0")
        xinit = np.zeros(matA.shape[1])

    return _fw_impl(matA, vecObs, lbd, max_iter, gap_tol, maxOp, xinit)


def _fw_impl(matA, vecObs, lbd, max_iter, gaptol, maxOp, xinit):
    """
    """
    # 0a. Initialize (w,x)
    (m, n) = matA.shape
    nb_mult = int(0)

    # $$ mult: -w,x: 0
    x = xinit
    w = np.linalg.norm(x, np.inf)

    # 0b. Initialize quantities related to saturations
    ind_sat_pos = np.zeros(n, dtype=bool)
    ind_sat_neg = np.zeros(n, dtype=bool)
    ind_sat = ind_sat_pos | ind_sat_neg
    nb_sat = 0
    gap = np.inf

    # 0c. Initialize quanties related to dictionary
    # $$ mult: -normY2: m
    # $$ mult: -vecu: mn
    # $$ mult: -Atu: nm
    # $$ mult: -Atu_ell1: 0
    # $$ mult: -vecu_admissible: m
    normY2 = np.linalg.norm(vecObs, 2)**2
    vecu = vecObs - matA @ x
    Atu = matA.T @ vecu
    Atu_ell1 = np.linalg.norm(Atu, 1)
    vecu_admissible = np.copy(vecu) / Atu_ell1
    nb_mult += m + m*n + m*n + m

    # $$ mult: -calling fprimal: m + 2
    # $$ mult: -calling fdual: 2m + 1
    fprimal = lambda w, u: .5 * np.linalg.norm(u, 2)**2 + lbd * w
    fdual   = lambda uadmiss: .5 * (normY2 - np.linalg.norm(vecObs - lbd * uadmiss, 2)**2)

    # -- 0d. Init monitoring
    if max_iter < TRESHOLD_IT_MONITORING:
        monotoring_sat = np.zeros(max_iter)
        monotoring_gap  = np.zeros(max_iter+1)
        monotoring_gamma = np.zeros(max_iter)

    # -- 1. Solving (easier?) saturated pb
    i = 0
    # $$ mult: -primal: m+2
    # $$ mult: -gap: 2m+1
    primal = fprimal(w, vecu)
    gap = primal - fdual(vecu_admissible)
    nb_mult += m+2 + 2*m+1

    if max_iter < TRESHOLD_IT_MONITORING:
        monotoring_gap[0] = gap
        
    while (i < max_iter) and (gap > gaptol) and (nb_mult < maxOp):
        # -- 1c. Detect saturation
        # $$ mult: -s_w: 1
        # $$ mult: -s_x: m (sign multiplication?)
        s_w = primal / lbd
        s_x = np.sign(Atu) * s_w
        nb_mult += 1 + m

        # $$ s_w: -condition: n + 1
        if - s_x.transpose() @ Atu + s_w * lbd > 0:
            # $$ s_w: -s_w: 0
            # $$ s_x: -s_x: (n-nb_sat)
            s_w = 0
            s_x *= 0
            nb_mult += n
        nb_mult += n+1

        # $$ mult: -dx,dw: 0
        dx = s_x - x
        dw = s_w - w

        # $$ mult: -Axbar: mn
        # $$ mult: -Adxbar: mn
        Axbar = matA @ x
        Adxbar = matA @ dx
        nb_mult += m*n + m*n

        # $$ mult: -num: m+m+1
        # $$ mult: -den: m
        num = vecObs.transpose() @ Adxbar \
            - Axbar.transpose() @ Adxbar \
            - lbd * dw
        den = Adxbar.transpose() @ Adxbar
        nb_mult += m+m+1 + m

        # $$ mult: -den: 1
        gamma = np.clip(num / den, 0., 1.)
        nb_mult += 1

        # $$ mult: -w: 2
        # $$ mult: -x: 2n
        w = (1 - gamma) * w + gamma * s_w
        x = (1 - gamma) * x + gamma * s_x
        nb_mult += 2 + 2*n

        # -- 1a. Compute common quantities
        # $$ mult: -vecu: m + mn
        vecu = vecObs - matA @ x
        nb_mult += m + m*n

        # $$ Atu: -x: mn
        # $$ Atu_ell1: -x: 0
        # $$ vecu_admissible: -x: m
        Atu = matA.transpose() @ vecu
        Atu_ell1 = np.linalg.norm(Atu, 1)
        vecu_admissible = vecu / Atu_ell1
        nb_mult += m*n + 0 + m

        # -- 1b. Compute primal and dual
        # $$ primal: -x: m+2
        # $$ gap: -x: 2m+1
        primal = fprimal(w, vecu)
        gap = primal - fdual(vecu_admissible)
        nb_mult += m+2 + 2*m+1

        # -- 1e. Monitoring
        if max_iter < TRESHOLD_IT_MONITORING:
            monotoring_gamma[i] = gamma
            monotoring_gap[i+1] = gap

        # -- increment
        i += 1

    # -- Fin: create output
    x[ind_sat_pos] = + w
    x[ind_sat_neg] = - w

    #monotoring_gap[i:] = monotoring_gap[i-1]

    monitoring = {}
    if max_iter < TRESHOLD_IT_MONITORING:
        monitoring['saturation'] = monotoring_sat
        monitoring['gamma'] = monotoring_gamma
        monitoring['vec_gap'] = monotoring_gap
    
    monitoring['nb_mult'] = nb_mult
    monitoring['gap'] = gap

    return (x, monitoring)


if __name__ == '__main__':
    pass

