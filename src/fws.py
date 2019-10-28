import numpy as np

import src.squeezing as squeezing

from src.utils import TRESHOLD_IT_MONITORING


def fws(matA, vecObs, lbd, stopping):
    """ Frank-Wolfe algorithm for antisparse coding with safe squeezing

    Solves:
        xhat = argmin_x { lambda*||x||_inf + .5 * ||A*x-b||^2_2 }
    
    Frank-Wolfe algorithm for approximating the `l_\\infty`-norm regularized 
    least-squares problems.
    Safe squeezing tests are interleaved with the iteration in order to (dynamically)
    reduce the dimension of the problem.
    The algorithm is described in

    C. Elvira and C. Herzet
    Safe squeezing for antisparse coding, submitted, 2019

    Parameters
    ----------
    matA : np.array
        m x n matrix
    b : np.array
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
    >>> xhat = fws(matA, b, lbd, max_iter, tol)
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

    return _fws_impl(matA, vecObs, lbd, max_iter, gap_tol, maxOp, xinit)


def _fws_impl(matA, vecObs, lbd, max_iter, gaptol, maxOp, xinit):
    """
    """
    (m, n) = matA.shape

    nb_mult = int(0)

    # 0c. Initialize quanties related to dictionary
        # $$ mult: -a0: 0
        # $$ mult: -norma0: 0
        # $$ mult: -normY2: m
    a0 = np.zeros(m)
    norma0 = 0.
    normY2 = np.linalg.norm(vecObs, 2)**2
    nb_mult += m

        # $$ mult: -calling fprimal: m + 2
        # $$ mult: -calling fdual: 2m + 1
    fprimal = lambda w, u: .5 * np.linalg.norm(u, 2)**2 + lbd * w
    fdual   = lambda uadmiss: .5 * (normY2 - np.linalg.norm(vecObs - lbd * uadmiss, 2)**2)

    # 0a. Initialize (w,x)
        # $$ mult: -w,x: 0
        # $$ mult: -vecu: m + mn
        # $$ mult: -Atu: nm
        # $$ mult: -a0tu: m
        # $$ mult: -Atu_ell1: 0
        # $$ mult: -vecu_admissible: m
    x = xinit
    w = np.linalg.norm(x, np.inf)
    vecu = vecObs - w * a0 - matA @ x
    Atu = matA.T @ vecu
    a0tu = a0.transpose() @ vecu
    Atu_ell1 = np.linalg.norm(Atu, 1)
    vecu_admissible = vecu / Atu_ell1
    nb_mult += m+m*n + m*n + m + m

    # 0b. Initialize quantities related to saturations
    ind_sat_pos = np.zeros(n, dtype=bool)
    ind_sat_neg = np.zeros(n, dtype=bool)
    ind_sat = ind_sat_pos | ind_sat_neg
    nb_sat = 0

    # -- 0c. Init monitoring
    if max_iter < TRESHOLD_IT_MONITORING:
        monotoring_sat = np.zeros(max_iter)
        monotoring_norma0 = np.zeros(max_iter)
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
        # -- 1a. Detect saturation
        if Atu_ell1 <= 0:
            sat = 0
        else:
                # $$ mult: -u_admiss: (n-nb_sat)
                # $$ mult: -gap_test_ball: 2
            sat = squeezing.gap_test_ball(gap, Atu[ind_sat == False] / Atu_ell1, lbd, n, ind_sat==False)
            nb_mult += (n-nb_sat) + 2

        nb_new_sat = np.sum(np.abs(sat))
        if nb_new_sat > 0:
            new_ind_pos = sat == + 1
            new_ind_neg = sat == - 1
            ind_sat_pos[new_ind_pos] = True
            ind_sat_neg[new_ind_neg] = True
            ind_sat = ind_sat_pos | ind_sat_neg
            nb_sat += nb_new_sat
            if max_iter < TRESHOLD_IT_MONITORING:
                monotoring_sat[i] = nb_new_sat

            # Update a0
                # $$ mult: -a0: 0
                # $$ mult: -norma0: m
            a0 += np.sum(matA[:, new_ind_pos], axis=1) - np.sum(matA[:, new_ind_neg], axis=1)
            norma0 = np.linalg.norm(a0, 2)
            nb_mult += 0 + m

            # Update a0tu - a few additions
                # $$ mult: -a0tu: 0
            a0tu += np.sum(Atu[new_ind_pos]) - np.sum(Atu[new_ind_neg])
            nb_mult += 0

        # -- 2. Compute descent direction
            # $$ mult: -s_w: 1
            # $$ mult: -s_x: m (sign multiplication?)
        s_w = primal / lbd 
        s_x = np.sign(Atu[ind_sat == False]) * s_w
        nb_mult += 1 + m

        # -- 3. Check if (0...0) is a better descent direction
            # $$ s_w: -condition: (n-nb_sat) + 1
        if - s_x.transpose() @ Atu[ind_sat == False] - s_w * (a0tu - lbd) > 0:
                # $$ s_w: -s_w: 0
                # $$ s_x: -s_x: (n-nb_sat)
            s_w = 0
            s_x *= 0
            nb_mult += (n-nb_sat)
        nb_mult += (n-nb_sat)+1

        # -- 3.  Compute convexe combination
            # $$ mult: -dx,dw: 0
        dx = x[ind_sat == False] - s_x
        dw = w - s_w

            # $$ mult: -Axbar: m(n-nb_sat) + m
            # $$ mult: -Adxbar: m(n-nb_sat) + m
        hres  = vecObs - (matA[:, ind_sat == False] @ s_x + s_w * a0)
        hdiff = matA[:, ind_sat == False] @ dx + dw * a0
        nb_mult += m*(n-nb_sat)+m + m*(n-nb_sat)+m

            # $$ mult: -num: m+m+1
            # $$ mult: -den: m
        num = hres @ hdiff \
            - lbd * dw
        den = hdiff.transpose() @ hdiff
        nb_mult += m+1 + m

            # $$ mult: -den: 1
        gamma = np.clip(num / den, 0., 1.)
        nb_mult += 1

        # -- 4. Update x and w
            # $$ mult: -w: 2
            # $$ mult: -x: 2(n-nb_sat)
        w = gamma * w + (1 - gamma) * s_w
        x[ind_sat == False] = gamma * x[ind_sat == False] + (1 - gamma) * s_x
        nb_mult += 2 + 2*(n-nb_sat)

        # --  Update common quantities
            # $$ mult: -vecu: m + m(n-nb_sat)
        vecu = vecObs - w * a0 - matA[:, ind_sat == False] @ x[ind_sat == False]
        nb_mult += m + m*(n-nb_sat)

            # $$ a0tu: -x: m
            # $$ Atu: -x: (n-nb_sat)m
            # $$ Atu_ell1: -x: 0
            # $$ vecu_admissible: -x: m
        a0tu = a0.transpose() @ vecu
        Atu[ind_sat == False] = matA[:, ind_sat == False].transpose() @ vecu
        Atu_ell1 = np.linalg.norm(Atu[ind_sat == False], 1) + a0tu
        vecu_admissible = vecu / Atu_ell1
        nb_mult += m + (n-nb_sat)*m + 0 + m

        # -- Compute primal and dual
            # $$ primal: -x: m+2
            # $$ gap: -x: 2m+1
        primal = fprimal(w, vecu)
        gap = primal - fdual(vecu_admissible)
        nb_mult += m+2 + 2*m+1

        # -- Monitoring
        if max_iter < TRESHOLD_IT_MONITORING:
            monotoring_norma0[i] = norma0
            monotoring_gamma[i] = gamma
            monotoring_gap[i+1] = gap

        # -- increment
        i += 1

    # -- Fin: create output
    x[ind_sat_pos] = + w
    x[ind_sat_neg] = - w

    if max_iter < TRESHOLD_IT_MONITORING:
        monotoring_gap[i:] = monotoring_gap[i-1]

    monitoring = {}
    if max_iter < TRESHOLD_IT_MONITORING:
        monitoring['saturation'] = monotoring_sat
        monitoring['norma0'] = monotoring_norma0
        monitoring['gamma'] = monotoring_gamma
        monitoring['vec_gap'] = monotoring_gap
    
    monitoring['nb_mult'] = nb_mult
    monitoring['gap'] = gap

    return (x, monitoring)


if __name__ == '__main__':
    pass

