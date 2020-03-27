import numpy as np
from enum import Enum

import safesqueezing.prox as prox
import safesqueezing.squeezing as squeezing

from safesqueezing.utils import _maxeig, _maxeig_with_saturation
from safesqueezing.utils import TRESHOLD_IT_MONITORING

class EnumAcceleration(Enum):
    none = 0,
    nesterov = 1,
    line_search = 2,
    nesterov2 = 3,



def pgs(matA, vecObs, lbd, params):
    """ Rescaled projected gradient for antisparse coding with safe squeezing

    Solves:
        xhat = argmin_x { .5 * ||A*x-b||^2_2 + lambda*||x||_inf }
    
    Rescaled projected gradient for approximating the `\\ell_\\infty`-norm regularized 
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
    params : dict
        Dictionary containing parameters  

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
    >>> xhat = pgs(matA, b, lbd, max_iter, tol)
    """


    (m, n) = matA.shape

    try:
        bprint = params['bprint']
    except:
        bprint = True

    try:
        max_iter = params['max_iter']
    except:
        if bprint:
            print("PGs: max_iter not found - default value = 1000 iterations")
        max_iter = 1000

    try:
        gap_tol = params['gap_tol']
    except:
        if bprint:
            print("PGs: gap tol not found - default value = 1e-3")
        gap_tol = 0.001

    try:
        maxOp = params['nbOperation']
    except:
        if bprint:
            print("PGs: nbOperation not found - default value = Inf")
        maxOp = np.Inf

    try:
        xinit = params['xinit']
    except:
        if bprint:
            print("PGs: xinit not found - default value = 0...0")
        xinit = np.zeros(matA.shape[1])

    try:
        bmonitor = params['monitor']
    except:
        if bprint:
            print("PGs: monitor not found - default value = False")
        bmonitor = False

    try:
        enumAcceleration = params['acceleration']
    except:
        if bprint:
            print("PGs: no acceleration - default value = line_search")
        enumAcceleration = EnumAcceleration.line_search

    return pgs_impl(matA, vecObs, lbd, xinit, max_iter, gap_tol, \
        maxOp, bmonitor=bmonitor, acceleration=enumAcceleration)


def pgs_impl(matA, vecObs, lbd, xinit, max_iter, \
        gaptol, maxOp, bmonitor=False, acceleration=EnumAcceleration.line_search):
    """

    """
    if bmonitor:
        print("Running PGs...")

    # -- Dimension of the problem
    (m, n) = matA.shape
    nb_mult = int(0)

    # -- Initialize quanties related to dictionary
    a0 = np.zeros(m)
    norma0 = 1. # at the begining

        # $$ mult: -normY2: m
    normY2 = np.linalg.norm(vecObs, 2)**2
    nb_mult += m

        # $$ mult: -calling fprimal: m + 2
        # $$ mult: -calling fdual: 2m + 1
    fprimal = lambda w, u: .5 * np.linalg.norm(u, 2)**2 + lbd * w
    fdual   = lambda uadmiss: .5 * (normY2 - np.linalg.norm(vecObs - lbd * uadmiss, 2)**2)

        # $$ mult: -wmax: 2
    wmax = 0.5 * normY2 / lbd
    nb_mult += 2

    # -- 0. Initialize (w,x)
        # $$ mult: -x,w: 0
        # $$ mult: -vecu: m+mn
        # $$ mult: -Atu: mn
        # $$ mult: -a0tu,Atu_ell1: 0
        # $$ mult: -vecu_admissible: m
    x = np.copy(xinit)
    w = np.linalg.norm(xinit, np.inf)
    vecu = vecObs - w * a0 - matA @ x
    Atu = matA.T @ vecu
    a0tu = 0
    Atu_ell1 = np.linalg.norm(Atu, 1)
    vecu_admissible = vecu / Atu_ell1
    nb_mult += m+m*n + m*n + m

    # -- Initialize quantities related to saturations
    ind_sat_pos = np.zeros(n, dtype=bool)
    ind_sat_neg = np.zeros(n, dtype=bool)
    ind_sat = ind_sat_pos | ind_sat_neg
    nb_sat = 0

    # -- Init monitoring
    if bmonitor:
        monotoring_sat = np.zeros(max_iter)
        monotoring_sk = np.zeros(max_iter)
        monotoring_gap  = np.zeros(max_iter+1)
        monotoring_norma0 = np.zeros(max_iter)

    if acceleration is EnumAcceleration.nesterov:
        lip, buf_mult = _maxeig(matA, n, 1e-4)
        nb_mult += buf_mult

    #####################################
    # -- Solving (easier?) saturated pb #
    #####################################
    x_red = np.copy(x)
    i = 0
    
    # -- Compute primal and dual
        # $$ mult: -gap: m+2 + 2m+1
    gap = fprimal(w, vecu) - fdual(vecu_admissible)
    nb_mult += m+2 + 2*m+1

    if bmonitor:
        monotoring_gap[0] = gap
    
    while (i < max_iter) and (gap > gaptol) and (nb_mult < maxOp):
        # ************************ #
        # -- 1. Detect saturation  #
        # ************************ #
        if Atu_ell1 <= 0:
            sat = 0
        else:
                # $$ mult: -u_admiss: (n-nb_sat)
                # $$ mult: -gap_test_ball: 2
            sat = squeezing.gap_test_ball(gap, Atu[ind_sat == False] / Atu_ell1, lbd, n, ind_sat==False)
            nb_mult += (n-nb_sat) + 2

        nb_new_sat = np.sum(np.abs(sat))

        bsat_detected = False
        if nb_new_sat > 0:
            bsat_detected = True
            new_ind_pos = sat == + 1
            new_ind_neg = sat == - 1
            ind_sat_pos[new_ind_pos] = True
            ind_sat_neg[new_ind_neg] = True
            ind_sat = ind_sat_pos | ind_sat_neg
            nb_sat += nb_new_sat
            if bmonitor:
                monotoring_sat[i] = nb_new_sat

            # -- Update a0
                # $$ mult: -a0: 0
                # $$ mult: -norma0: m
            a0 += np.sum(matA[:, new_ind_pos], axis=1) - np.sum(matA[:, new_ind_neg], axis=1)
            norma0 = np.linalg.norm(a0, 2)
            nb_mult += 0 + m

            # -- Update a0tu - a few additions
                # $$ mult: -a0tu: 0
            a0tu += np.sum(Atu[new_ind_pos]) - np.sum(Atu[new_ind_neg])
            nb_mult += 0

            x_red = x[ind_sat == False]

            if acceleration is EnumAcceleration.nesterov:
                lip, buf_mult = _maxeig_with_saturation(matA[:, ind_sat==False], a0 / norma0, 1e-4)
                nb_mult += 1 + buf_mult
                #lip = max(1., float(n - nb_sat))

        if nb_sat == n:
            w = (a0.T @ vecObs - lbd) / (norma0**2)
            w = np.clip(w, 0, +np.Inf)

            x[ind_sat_pos] = + w
            x[ind_sat_neg] = - w

            vecu = vecObs - w * a0
            a0tu = a0.T @ vecu
            if a0tu > 0:
                vecu_admissible = vecu / a0tu 

            gap = fprimal(w, vecu) - fdual(vecu_admissible)

            # -- Monitoring
            if bmonitor:
                monotoring_norma0[i] = norma0
                monotoring_gap[i+1] = gap

            i += 1
            break


        # ************************ #
        # -- 2. Update (w, x)      #
        # ************************ #
              
        if acceleration is EnumAcceleration.none:
            # Find alpha_k = 1.

            raise NotImplemented

        elif acceleration is EnumAcceleration.nesterov:
            if i == 0:
                tk = 1.
                tkOld = 1.
                x_old = np.copy(x)
                w_old = w

            # -- 2. Perform interpolation
                # $$ ratio: 1
                # $$ yred: 1 + n - nb_sat
                # $$ wbis: 1
            ratio = (tkOld - 1.) / tk
            y_red = x_red + ratio * (x_red - x_old[ind_sat==False])
            wbis  = w + ratio * (w - w_old)
            nb_mult += (1) + (1 + n - nb_sat) + (1)

                # $$ tk: 3
            tkOld = tk
            tk = (np.sqrt(4*tk**2 + 1.) + 1.) / 2.
            nb_mult += 3

            # Copy for next update
            # x_red_old = np.copy(x_red)
            # x_old is done later on
            w_old = w

            # -- 3a. Compute q^{t+1/2}
                # $$ newu: m * (n-nb_sat) + 1
                # $$ dq: m * (n-nb_sat)
                # $$ y_red: 1 + n - nb_sat
            newu = vecObs - matA[:, ind_sat == False] @ y_red  - wbis * a0
            dq = - matA[:, ind_sat == False].T @ newu
            y_red = y_red - (1. / lip) * dq
            nb_mult += (m * (n-nb_sat) + 1) + (m * (n-nb_sat)) + (1 + n - nb_sat)

            # -- 3b. Compute w_tilde^{t+1/2}
                # $$ dw_tilde: -prox: m + 1
                # $$ w_tilde: -prox: 1 + 1 + 1
            dw_tilde = (lbd - a0.T @ newu) / norma0
            w_tilde = norma0 * wbis  - (1. / lip) * dw_tilde
            nb_mult += (m + 1) + (3)

            # -- 4. Proximal step
            (w_tilde, buf_nb_mult) = prox.prox_scaled_joint_pb(y_red, w_tilde, norma0)
            nb_mult += buf_nb_mult

            # Update
            x_red = np.copy(y_red)
            w = w_tilde / norma0
            nb_mult += 1



        elif acceleration is EnumAcceleration.line_search:
            # Find alpha_k = argmin_{alpha} f(alpha x + (1 - alpha) descent)

            # -- 2a. Compute the positive scalar s^{k}
                # $$ mult: -dq: 0
                # $$ mult: -dw_tilde: 1
            dq = - Atu[ind_sat == False]
            dw_tilde = (lbd - a0tu) / norma0
            nb_mult += 0 + 1

                # $$ mult: -Ad: m*(n-nb_sat) + m
                # $$ mult: -sk: 2 + m + m +1            
            Ad = matA[:, ind_sat == False] @ dq + a0 * (dw_tilde / norma0)
            sk = (lbd * dw_tilde / norma0 - vecu.T @ Ad) / (np.linalg.norm(Ad, 2)**2)
            nb_mult += (m*(n-nb_sat) + m) + (2 + m + m + 1)

            # Clip gradient step
            sk = np.clip(sk, 1. / float(n- nb_sat), 50. / float(n- nb_sat))
            nb_mult += 2
        

            # -- 2b. Compute w_tilde^{t+1/2}
                # $$ w_tilde_half: -prox: 4
            w_tilde_half = norma0 * w  - sk * dw_tilde
            nb_mult += 2

            # -- 2c. Compute q^{t+1/2}
                # $$ mult: -x_red_half: (n-nb_sat)
            x_red_half = x_red - sk * dq
            nb_mult += n - nb_sat

            # -- 3. Proximal step
                # $$ mult: -prox: output of _prox
            (w_tilde_half, buf_nb_mult) = prox.prox_scaled_joint_pb(x_red_half, w_tilde_half, norma0) # prox modifies x_red
            nb_mult += buf_nb_mult

            # -- 4. Interpolation step (line search)
                # $$ mult: -Adiff: m*(n-nb_sat) + m + 1
                # $$ mult: -normdiff: 1
            Adiff = matA[:, ind_sat == False] @ (x_red - x_red_half) \
                + a0 * (w - w_tilde_half / norma0)
            normdiff = norma0 * w - w_tilde_half
            nb_mult += (m*(n-nb_sat) + m + 1) + (1)

                # $$ mult: -alpha_k: m + m*(n-nb_sat) + m + 1 + 2
            alpha_k = (Adiff.transpose() @ (vecObs - matA[:, ind_sat == False] @ x_red_half - a0 * (w_tilde_half / norma0))
                - lbd * normdiff / norma0
                ) / (1e-16 + np.linalg.norm(Adiff, 2)**2)
            alpha_k = np.clip(alpha_k, 0., 1.)
            nb_mult += m + m*(n-nb_sat) + m + 1 + 2

            # Security :(
            if alpha_k >= .99 or np.isnan(alpha_k):
                # Security :(
                alpha_k = 0.

            # -- 5. Update / renormalization
                # $$ mult: -x_red: 2 * (n-nb_sat)
                # $$ mult: -w_tilde: 3
            x_red   = alpha_k * x_red + (1. - alpha_k) * x_red_half
            w_tilde = alpha_k * norma0 * w + (1. - alpha_k) * w_tilde_half
            nb_mult += (2 * (n-nb_sat)) + (3)

                # $$ mult: -w: 1
            w = w_tilde / norma0
            nb_mult += 1

            if bmonitor:
                monotoring_sk[i] = sk




        # ***************** #
        #   End of update
        # ***************** #

        nb_mult += 1

        if acceleration is EnumAcceleration.nesterov:
            x_old = np.copy(x)

        x[ind_sat == False] = x_red
        x[ind_sat_pos] = + w
        x[ind_sat_neg] = - w


        # -- Compute common quantities
            # $$ mult: -vecu: m + m(n-nb_sat)
        vecu = vecObs - w * a0 - matA[:, ind_sat == False] @ x_red
        nb_mult += m + m*(n-nb_sat)

            # $$ a0tu: -x: m
            # $$ Atu: -x: (n-nb_sat)m
            # $$ Atu_ell1: -x: 0
            # $$ vecu_admissible: -x: m
        a0tu = a0.transpose() @ vecu
        Atu[ind_sat == False] = matA[:, ind_sat == False].transpose() @ vecu
        Atu_ell1 = np.linalg.norm(Atu[ind_sat == False], 1) + a0tu
        vecu_admissible = np.copy(vecu)
        if Atu_ell1 > 0:
            vecu_admissible /= Atu_ell1
            nb_mult += m
        nb_mult += m + (n-nb_sat)*m + 0

        # -- Compute primal and dual
            # $$ mult: -gap: m+2 + 2m+1
        gap = fprimal(w, vecu) - fdual(vecu_admissible)
        nb_mult += m+2 + 2*m+1

        # -- Monitoring
        if bmonitor:
            monotoring_norma0[i] = norma0
            monotoring_gap[i+1] = gap

        # -- increment
        i += 1


    # -- Fin: create output
    x[ind_sat_pos] = + w
    x[ind_sat_neg] = - w

    monitoring = {}
    if bmonitor:
        monotoring_gap[i:] = monotoring_gap[i]

        monitoring['saturation'] = monotoring_sat
        monitoring['sk'] = monotoring_sk
        monitoring['norma0'] = monotoring_norma0
        monitoring['vec_gap'] = monotoring_gap
    
    monitoring['nb_mult'] = nb_mult
    monitoring['gap'] = gap
    monitoring['nbSat'] = nb_sat

    return (x, monitoring)


if __name__ == '__main__':
    w = -1
    x = np.array([1.1])
    out = _prox(x, w)
    print(out)

