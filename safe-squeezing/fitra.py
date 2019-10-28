import numpy as np

from utils import _maxeig
from prox import prox_linf

from utils import TRESHOLD_IT_MONITORING


def fitra(matA, b, lbd, stopping):
    """ Implementation of the Fitra algorithm

    Solves:
    
    .. math::
        xhat = argmin_x { lambda*||x||_inf + .5 * ||A*x-b||^2_2 }

     First-order algorithm for approximating l_infty-norm regularized 
    least-squares problems. The algorithm builds on FISTA as described 
    by A. Beck and M. Teboulle 2009 and contains a fast proximal-map
    computation step to reduce the computational complexity.
 
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
    >>> xhat = FITRA(matA, b, lbd, max_iter, tol)
    """ 

    try:
        bprint = stopping['bprint']
    except:
        bprint = True

    try:
        max_iter = stopping['max_iter']
    except:
        if bprint:
            print("fitra: max_iter not found - default value = 1000 iterations")
        max_iter = 1000

    try:
        tol = stopping['gap_tol']
    except:
        if bprint:
            print("fitra: tol not found - default value = 1e-3")
        tol = 0.001

    try:
        maxOp = stopping['nbOperation']
    except:
        if bprint:
            print("fitra: nbOperation not found - default value = Inf")
        maxOp = np.Inf

    try:
        xinit = stopping['xinit']
    except:
        if bprint:
            print("xinit not found - default value = 0...0")
        xinit = np.zeros(matA.shape[1])

    try:
        lip = stopping['lip']
    except:
        if bprint:
            print("fitra: lip not found - default value = None")
        lip = None

    return fitra_impl(matA, b, lbd, max_iter, tol, maxOp, xinit, lip=None)


def fitra_impl(A, b, lbd, max_iter, tol, maxOp, xinit, lip):
    """ Implementation of Fitra

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
    lip : float
        The highest eigenvalue of `A^TA`
    
    Returns
    -------
    xhat : np.array
        The output of the algorithm
    monitoring : dict
        A dictionary containing ms
    """

    # -- initialization
    m = A.shape[0]
    n = A.shape[1]
    xkp = xinit
    yk = 0 * xkp
    xk = 0 * xkp
    tk = 1.

    nb_mult = int(0)

    # $$ mult: m
    normY2 = np.linalg.norm(b, 2)**2
    nb_mult += m

    # $$ mult: -calling fprimal: m + 2
    # $$ mult: -calling fdual: 2m + 1
    fprimal = lambda x, u: .5 * np.linalg.norm(u, 2)**2 + lbd * np.linalg.norm(x, np.inf)
    fdual   = lambda uadmiss: .5 * (normY2 - np.linalg.norm(b - lbd * uadmiss, 2)**2)
   
    # $$ mult: -vecu: mn
    # $$ mult: -vecu_admissible: m + mn
    # $$ mult: -gap: m+2 + 2m+1
    vecu = b - A @ xkp
    vecu_admissible = vecu / np.linalg.norm(A.T @ vecu, 1)
    gap = fprimal(xkp, vecu) - fdual(vecu_admissible)
    nb_mult += m*n + m+m*n + 3*m+3

    if max_iter < TRESHOLD_IT_MONITORING:
        vec_gap = np.zeros(max_iter+1)
        vec_gap[0] = gap

    # -- calculate Lipschitz constant (backtracking might be faster)
    if lip is None:
        #Lip = float(n)
        lip, buf_mult = _maxeig(A, n, 1e-9)
        nb_mult += buf_mult

    # $$ mult: -AtA: mn^2
    # $$ mult: -Atb: mn
    AtA = A.transpose() @ A
    Atb = A.transpose() @ b
    nb_mult += m*n*n + m*n
  
    # -- begin main (outer) iteration
    i = 0
    while (i < max_iter) and (gap > tol) and (nb_mult < maxOp): 
        # -- gradient step
        # $$ mult: -w: n^2 + n
        w = yk - 1. / lip * (AtA @ yk  - Atb)
        nb_mult += n*n + n
    
        # -- N * log(N) method to compute truncation level
        (xk, buf_mult) = prox_linf(w, lbd / lip)
        nb_mult += buf_mult
      
        # -- perform FISTA-like iterations    
        # $$ mult: -tkn: 3
        # $$ mult: -ykn: 1+n
        tkn = (1. + np.sqrt(1. + 4. * tk**2)) / 2.
        ykn = xk + (tk - 1.) / tkn * (xk - xkp)
        nb_mult += 3 + 1+n

        xkp = np.copy(xk)
        yk = np.copy(ykn)
        tk = np.copy(tkn)

        # -- Computing dual gap
        # $$ mult: -vecu: mn
        # $$ mult: -vecu_admissible: m + mn
        # $$ mult: -gap: 3*m+3
        vecu = b - A @ xk
        vecu_admissible = np.copy(vecu) / np.linalg.norm(A.T @ vecu, 1)
        gap = fprimal(xk, vecu) - fdual(vecu_admissible)
        nb_mult += m*n + m+m*n + 3*m+3

        if max_iter < TRESHOLD_IT_MONITORING:
            vec_gap[i+1] = gap

        # -- Increment
        i += 1

    monitoring = {"nb_mult": nb_mult, 'gap': gap, 'lip': lip}
    if max_iter < TRESHOLD_IT_MONITORING:
        monitoring['vec_gap'] = vec_gap

    return (xk, monitoring)

