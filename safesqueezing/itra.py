import numpy as np

from safesqueezing.utils import _maxeig
from safesqueezing.prox import prox_linf

from safesqueezing.utils import TRESHOLD_IT_MONITORING



def itra(A, b, lbd, stopping):
    """
        Solves:
            xhat = argmin_x { .5 * ||A*x-b||^2_2 + lambda*||x||_inf }

        proximal gradient algorithm for approximating l_infty-norm regularized 
        least-squares problems.
 
        Parameters
        ----------
        A: np.array
            m x n matrix
        b: np.array
            signal to be represented
        lbd: float
            regularization parameter
        max_iter: int 
            maximum number of iterations
        stopping: dict
            Dictionary containing stopping criteria  
    
        Returns
        -------
        xhat: np.array
            The output of the algorithm
        monitoring: dict
            A dictionary containing ms

        Raises
        ------

        Example
        ------
        >>> xhat = itra(A, b, lbd, max_iter, tol)
    """ 
    try:
        bprint = stopping['bprint']
    except:
        bprint = True

    try:
        max_iter = stopping['max_iter']
    except:
        if bprint:
            print("ITRA: max_iter not found - default value = 1000 iterations")
        max_iter = 1000

    try:
        tol = stopping['gap_tol']
    except:
        if bprint:
            print("ITRA: tol not found - default value = 1e-6")
        tol = 1e-6

    try:
        maxOp = stopping['nbOperation']
    except:
        if bprint:
            print("ITRA: nbOperation not found - default value = Inf")
        maxOp = np.Inf

    try:
        xinit = stopping['xinit']
    except:
        if bprint:
            print("xinit not found - default value = 0...0")
        xinit = np.zeros(A.shape[1])

    try:
        lip = stopping['lip']
    except:
        if bprint:
            print("itra: lip not found - default value = None")
        lip = None

    return itra_impl(A, b, lbd, max_iter, tol, maxOp, xinit, lip=lip)


def itra_impl(A, b, lbd, max_iter, tol, maxOp, xinit, lip):

    # -- initialization
    m = A.shape[0]
    n = A.shape[1]
    xhat = xinit

    nb_mult = int(0)

    # $$ mult: m
    normY2 = np.linalg.norm(b, 2)**2
    nb_mult += m

    # $$ mult: -calling fprimal: m + 2
    # $$ mult: -calling fdual: 2m + 1
    fprimal = lambda x, u: 0.5 * np.linalg.norm(u, 2)**2 + lbd * np.linalg.norm(x, np.inf)
    fdual   = lambda uadmiss: 0.5 * (normY2 - np.linalg.norm(b - lbd * uadmiss, 2)**2)

    # $$ mult: -vecu: mn
    # $$ mult: -vecu_admissible: m + mn
    # $$ mult: -gap: m+2 + 2m+1
    vecu = b - A @ xhat
    vecu_admissible = np.copy(vecu) / np.linalg.norm(A.T @ vecu, 1)
    gap = fprimal(xhat, vecu) - fdual(vecu_admissible)
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
    AtA = A.T @ A
    Atb = A.T @ b
    nb_mult += m*n*n + m*n
  
    # -- begin main (outer) iteration
    i = 0
    while (i < max_iter) and (gap > tol) and (nb_mult < maxOp): 
        # -- 2a. gradient step
        # $$ mult: -w: n^2 + n
        w = xhat - 1. / lip * (AtA @ xhat  - Atb)
        nb_mult += n*n + n

        # -- 2b. N * log(N) method to compute truncation level
        (xhat, buf_mult) = prox_linf(w, lbd / lip)
        nb_mult += buf_mult

        # -- Computing dual gap
        # $$ mult: -vecu: mn
        # $$ mult: -vecu_admissible: m + mn
        # $$ mult: -gap: 3*m+3
        vecu = b - A @ xhat
        vecu_admissible = vecu / np.linalg.norm(A.T @ vecu, 1)
        gap = fprimal(xhat, vecu) - fdual(vecu_admissible)

        if max_iter < TRESHOLD_IT_MONITORING:
            vec_gap[i+1] = gap

        nb_mult += m*n + m+m*n + 3*m+3

        # -- Increment
        i += 1

    monitoring = {"nb_mult": nb_mult, 'gap': gap, 'lip': lip}
    if max_iter < TRESHOLD_IT_MONITORING:
        monitoring['vec_gap'] = vec_gap


    return (xhat, monitoring)
    