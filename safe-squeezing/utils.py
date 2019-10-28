import numpy as np
import os

TRESHOLD_IT_MONITORING = 1e7

def _maxeig(A, n, tol):
    """
    """
    iters = 0
    nb_mult = 0
    err = np.inf

    #-- 1. Create initialization
    bk = np.random.randn(n)
    bk = bk / np.linalg.norm(bk, 2)
    nb_mult += n

    AtA = A.transpose() @ A
    nb_mult += n**3

    while err > tol:
        iters += 1
        bkold = bk
        tmp = AtA @ bk
        val = np.linalg.norm(tmp, 2)
        bk = tmp / val
        err = np.linalg.norm(bkold-bk, 2)**2 / float(n)

    nb_mult += iters * n**3

    return val, nb_mult

def _maxeig_with_saturation(A, a0, tol):
    """
    """
    m = A.shape[0]
    n = A.shape[1]
    nb_mut = 0
    if a0 is not None:
        #   $$ ATA: 0
        AtA = np.zeros((n+1, n+1))
        #   $$ ATA: n*m*n
        AtA[:n,:n] = A.transpose() @ A
        #   $$ ATA: m * n
        AtA[:n, n] = A.transpose() @ a0
        #   $$ ATA: 0
        AtA[n, :n] = AtA[:n, n].transpose()
        #   $$ ATA: 0 a0 is normalized
        AtA[n,n] = np.linalg.norm(a0, 2)**2
        n += 1

        nb_mut += (n*m*n) + (m*n)

    else:
        AtA = A.transpose() @ A
        nb_mut += (n*m*n)


    iters = 0
    bk = np.random.randn(n)
    bk = bk / np.linalg.norm(bk, 2)
    err = np.inf
    nb_mut += n + n

    while err > tol:
        iters += 1
        bkold = bk
        #   $$ ATA: n*n
        tmp = AtA @ bk
        #   $$ ATA: n
        val = np.linalg.norm(tmp, 2)
        #   $$ ATA: n
        bk = tmp / val
        #   $$ ATA: n + n
        err = np.linalg.norm(bkold-bk, 2)**2 / float(n)

        nb_mut += (n*n) + (n) + + (n) + (n+1)

    return val, nb_mut


def _maxeig_gersh(n, r):

    return (1 + n - r)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# The notifier function
def notify(title, subtitle, message):
    t = '-title {!r}'.format(title)
    s = '-subtitle {!r}'.format(subtitle)
    m = '-message {!r}'.format(message)
    os.system('terminal-notifier {}'.format(' '.join([m, t, s])))


if __name__ == '__main__':
    notify(title    = 'A Real Notification',
       subtitle = 'with python',
       message  = 'Hello, this is me, notifying you!')