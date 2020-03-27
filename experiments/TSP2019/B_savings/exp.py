import yaml, os
import numpy as np

import time, pickle, sys

from safesqueezing.dictionaries import sample_dictionary
from safesqueezing.itra import itra
from safesqueezing.fitra import fitra
from safesqueezing.pgs import pgs
from safesqueezing.fws import fws
from safesqueezing.fw import fw

from safesqueezing.utils import printProgressBar
from safesqueezing.utils import notify

FOLDER = ""
NB_ALGO = 5
ROW_ITRA = 0 # ITRA
ROW_FITRA = 1 # FITRA
ROW_GR_PR = 2 # Gradient proximal
ROW_FW_VA = 3 # Frank Wolfe vanilla
ROW_FW_SC = 4 # Frank Wolfe screening

def _one_run_algo(matA, yObs, solver, range_lbd, lambda_max, stopping):
    """
    """
    #print("Starting " + str(solver))
    [m, n] = matA.shape

    # -- output
    vec_nb_mult = 0 * range_lbd

    # -- starting loop
    xinit = np.zeros(n)
    stopping['xinit'] = xinit
    for (i, r) in np.ndenumerate(range_lbd):
        lbd = r * lambda_max
        (xsol, mon_fitra)  = solver(matA, yObs, lbd, stopping)
        vec_nb_mult[i] = mon_fitra['nb_mult']
        stopping['xinit'] = xsol

        if i[0] == 0:
            if solver == fitra or solver == itra:
                stopping['lip'] = mon_fitra['lip']


        #print('lbd=' + str(round(lbd, 2)) + ': gap=' + str(round(mon_fitra['gap'], 8)))

    return vec_nb_mult


def _one_run(type_dico, m, n, range_lbd, dualgapGP, dualgapFW, maxiter):
    '''

    '''
    # -- Generate data
    matA = sample_dictionary(type_dico, m, n, True)
    yObs = np.random.randn(m)
    lambda_max = np.linalg.norm(matA.transpose() @ yObs, 1)

    # -- Create vector of results
    results_complexity = np.zeros((NB_ALGO, range_lbd.shape[0]))

    # -- run algorithms
    stopping = {"max_iter": maxiter, "gap_tol": dualgapGP, "bprint": False}
    results_complexity[ROW_ITRA, :]  += _one_run_algo(matA, yObs, itra, range_lbd, lambda_max, stopping)
    results_complexity[ROW_FITRA, :] += _one_run_algo(matA, yObs, fitra, range_lbd, lambda_max, stopping)
    results_complexity[ROW_GR_PR, :] += _one_run_algo(matA, yObs, pgs, range_lbd, lambda_max, stopping)

    stopping = {"max_iter": maxiter, "gap_tol": dualgapFW, "bprint": False}
    results_complexity[ROW_FW_VA, :] += _one_run_algo(matA, yObs, fws, range_lbd, lambda_max, stopping)
    results_complexity[ROW_FW_SC, :] += _one_run_algo(matA, yObs, fws, range_lbd, lambda_max, stopping)

    return results_complexity


def save_xp(results_complexity):
    parameters = {
        "expName": expName, \
        "nbRepet": nbRepet, \
        "dualgapGP": dualgapGP, \
        "dualgapFW": dualgapFW, \
        "maxIt": maxIt, \
        "m": m, \
        "n": n, \
        "minus_log_dots": minus_log_dots, \
        "listDico": listDico, \
        "version": version
    }

    output_file = FOLDER + "Results/" + expName + 'V' + str(version) \
        + ".pkl"

    # Saving the objects:
    with open(output_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([results_complexity, parameters], f)
        #np.save(output_file, results)


if __name__ == '__main__':

    # 1. Read configuration file
    with open(FOLDER + 'exp.yaml', 'r') as stream:
        try:
            configuration = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            raise exc

    expName  = str(configuration['expName'])
    nbRepet   = int(configuration['nbRepet'])
    
    dualgapGP = float(configuration['dualgap'])
    dualgapFW = float(configuration['dualgapFW'])
    try:
        maxIt   = int(configuration['maxIt'])
    except ValueError as e:

        if 'inf' in e.__str__():
            maxIt  = np.Inf
            print("Use maxIt=Inf")
        else:
            raise e

    listM = configuration['listM']
    listN = configuration['listN']
    listDico = configuration['listDico']
    version = str(configuration['version'])

    a = float(configuration['log_ten_lbd_lbdmax_min'])
    b = float(configuration['log_ten_lbd_lbdmax_max'])
    step = float(configuration['log_ten_lbd_lbdmax_step'])
    minus_log_dots = np.arange(a, b, step)
    range_lbd_lbdmax = 10**(-minus_log_dots)


    Nb_dico = len(listDico)

    results_complexity = np.zeros((NB_ALGO, range_lbd_lbdmax.shape[0], nbRepet, Nb_dico))

    # Merge
    # with open('Results/complexityV7.pkl', 'rb') as f:
    #     [results_complexity1, parameters1] = pickle.load(f)

    # with open('Results/complexityV1.pkl', 'rb') as f:
    #     [results_complexity2, parameters2] = pickle.load(f)

    # print(results_complexity1.shape)
    # print(results_complexity2.shape)

    # print(parameters1)
    # print(parameters2)

    # results_complexity[:, :, :, :3] = results_complexity1
    # results_complexity[:, :, :, 3] = results_complexity2[:, :, :, 0]

    # m = int(listM[0])
    # n = int(listN[0])
    # save_xp(results_complexity)
    # exit()

    for i_dico in range(len(listDico)):
        dico = str(listDico[i_dico])
        print("Dictionary " + str(dico))
        for i_size in range(len(listM)):
            m = int(listM[i_size])
            n = int(listN[i_size])
            # Creating result matrix
                # row 0: itra
                # row 1: fitra
                # row 2: prox + screening
                # row 3: fw
                # row 4: fw + screening

            # Loading parameters
            print("[m,n] = " + str([m,n]))
            printProgressBar(0, nbRepet, \
                prefix = 'Progress:', suffix = 'Complete', length = 25)

            ttotal = 0.
            for rep in range(nbRepet):
                nbPb = 0
                #try:
                t1 = time.time()
                results_com_one_it = _one_run(dico, m, n, range_lbd_lbdmax, dualgapGP, dualgapFW, maxIt)
                results_complexity[:, :, rep, i_dico] = results_com_one_it
                ttotal += time.time() - t1
                #except Exception as e:
                #    nbPb + 1
                sremaining = "Remaining: " + str(int((nbRepet - rep) * ttotal / float(rep+1))) + " s"
                printProgressBar(rep+1, nbRepet, \
                    prefix = 'Progress:', suffix = 'Complete ' + sremaining, length = 25)
                
                notify(title    = 'Xp ' + expName,
                    subtitle = 'Dictionary ' + dico,
                    message  = 'Rep ' + str(rep+1) + ' / ' + str(nbRepet))

                save_xp(results_complexity)

    # Saving results
    save_xp(results_complexity)






