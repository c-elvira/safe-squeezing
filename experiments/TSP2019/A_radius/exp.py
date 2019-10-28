import yaml, os, pickle, sys
import numpy as np

from src.dictionaries import sample_dictionary
from src.pgs import pgs
from src.utils import printProgressBar

FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'


def _solve(yObs, matA, lbd, gap, maxIter):
    '''
        Find a "high accuracy" solution of the antisparse problem
        Use frank wolfe
    '''
    stopping = {"max_iter": maxIter, "gap_tol":gap, 'bprint': False}
    #(x_out, _) = frank_wolfe_antisparse(matA, yObs, lbd, stopping)
    (x_out, _) = pgs(matA, yObs, lbd, stopping)

    return x_out


def _run_one_vec(yObs, matA, vecx, nbPoint, lbd):
    '''
        Given the solution of the antisparse problem,
        compute the vector of tests for various radius of gap sphere
        (centered in u^\\star).
    '''
    vecu = yObs - matA @ vecx
    vecu_admissible = vecu / np.linalg.norm(matA.transpose() @ vecu, 1) 

    # Compute Gap
    primal = .5 * np.linalg.norm(vecu, 2)**2 + lbd * np.linalg.norm(vecx, np.Inf)
    dual   = .5 * (np.linalg.norm(yObs, 2)**2 - np.linalg.norm(yObs - lbd * vecu_admissible, 2)**2)
    gap = primal - dual
    if gap < 0:
        # may happen
        gap = 1e-15

    # Perform test
    r_gap = np.sqrt(2 * gap) / lbd
    range_radius = np.linspace(0.001, 2, nbPoint)
    n = matA.shape[1]

    r_st1 = np.linalg.norm(yObs - lbd * vecu_admissible, 2)

    test_gap = 0 * range_radius
    test_st1 = 0 * range_radius
    for i, r in enumerate(range_radius):
        # -- Test: |A^tc| > r || a_i || ?
        test_gap[i] = np.sum(np.abs(matA.transpose() @ vecu) > r_gap + r)
        test_st1[i] = np.sum(np.abs(matA.transpose() @ yObs) > r_st1 + r)

    return test_gap, test_st1, gap


def _one_run(type_dico, m, n, ratiolbd, gap, nbIt, nbPoint, tol_saturation):
    '''

    '''
    matA = sample_dictionary(type_dico, m, n, True)
    yObs = np.random.randn(m)

    lambda_max = np.linalg.norm(matA.transpose() @ yObs, 1)
    lbd = ratiolbd * lambda_max

    vecx = _solve(yObs, matA, lbd, gap, nbIt)
    sat_gap, sat_ST1, gap = _run_one_vec(yObs, matA, vecx, nbPoint, lbd)

    nb_sat = _detect_nb_sat(vecx, tol_saturation)
    sat_gap /= float(nb_sat)
    sat_ST1 /= float(nb_sat)

    return sat_gap, sat_ST1, gap


def _detect_nb_sat(vecx, tol):
    '''
    '''
    xmax = np.linalg.norm(vecx, np.Inf)
    nb_sat = np.sum(np.abs(vecx) - xmax + tol > 0)

    return nb_sat


if __name__ == '__main__':
    # 1. Read configuration file
    with open(FOLDER + 'exp.yaml', 'r') as stream:
        try:
            configuration = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            raise exc

    expName  = str(configuration['expName'])
    nbRepet   = int(configuration['nbRepet'])
    
    dualgap = float(configuration['dualgap'])
    nbIt    = int(configuration['nbIt'])
    tol_saturation = float(configuration['tol_saturation'])
    
    listM = configuration['listM']
    listN = configuration['listN']
    listLbd = configuration['listLambda_over_lambdaMax']
    listDico = configuration['listDico']

    nbPoint = int(configuration['nbPoints'])

    version = str(configuration['version'])


    results_gap = np.zeros( (len(listDico), len(listM), len(listLbd), nbPoint) )
    results_st1 = np.zeros( (len(listDico), len(listM), len(listLbd), nbPoint) )

    for i_dico in range(len(listDico)):
        dico = str(listDico[i_dico])
        print("Dictionary " + str(dico))
        for i_size in range(len(listM)):
            m = int(listM[i_size])
            n = int(listN[i_size])
            for i_lbd in range(len(listLbd)):
                lbd = float(listLbd[i_lbd])
                print("[m,n] = " + str([m,n]) + " lbd=" + str(lbd) + str("lbd_max"))
                printProgressBar(0, nbRepet, \
                    prefix = 'Progress:', suffix = 'Complete', length = 25)

                vec_gap = np.zeros(nbRepet)
                for rep in range(nbRepet):
                    nbPb = 0
                    # try:
                    sat_gap, sat_ST1, gap = _one_run(dico, m, n, lbd, dualgap, nbIt, nbPoint, tol_saturation)
                    results_gap[i_dico, i_size, i_lbd, :] += sat_gap
                    results_st1[i_dico, i_size, i_lbd, :] += sat_ST1
                    vec_gap[rep] = gap
                    # except Exception as e:
                    #     nbPb + 1

                    printProgressBar(rep+1, nbRepet, \
                        prefix = 'Progress:', suffix = 'Complete', length = 25)

                print("Vector of dual gaps")
                print(vec_gap)
                print("")

                #results[i_dico, i_size, i_lbd, :] /= float(nbRepet - nbPb)

    parameters = {
        "expName": expName, \
        "nbRepet": nbRepet, \
        "dualgap": dualgap, \
        "nbIt": nbIt, \
        "listM": listM, \
        "listN": listN, \
        "listLbd": listLbd, \
        "listDico": listDico, \
        "nbPoint": nbPoint, \
        "tol_saturation": tol_saturation, \
        "version": version, \
    }

    output_file = FOLDER + "Results/" + expName + ".pkl"

    # Saving the objects:
    with open(output_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([results_gap, results_st1, parameters], f)
        #np.save(output_file, results)





