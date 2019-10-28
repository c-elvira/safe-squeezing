import yaml, os, pickle
import numpy as np

from src.dictionaries import sample_dictionary
from src.itra import itra
from src.fitra import fitra
from src.pgs import pgs, EnumAcceleration
from src.fws import fws
from src.fw import fw

from src.utils import printProgressBar

FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'
NB_ALGO = 4
ROW_FITRA = 0   # FITRA
ROW_PGS = 1     # Projected gradient
ROW_FW = 2      # Frank Wolfe
ROW_FWS = 3     # Frank Wolfe squeezing


def _one_run(type_dico, m, n, rlbd, maxOpGP, maxOpFW):
    '''

    '''
    # -- Generate data
    matA = sample_dictionary(type_dico, m, n, True)

    yObs = np.random.randn(m)
    lambda_max = np.linalg.norm(matA.T @ yObs, 1)

    lbd = rlbd * lambda_max

    # -- Create vector of results
    vec_gap = np.zeros(NB_ALGO)


    # -- run algorithms
    stopping = {"max_iter": np.Inf, "gap_tol": 0., "nbOperation": maxOpGP, \
        "bprint": False, 'acceleration': EnumAcceleration.line_search}

    (_, monitoring) = fitra(matA, yObs, lbd, stopping)
    vec_gap[ROW_FITRA] = monitoring["gap"]
    (xhat, monitoring_sin) = pgs(matA, yObs, lbd, stopping)
    vec_gap[ROW_PGS] = monitoring_sin["gap"]


    stopping = {"max_iter": np.Inf, "gap_tol": 0, "nbOperation": maxOpFW, \
        "bprint": False}
    (_, monitoring) = fw(matA, yObs, lbd, stopping)
    vec_gap[ROW_FW] = monitoring["gap"]
    (_, monitoring) = fws(matA, yObs, lbd, stopping)
    vec_gap[ROW_FWS] = monitoring["gap"]

    return vec_gap


if __name__ == '__main__':
    # 1. Read configuration file
    with open(FOLDER + 'exp.yaml', 'r') as stream:
        try:
            configuration = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            raise exc

    expName  = str(configuration['expName'])
    nbRepet   = int(configuration['nbRepet'])
    
    NbopGP = float(configuration['NbopGP'])
    NbopFW = float(configuration['NbopFW'])

    listM = configuration['listM']
    listN = configuration['listN']
    listLbd = configuration['listLbd']
    listDico = configuration['listDico']
    version = str(configuration['version'])

    Nb_dico = len(listDico)
    Nb_lbd = len(listLbd)

    print('-- Running ' + str(expName))
    print('\tVersion ' + str(version))

    results_gap = np.nan * np.zeros((NB_ALGO, nbRepet, Nb_dico, Nb_lbd))
    for i_lbd in range(len(listLbd)):
        lbd = float(listLbd[i_lbd])
        print('lbd=' + str(lbd))
        for i_dico in range(len(listDico)):
            dico = str(listDico[i_dico])
            print("   Dictionary " + str(dico))
            for i_size in range(len(listM)):
                m = int(listM[i_size])
                n = int(listN[i_size])
                # Creating result matrix
                    # row 1: fitra
                    # row 2: prox + squeezing
                    # row 3: fw
                    # row 4: fw + squeezing

                # Loading parameters
                print("[m,n] = " + str([m,n]))
                printProgressBar(0, nbRepet, \
                    prefix = '   Progress:', suffix = 'Complete', length = 25)

                for rep in range(nbRepet):
                    vec_gap_one_it = _one_run(dico, m, n, lbd, NbopGP, NbopFW)
                    results_gap[:, rep, i_dico, i_lbd] = vec_gap_one_it

                    printProgressBar(rep+1, nbRepet, \
                        prefix = 'Progress:', suffix = 'Complete', length = 25)

    # Saving results
    parameters = {
        "expName": expName, \
        "nbRepet": nbRepet, \
        "NbopGP": NbopGP, \
        "NbopFW": NbopFW, \
        "m": m, \
        "n": n, \
        "listLbd": listLbd, \
        "listDico": listDico, \
        "version": version
    }

    output_file = FOLDER + "Results/" + expName \
        + "V" + str(version) \
        + ".pkl"

    # Saving the objects:
    with open(output_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([results_gap, parameters], f)
        #np.save(output_file, results)





