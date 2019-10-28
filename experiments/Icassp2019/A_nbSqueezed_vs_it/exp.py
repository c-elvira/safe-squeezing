import yaml, os, pickle, time
import numpy as np



from src.dictionaries import sample_dictionary
from src.pgs import pgs

from src.utils import printProgressBar

FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'


def _one_run(type_dico, m, n, range_lbd_lbdmax, maxit, dualgap):
    '''

    '''
    # -- Generate data
    matA = sample_dictionary(type_dico, m, n, True)
    yObs = np.random.randn(m)
    lambda_max = np.linalg.norm(matA.transpose() @ yObs, 1)

    # -- Create vector of results
    vec_gap = np.zeros((maxItPower2, len(range_lbd_lbdmax)))

    # -- run algorithms
    xinit = np.zeros(n)

    if bcontinuation:

        xinit = np.zeros(n)
        for i in range(len(range_lbd_lbdmax)):
            ratio_lbd = range_lbd_lbdmax[i]
            lbd = ratio_lbd * lambda_max    

            stopping = {"max_iter": maxit, "gap_tol": dualgap, "bprint": False, "monitor": True, \
            "xinit":xinit}
            (xinit, monitoring) = pgs(matA, yObs, lbd, stopping)

            for j in range(maxItPower2):
                vec_gap[j, i] = np.sum(monitoring["saturation"][:2**(j+1)-1])

    else:
        for i in range(len(range_lbd_lbdmax)):
            ratio_lbd = range_lbd_lbdmax[i]
            #print(str(i+1) + '/' + str(len(range_lbd_lbdmax)))
            lbd = ratio_lbd * lambda_max    

            #print(str(i) + '/'+ str(len(range_lbd_lbdmax)))

            # Normal experiment
            stopping = {"max_iter": maxit, "gap_tol": dualgap, "bprint": False, "monitor": True, \
            "xinit":np.zeros(n)}
            (xmanip, monitoring_manip) = pgs(matA, yObs, lbd, stopping)

            #if i == 0:
            xhighAccuracy = xmanip

            # High accuracy experiment
            stopping = {"max_iter": np.Inf, "gap_tol": 1e-6, "bprint": False, "monitor": False, \
            "xinit":xhighAccuracy}
            (xhighAccuracy, monitoringAccuracy) = pgs(matA, yObs, lbd, stopping)


            # Figure-of-merits
            for j in range(maxItPower2):
                if monitoring_manip['gap'] <= 1e-6:
                    # The initial solution is already with high accuracy
                    monitoringAccuracy["nbSat"] = monitoring_manip["nbSat"]
                vec_gap[j, i] = np.sum(monitoring_manip["saturation"][:2**(j+1)-1]) / float(monitoringAccuracy["nbSat"])



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
    
    dualgap = float(configuration['dualgap'])
    maxItPower2 = int(float(configuration['maxItPower2']))
    maxit   = 2**maxItPower2

    a = float(configuration['log_ten_lbd_lbdmax_min'])
    b = float(configuration['log_ten_lbd_lbdmax_max'])
    step = float(configuration['log_ten_lbd_lbdmax_step'])
    minus_log_dots = np.arange(a, b, step)
    range_lbd_lbdmax = 10**(-minus_log_dots)


    listM = configuration['listM']
    listN = configuration['listN']
    listDico = configuration['listDico']
    version = str(configuration['version'])

    bcontinuation = bool(configuration['continuation'])

    Nb_dico = len(listDico)
    Nb_lbd = len(range_lbd_lbdmax)

    print('-- Running ' + str(expName))
    print('\tVersion ' + str(version))

    results_gap = np.zeros((maxItPower2, Nb_lbd, nbRepet, Nb_dico))
    for i_dico in range(len(listDico)):
        dico = str(listDico[i_dico])
        print("   Dictionary " + str(dico))
        for i_size in range(len(listM)):
            m = int(listM[i_size])
            n = int(listN[i_size])

            # Loading parameters
            print("[m,n] = " + str([m,n]))
            printProgressBar(0, nbRepet, \
                prefix = '   Progress:', suffix = 'Complete', length = 25)

            ttotal = 0.
            for rep in range(nbRepet):
                t1 = time.time()
                vec_gap_one_it = _one_run(dico, m, n, range_lbd_lbdmax, maxit, dualgap)
                results_gap[:, :, rep, i_dico] += vec_gap_one_it

                ttotal += time.time() - t1
                sremaining = "Remaining: " + str(int((nbRepet - rep) * np.round(ttotal / float(rep+1), 1))) + " s"
                printProgressBar(rep+1, nbRepet, \
                    prefix = 'Progress:', suffix = 'Complete ' + sremaining, length = 25)

    # Saving results
    parameters = {
        "expName": expName, \
        "nbRepet": nbRepet, \
        "minus_log_dots": minus_log_dots, \
        "range_lbd_lbdmax": range_lbd_lbdmax, \
        "m": m, \
        "n": n, \
        "listDico": listDico, \
        "version": version, \
        "dualgap": dualgap, \
        "maxit": maxit, \
        "maxItPower2": maxItPower2, \
        "bcontinuation": bcontinuation
    }

    output_file = FOLDER + "Results/" + expName \
        + "V" + str(version) \
        + ".pkl"

    # Saving the objects:
    with open(output_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([results_gap, parameters], f)
        #np.save(output_file, results)





