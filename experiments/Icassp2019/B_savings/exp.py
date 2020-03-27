import yaml, os, pickle, time
import numpy as np

from safesqueezing.dictionaries import sample_dictionary
from safesqueezing.fitra import fitra
from safesqueezing.pgs import pgs

from safesqueezing.utils import printProgressBar, _maxeig

FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'


###########################################
#
#         Experience parameters
#
###########################################

# Read configuration file
with open(FOLDER + 'exp.yaml', 'r') as stream:
    try:
        configuration = yaml.load(stream, Loader=yaml.Loader)
    except yaml.YAMLError as exc:
        raise exc

# Classique
expName  = str(configuration['expName'])
nbRepet   = int(configuration['nbRepet'])
version = str(configuration['version'])

listDico = configuration['listDico']
listM = configuration['listM']
listN = configuration['listN']

# Experience-specific
log_ten_gap_min = float(configuration['log_ten_gap_min'])
log_ten_gap_max = float(configuration['log_ten_gap_max'])
log_step = float(configuration['log_step'])

a = float(configuration['log_ten_lbd_lbdmax_min'])
b = float(configuration['log_ten_lbd_lbdmax_max'])
step = float(configuration['log_ten_lbd_lbdmax_step'])

# Misc
Nb_dico = len(listDico)

print('-- Running ' + str(expName))
print('\tVersion ' + str(version))


###########################################
#
#            Manip quantites
#
###########################################

range_gap = 10**(-np.linspace(log_ten_gap_min, log_ten_gap_max, \
    float(log_ten_gap_max - log_ten_gap_min) / float(log_step)+1))


# T = 9
# range_lbd = 0.95 * 10**(-2. * np.linspace(0, T, T) / (T - 1.))


#minus_log_dots = np.arange(0.025, .6, 0.025)
#range_lbd = 10**(-minus_log_dots)

minus_log_dots = np.arange(a, b, step)
range_lbd = 10**(-minus_log_dots)


nb_gap = len(range_gap)
nb_lbd = len(range_lbd)

results_Ops_PGS = np.zeros((nb_gap, nb_lbd, nbRepet, Nb_dico))
results_Ops_FITRA = np.zeros((nb_gap, nb_lbd, nbRepet, Nb_dico))


###########################################
#
#               Manip
#
###########################################

def _one_run(type_dico, m, n):
    """
    """

    # -- Generate data
    matA = sample_dictionary(type_dico, m, n, True)
    yObs = np.random.randn(m)
    lambda_max = np.linalg.norm(matA.transpose() @ yObs, 1)

    # -- Computing Lipchitz constant
    lip, lip_mult = _maxeig(matA, n, 1e-9)

    # -- Create vector of results
    vec_Ops_PGS   = np.zeros((nb_gap, nb_lbd))
    vec_Ops_FITRA = np.zeros((nb_gap, nb_lbd))

    # -- run algorithms
    Xlbds_PGS   = np.zeros((n, nb_lbd))
    Xlbds_FITRA = np.zeros((n, nb_lbd))

    for i_gap, gap in enumerate(range_gap):
        # Loop over all stopping criterion
        for i_lbd, rlbd in enumerate(range_lbd):

            # seting xinit
            if i_gap == 0:
                if i_lbd == 0:
                    xinit_PGS = np.zeros(n)
                    xinit_FITRA = np.zeros(n)
                else:
                    xinit_PGS = Xlbds_PGS[:, i_lbd-1]
                    xinit_FITRA = Xlbds_FITRA[:, i_lbd-1]
            else:
                xinit_PGS = Xlbds_PGS[:, i_lbd]
                xinit_FITRA = Xlbds_FITRA[:, i_lbd]

            # Loop over all stoping criterion
            lbd = rlbd * lambda_max

            #print(rlbd)

            # 1. Running Projected gradient
            stopping_PG = {"max_iter": np.Inf, "gap_tol": gap, \
                "bprint": False, "monitor": False, \
                "xinit":xinit_PGS}
            (xsol_PGS, monitoring_PGS) = pgs(matA, yObs, lbd, stopping_PG)

            # 2. Running Projected gradient
            stopping_FITRA = {"max_iter": np.Inf, "gap_tol": gap, \
                "bprint": False, "monitor": False, \
                "xinit":xinit_FITRA, \
                "lip": lip}
            (xsol_FITRA, monitoring_FITRA) = fitra(matA, yObs, lbd, stopping_FITRA)

            # Results
            vec_Ops_PGS[i_gap, i_lbd] = monitoring_PGS["nb_mult"]
            vec_Ops_FITRA[i_gap, i_lbd] = monitoring_FITRA["nb_mult"]

            if i_gap == 0 and i_lbd == 0:
                # Computing the Lipchtiz constant 
                # has to be done at the beginning
                vec_Ops_FITRA[i_gap, i_lbd] += lip_mult

            # Save
            Xlbds_PGS[:, i_lbd] = np.copy(xsol_PGS)
            Xlbds_FITRA[:, i_lbd] = np.copy(xsol_FITRA)

 
    return vec_Ops_PGS, vec_Ops_FITRA


if __name__ == '__main__':

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

                for_pg, for_fitra = _one_run(dico, m, n)
                results_Ops_PGS[:, :, rep, i_dico]    += for_pg
                results_Ops_FITRA[:, :, rep, i_dico] += for_fitra

                ttotal += time.time() - t1
                sremaining = "Remaining: " + str(int((nbRepet - rep) * np.round(ttotal / float(rep+1), 1))) + " s"
                printProgressBar(rep+1, nbRepet, \
                    prefix = 'Progress:', suffix = 'Complete ' + sremaining, length = 25)

    # Saving results

    parameters = {
        "expName": expName, \
        "nbRepet": nbRepet, \
        "version": version, \
        #
        "listDico": listDico, \
        "m": m, \
        "n": n, \
        #
        # "log_ten_gap_min": log_ten_gap_min, \
        # "log_ten_gap_max": log_ten_gap_max, \
        # "log_step": log_step, \
        "a": a, \
        "b": b, \
        "step": step, \
    }

    output_file = FOLDER + "Results/" + expName \
        + "V" + str(version) \
        + ".pkl"

    # Saving the objects:
    with open(output_file, 'wb') as f:
        pickle.dump([results_Ops_PGS, results_Ops_FITRA, parameters], f)






