import yaml, os, sys, pickle, argparse, decimal
import numpy as np
from packaging import version

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'

# Bar plot: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

parser=argparse.ArgumentParser()
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--version', help='numero xp', type=int,
    default=1)
args=parser.parse_args()

## Constants
NB_ALGO = 5
ROW_ITRA = 0 # ITRA
ROW_FITRA = 1 # FITRA
ROW_GR_PR = 2 # Gradient proximal
ROW_FW_VA = 3 # Frank Wolfe vanilla
ROW_FW_SC = 4 # Frank Wolfe screening

COLOR_ORANGE = np.array([252,78,42]) / 255.
COLOR_BLUE = np.array([4,90,141]) / 255.

LINEWIDTH = 2.5

DICO_POS_RAND = 0
DICO_NORM = 1
DICO_DCT = 2

## Load parameters
output_file = FOLDER + "Results/" + "BenchmarkV" + str(args.version) + ".pkl"

with open(output_file, 'rb') as f:
    [results_gap, parameters] = pickle.load(f)


expName  = str(parameters['expName'])
nbRepet   = int(parameters['nbRepet'])
vers_xp = str(parameters['version'])

m = parameters['m']
n = parameters['n']
listDico = parameters['listDico']
listLbd = parameters['listLbd']

dics_to_display = listDico


print("--------------------------")
print('Showing results V' + str(args.version))
print('Number of operations:' \
    +'%.1E' % decimal.Decimal(parameters["NbopGP"]))

print("--------------------------")

def create_roc_curve(rangegap, vec_algo_gap):

    out = 0 * rangegap
    for i in range(rangegap.size):
        out[i] = np.sum(vec_algo_gap <= rangegap[i])

    return out


def fig_bench():

    fontsize = 18

    # Display Results
    r = .8
    f, ax = plt.subplots(len(listLbd), len(dics_to_display), sharex=True, sharey=True, figsize=(r*21,r*9))
    
    plt.subplots_adjust(left = 0.1,  # the left side of the subplots of the figure
        right = 0.9,   # the right side of the subplots of the figure
        bottom = 0.05,  # the bottom of the subplots of the figure
        top = 0.95,     # the top of the subplots of the figure
        wspace = 0.05,  # the amount of width reserved for space between subplots,
        hspace = 0.05,  )

    for i_lbd in range(len(listLbd)):
        lbd = listLbd[i_lbd]

        for i_dico, dico in enumerate(dics_to_display):

            if len(dics_to_display) == 1:
                myax = ax[i_lbd]
            else:
                myax = ax[i_lbd, i_dico]


            if i_lbd == 0:
                if dico == "norm":
                    diconame = 'Gaussian'
                elif dico == "dct":
                    diconame = 'DCT'
                elif dico == "pos_rand":
                    diconame = 'Uniform'
                elif dico == "top":
                    diconame = 'Toeplitz'
                myax.set_title(diconame, fontsize=fontsize)


            #ax.plot(minus_log_dots, np.cumsum(results_complexity[ROW_ITRA, :]) / float(nbRept))

            # if version.parse(vers_xp) == version.parse("10"):
            minx = 1e-16
            maxx = 5e1

            range_gap = np.logspace(np.log(minx), np.log(maxx), 2001)

            # print(i)
            # print(results_gap[ROW_FITRA, :, i_dico, i_lbd])

            myax.plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_FITRA, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '--', \
                linewidth=LINEWIDTH, color=COLOR_BLUE, \
                label='FITRA')

            myax.plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_GR_PR, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '-', \
                linewidth=LINEWIDTH, color=COLOR_BLUE, \
                label='PGs')
            

            myax.plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_FW_VA, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '--', \
                linewidth=LINEWIDTH, color=COLOR_ORANGE, \
                label='FW')

            myax.plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_FW_SC, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '-', \
                linewidth=LINEWIDTH,
                color=COLOR_ORANGE, \
                label='FWs')


            myax.set_xlim([minx, maxx])


            myax.set_ylim([0, 102])
            myax.grid(False, which="both",ls="-")

            myax.set_xscale('log')
            if i_dico == 0:
                # ylabel = "%" + " of algorithm such that gap$^{(t)}$<gap"
                ylabel = "$\\rho_{s}(\\tau)$"
                myax.set_ylabel(ylabel, fontsize=fontsize-2)

                myax.set_yticklabels(['{}%'.format(x) \
                    for x in [0, 20, 40, 60, 80, 100]], fontsize=fontsize-4)
            
            if i_lbd == 1:
                myax.set_xlabel("$\\tau$ (Dual gap)", fontsize=fontsize-2)

                for tick in myax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize-6)
                    tick.label.set_rotation(20) 

                if i_dico == 0:
                    myax.legend(fontsize=fontsize-4, loc="lower right")




    if args.save:

        folderV = FOLDER + 'Results'

        name = 'bench_final' + 'lbd' + str(lbd).replace('.', '') \
            + 'V' + str(vers_xp) + '.pdf'

        f.savefig(folderV  + '/' + name, \
            dpi=None, facecolor='w', \
            edgecolor='w', orientation='portrait', \
            papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, \
            metadata=None)


if __name__ == '__main__':

    fig_bench()
    # fig_bench(1)
    #for i in range(len(listLbd)):
    #    fig_bench(i)

    if not args.save:
        plt.show()


