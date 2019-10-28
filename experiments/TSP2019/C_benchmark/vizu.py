import yaml, os, pickle, argparse, decimal

import numpy as np
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
NB_ALGO = 4
ROW_FITRA = 0   # FITRA
ROW_PGS = 1     # Projected gradient
ROW_FW = 2      # Frank Wolfe
ROW_FWS = 3     # Frank Wolfe squeezing

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
dics_to_display = [DICO_POS_RAND, DICO_NORM]



print('Budget in terms of nb of operations:' \
    +'%.1E' % decimal.Decimal(parameters["NbopGP"]))


def create_roc_curve(rangegap, vec_algo_gap):

    out = 0 * rangegap
    for i in range(rangegap.size):
        out[i] = np.sum(vec_algo_gap <= rangegap[i])

    return out


def fig_bench():

    # Display Results
    f, ax = plt.subplots(2, len(dics_to_display), sharex=True, sharey=True, figsize=(10,9))
    
    plt.subplots_adjust(left = 0.1,  # the left side of the subplots of the figure
        right = 0.9,   # the right side of the subplots of the figure
        bottom = 0.05,  # the bottom of the subplots of the figure
        top = 0.95,     # the top of the subplots of the figure
        wspace = 0.05,  # the amount of width reserved for space between subplots,
        hspace = 0.05,  )

    for i_lbd in range(len(listLbd)):
        lbd = listLbd[i_lbd]

        for i, i_dico in enumerate(dics_to_display):

            if i_dico == DICO_POS_RAND:
                dico = 'Nonnegative'
            elif i_dico == DICO_NORM:
                dico = 'Gaussian'
            elif i_dico == DICO_DCT:
                dico = 'DCT'

            if i_lbd == 0:
                ax[i_lbd, i].set_title(dico, fontsize=20)

            minx = 1e-16
            maxx = 5e1

            range_gap = np.logspace(np.log(minx), np.log(maxx), 2001)

            ax[i_lbd, i].plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_FITRA, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '--', \
                linewidth=LINEWIDTH, color=COLOR_BLUE, \
                label='FITRA')

            ax[i_lbd, i].plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_PGS, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '-', \
                linewidth=LINEWIDTH, color=COLOR_BLUE, \
                label='PGs')
            

            ax[i_lbd, i].plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_FW, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '--', \
                linewidth=LINEWIDTH, color=COLOR_ORANGE, \
                label='FW')

            ax[i_lbd, i].plot(range_gap, \
                create_roc_curve(
                    range_gap,
                    results_gap[ROW_FWS, :, i_dico, i_lbd]) \
                / float(nbRepet) * 100., \
                '-', \
                linewidth=LINEWIDTH,
                color=COLOR_ORANGE, \
                label='FWs')


            ax[i_lbd, i].set_xlim([minx, maxx])


            ax[i_lbd, i].set_ylim([0, 102])
            ax[i_lbd, i].grid(False, which="both",ls="-")

            ax[i_lbd, i].set_xscale('log')
            if i == 0:
                # ylabel = "%" + " of algorithm such that gap$^{(t)}$<gap"
                ylabel = "$\\rho_{s}(\\tau)$"
                ax[i_lbd, i].set_ylabel(ylabel, fontsize=18)

                ax[i_lbd, i].set_yticklabels(['{}%'.format(x) \
                    for x in [0, 20, 40, 60, 80, 100]], fontsize=16)
            
            if i_lbd == 1:
                ax[i_lbd, i].set_xlabel("$\\tau$ (Dual gap)", fontsize=18)

                for tick in ax[i_lbd, i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(14)
                    tick.label.set_rotation(20) 

                if i == 1:
                    ax[i_lbd, i].legend(fontsize=18, loc="center left")




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

    if not args.save:
        plt.show()


