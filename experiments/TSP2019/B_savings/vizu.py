import yaml, os, pickle, argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'

parser=argparse.ArgumentParser()
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--version', help='numero xp', type=int,
    default=1)
args=parser.parse_args()


# Bar plot: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

## Constants
NB_ALGO = 4
ROW_FITRA = 0   # FITRA
ROW_PGS = 1     # Projected gradient
ROW_FW = 2      # Frank Wolfe
ROW_FWS = 3     # Frank Wolfe squeezing

COLOR_ORANGE = np.array([252,78,42]) / 255.
COLOR_BLUE = np.array([4,90,141]) / 255.

ARTIC_DARK_BLUE = '#5e81ac'
ARTIC_LOW_BLUE = '#81a1c1'

ARCTIC_1 = '#8fbcbb'
ARCTIC_2 = '#88c0d0'
ARCTIC_3 = '#81a1c1'
ARCTIC_4 = '#5e81ac'

AURORA_1 = '#bf616a'
AURORA_2 = '#d08770'
AURORA_3 = '#ebcb8b'
AURORA_4 = '#a3be8c'
AURORA_5 = '#b48ead'


COLOR_PROX = COLOR_BLUE
COLOR_PROX_LOW = ARCTIC_3
COLOR_FW = COLOR_ORANGE
COLOR_FW_LOW = AURORA_3

LINEWIDTH = 2

## Load parameters
output_file = FOLDER + "Results/" + "complexityV" + str(args.version) + ".pkl"
#output_file = FOLDER + "Results/" + "complexitynorm_m100_n150" + ".pkl"
#output_file = FOLDER + "Results/" + "complexitynorm_m100_n150_it100000.pkl"

with open(output_file, 'rb') as f:
    [results_complexity, parameters] = pickle.load(f)

expName  = str(parameters['expName'])
nbRept   = int(parameters['nbRepet'])

dualgapGP = float(parameters['dualgapGP'])
dualgapFW = float(parameters['dualgapFW'])
#maxIt    = int(parameters['maxIt'])

m = parameters['m']
n = parameters['n']
minus_log_dots = parameters['minus_log_dots']
listDico = parameters['listDico']


def fig_gain():

    # Display Results
    f, ax = plt.subplots(3, 1, sharex=False, figsize=(6,9))
    for i_dico in range(len(listDico)):

        dico = listDico[i_dico]

        average_FITRA = np.mean(results_complexity[ROW_FITRA, :, :, i_dico], 1)
        std_FITRA = np.std(results_complexity[ROW_FITRA, :, :, i_dico], 1)
        pm = 0.434 * std_FITRA / average_FITRA

        ax[i_dico].plot(minus_log_dots, \
            average_FITRA, \
            '*--', color=COLOR_PROX, linewidth=LINEWIDTH, \
            label='FITRA')
        ax[i_dico].plot(minus_log_dots, \
            np.sum(results_complexity[ROW_PGS, :, :, i_dico], 1) / float(nbRept), \
            '*-', color=COLOR_PROX, linewidth=LINEWIDTH, \
            label='PGs')
        
        ax[i_dico].plot(minus_log_dots, \
            np.sum(results_complexity[ROW_FW, :, :, i_dico], 1) / float(nbRept), \
            '*--', color=COLOR_FW, linewidth=LINEWIDTH, \
            label='FW')
        ax[i_dico].plot(minus_log_dots, \
            np.sum(results_complexity[ROW_FWS, :, :, i_dico], 1) / float(nbRept), \
            '*-', color=COLOR_FW, linewidth=LINEWIDTH, \
            label='FWs')



        ax[i_dico].set_yscale('log')
        ax[i_dico].set_ylabel('number of operations')

        #ax.legend(['fitra', 'gradient proximal'])
        if i_dico == 2:
            ax[i_dico].legend(loc='lower right', ncol=2)
            ax[i_dico].set_xlabel('$-\\log_{10}(\\lambda/\\lambda_\\max)$')
        else:
            pass

        ax[i_dico].grid(True, which="both",ls="-")



    if args.save:
        name = 'Results/' + expName + 'gainV' + str(args.version) + '.pdf'

        f.savefig(FOLDER + name, \
            dpi=None, facecolor='w', \
            edgecolor='w', orientation='portrait', \
            papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, \
            metadata=None)

if __name__ == '__main__':
    fig_gain()

    if not args.save:
        plt.show()


