import yaml, os, sys, pickle,  argparse
import numpy as np

from packaging import version
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
NB_ALGO = 5
ROW_ITRA = 0 # ITRA
ROW_FITRA = 1 # FITRA
ROW_GR_PR = 2 # Gradient proximal
ROW_FW_VA = 3 # Frank Wolfe vanilla
ROW_FW_SC = 4 # Frank Wolfe screening

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
output_file = FOLDER + "Results/" + "savingsV" + str(args.version) + ".pkl"

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

version_xp = parameters['version']

# if version.parse(str(version_xp)) < version.parse("4"):
#     raise Exception("Version lower that 4 are no longer supported")




def fig_gain():

    # Display Results
    f, ax = plt.subplots(len(listDico), 1, sharex=True, figsize=(6,3*len(listDico)))
    for i_dico in range(len(listDico)):

        if len(listDico) == 1:
            myax = ax
        else:
            myax = ax[i_dico]

        dico = listDico[i_dico]

        average_FITRA = np.mean(results_complexity[ROW_FITRA, :, :, i_dico], 1)
        std_FITRA = np.std(results_complexity[ROW_FITRA, :, :, i_dico], 1)
        pm = 0.434 * std_FITRA / average_FITRA

        myax.plot(minus_log_dots, \
            average_FITRA, \
            '*--', color=COLOR_PROX, linewidth=LINEWIDTH, \
            label='FITRA')
        myax.plot(minus_log_dots, \
            np.sum(results_complexity[ROW_GR_PR, :, :, i_dico], 1) / float(nbRept), \
            '*-', color=COLOR_PROX, linewidth=LINEWIDTH, \
            label='PGs')
        
        myax.plot(minus_log_dots, \
            np.sum(results_complexity[ROW_FW_VA, :, :, i_dico], 1) / float(nbRept), \
            '*--', color=COLOR_FW, linewidth=LINEWIDTH, \
            label='FW')
        myax.plot(minus_log_dots, \
            np.sum(results_complexity[ROW_FW_SC, :, :, i_dico], 1) / float(nbRept), \
            '*-', color=COLOR_FW, linewidth=LINEWIDTH, \
            label='FWs')



        myax.set_yscale('log')
        myax.set_ylabel('number of operations')

        #ax.legend(['fitra', 'gradient proximal'])
        if i_dico == len(listDico)-1:
            myax.legend(loc='lower right', ncol=2)
            myax.set_xlabel('$-\\log_{10}(\\lambda/\\lambda_\\max)$')
        else:
            pass

        myax.grid(True, which="both",ls="-")



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


