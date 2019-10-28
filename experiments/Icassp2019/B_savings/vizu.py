import sys, os, pickle, yaml, argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from decimal import Decimal

from mpl_toolkits.axes_grid1 import make_axes_locatable

FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'

parser=argparse.ArgumentParser()
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--version', help='numero xp', type=int,
    default=1)
args=parser.parse_args()


# Bar plot: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

FONTSIZE = 18

COLOR_ORANGE = np.array([252,78,42]) / 255.
COLOR_BLUE = np.array([4,90,141]) / 255.

## Load parameters
output_file = FOLDER + "Results/" + "gap_savingsV" + str(args.version) + ".pkl"

with open(output_file, 'rb') as f:
    [results_Ops_PG, results_Ops_FITRA, parameters] = pickle.load(f)

###########################################
#
#         Experience parameters
#
###########################################

expName = parameters["expName"]
nbRepet = parameters["nbRepet"]
version_xp  = parameters["version"]
#
listDico = parameters["listDico"]
m = parameters["m"] 
n = parameters["n"] 

a = parameters["a"]
b = parameters["b"]
step = parameters["step"]
#

print('-- Displaying ' + str(expName))
print('\tVersion ' + str(args.version))


###########################################
#
#            Manip quantites
#
###########################################

log_ten_gap_min = 2.
log_ten_gap_max = 6.
log_step = 1.
range_gap = 10**(-np.linspace(log_ten_gap_min, log_ten_gap_max, \
    float(log_ten_gap_max - log_ten_gap_min) / float(log_step)+1))

minus_log_dots = np.arange(a, b, step)
range_lbd = 10**(-minus_log_dots)


Nb_dico = len(listDico)
Nb_lbd = len(range_lbd)
#print(range_lbd)



def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            '%.1E' % Decimal(height),
            #'{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontsize=FONTSIZE-6)





def fig_bench(i_lbd):
    r = 0.6
    for i_dico in range(len(listDico)):
        
        f, ax = plt.subplots(figsize=(r*16,r*9))
        dico = listDico[i_dico]

        # results_Ops = np.zeros((nb_gap, nb_lbd, nbRepet, Nb_dico))
        to_plot_pg = np.sum(
            results_Ops_PG[:, :, :, i_dico], axis=1
        )
        to_plot_pg = np.mean(
            to_plot_pg[:, :], axis=1
            )

        to_plot_fi = np.sum(
            results_Ops_FITRA[:, :, :, i_dico], axis=1
        )
        to_plot_fi = np.mean(
            to_plot_fi[:, :], axis=1
            )


        width = 0.35  # the width of the bars
        offset  = 0.05

        x = np.arange(to_plot_fi.shape[0])
        rects1 = ax.bar(x - (width+offset)/2., np.cumsum(to_plot_fi), \
            width, label='FITRA', alpha=.9)
        rects2 = ax.bar(x + (width+offset)/2., np.cumsum(to_plot_pg), \
            width, label='sPG', alpha=.9)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_yscale('log')
        ax.set_xlabel('$-\\log_{10}($duality gap$)$', fontsize=FONTSIZE)
        ax.set_ylabel('Number of operations', fontsize=FONTSIZE)
        #ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(['%.0f' %(-np.log10(gap)) for gap in range_gap], fontsize=FONTSIZE-4)
        ax.legend(fontsize=FONTSIZE)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(FONTSIZE-4) 

        for spine in ax.spines.values():
            if spine.spine_type in ['top', 'right']:
                spine.set_visible(False)

        autolabel(ax, rects1)
        autolabel(ax, rects2)

        if args.save:
            name = 'Results/' + expName + 'V' + str(version_to_load) + '_' + dico + '.pdf'

            f.savefig(FOLDER + name, \
                dpi=None, facecolor='w', \
                edgecolor='w', orientation='portrait', \
                papertype=None, format=None, transparent=False, \
                bbox_inches='tight', pad_inches=0.1, \
                metadata=None)

if __name__ == '__main__':
    fig_bench(0)

    if not args.save:
        plt.show()


