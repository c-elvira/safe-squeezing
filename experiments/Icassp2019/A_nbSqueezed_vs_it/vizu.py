import yaml, os
import numpy as np

import argparse
import pickle
from packaging import version
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'

parser=argparse.ArgumentParser()
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--version', help='numero xp', type=int,
    default=1)
args=parser.parse_args()

# Bar plot: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

FONTSIZE = 22

COLOR_ORANGE = np.array([252,78,42]) / 255.
COLOR_BLUE = np.array([4,90,141]) / 255.

## Load parameters
output_file = FOLDER + "Results/" + "squeezing_vs_itV" + str(args.version) + ".pkl"

with open(output_file, 'rb') as f:
    [results_gap, parameters] = pickle.load(f)


expName = parameters["expName"]
nbRepet = parameters["nbRepet"] 
minus_log_dots = parameters["minus_log_dots"]
range_lbd_lbdmax = parameters["range_lbd_lbdmax"]
m = parameters["m"] 
n = parameters["n"] 
listDico = parameters["listDico"]
version_xp  = parameters["version"]
maxItPower2 = parameters["maxItPower2"]

Nb_dico = len(listDico)
Nb_lbd = len(range_lbd_lbdmax)


def fig_bench(i_lbd):

    for i_dico in range(len(listDico)):
        
        f, ax = plt.subplots(figsize=(16,12))
        dico = listDico[i_dico]

        img = np.nanmean(results_gap[:, :, :, i_dico], 2)

        extents = [np.max(range_lbd_lbdmax) , np.min(range_lbd_lbdmax), \
        1, maxItPower2]
        im = ax.imshow(img, origin='lower', extent=extents, \
            aspect='auto', \
            cmap=plt.get_cmap('coolwarm'))

        #ax.set_xticklabels(range_lbd_lbdmax)
        ax.set_xlabel("$\\lambda / \\lambda_{\\max}$", 
            fontsize=FONTSIZE+5)

        ax.set_ylabel("$\\log_2(t)$", 
            fontsize=FONTSIZE + 5)

        ax.set_ylim([16, 1])

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(FONTSIZE)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(FONTSIZE) 

        ## Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=1., pack_start=True)
        f.add_axes(cax)
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=FONTSIZE)



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


