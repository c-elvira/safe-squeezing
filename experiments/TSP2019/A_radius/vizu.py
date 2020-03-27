import yaml, os
import numpy as np

import argparse
import pickle
import matplotlib
from packaging import version
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from matplotlib.legend_handler import HandlerBase

import sys
sys.path.insert(0,'antisparse_screening')

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle='-', color=orig_handle[0],
                           linewidth=2)
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           linestyle='--', color=orig_handle[1],
                           linewidth=2)
        return [l1, l2]


FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'
VERSION = ""


parser=argparse.ArgumentParser()
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--version', help='numero xp', default=1)
args=parser.parse_args()


print("\tVersion " + str(args.version))

if args.save:
    print("\tsaving")
else:
    print("\tno saving")

blue_colors = np.array( [\
    [158,202,225], \
    [116,169,207], \
    [5,112,176], \
    [2,56,88] \
    ]) / 255.

orange_colors = np.array( [\
    [239,101,72], \
    [215,48,31], \
    [127,0,0] \
    ]) / 255.

sequence_color = np.array([
    # Vert
    [90,174,97], # Foncé
    [27,120,55], # Moyen 
    [0,68,27], # claire
    # Violet
    [153,112,171], # Foncé
    [118,42,131], # Moyen 
    [64,0,75], # claire
    ]) / 255.









# Load Results
output_file = FOLDER + "Results/" + "radiuswithST1_V" + str(args.version) + ".pkl"

with open(output_file, 'rb') as f:
    [results_gap, results_st1, parameters] = pickle.load(f)

expName  = str(parameters['expName'])
nbRept   = int(parameters['nbRepet'])

dualgap = float(parameters['dualgap'])
nbIt    = int(parameters['nbIt'])

listM = parameters['listM']
listN = parameters['listN']
listLbd = parameters['listLbd']
listDico = parameters['listDico']

nbPoint = int(parameters['nbPoint'])

vers = str(parameters['version'])
assert(version.parse(vers) >= version.parse(VERSION))

ratio = 1.
figsize = (ratio*4,ratio*4)
range_radius = np.linspace(0.001, 2, nbPoint)
fontsize = 15# 11 pour 16/9
LINEWIDTH = 3

SAVE_DIC_1 = 'pos_rand'
SAVE_LBD_2 = .2

print(listDico)

def figs_one_dic():
    # Display Results


    for i_dico in range(len(listDico)):
        dico = str(listDico[i_dico])

        if dico != SAVE_DIC_1:
            continue

        for i_size in range(len(listM)):
            m = int(listM[i_size])
            n = int(listN[i_size])

            f, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=figsize)

            listLines = []

            for i_lbd in range(len(listLbd)):
                lbd = float(listLbd[i_lbd])

                # ax.set_title("Dico: " + dico \
                #     + " size: "+ str([m,n])\
                #     + " nbRet: " + str(nbRept))

                if i_dico == 0:
                    title = "positive dictionary"
                elif i_dico == 1:
                    title = "Gaussian dictionary"

                #ax.set_title(title)

                l1 = ax.plot(range_radius, 100. * results_gap[i_dico, i_size, i_lbd, :] / float(nbRept), \
                    color=sequence_color[i_lbd, :], linewidth=LINEWIDTH, label='$\\lambda / \\lambda_{\\max}=' + str(lbd) + '$')
                l2 = ax.plot(range_radius, 100. * results_st1[i_dico, i_size, i_lbd, :] / float(nbRept), \
                    '--', color=sequence_color[3+i_lbd, :], linewidth=LINEWIDTH, \
                    label='$\\lambda / \\lambda_{\\max}=' + str(lbd) + '$')

                listLines.append([l1, l2])

                #ax.set_xlim([range_radius[0], range_radius[-1]])
                ax.set_xlim([range_radius[0], range_radius[-1]])
                ax.set_ylim([0, 100])


                deltax = 50*(range_radius[1] - range_radius[0])
                line = np.array([.5, 100. * results_gap[i_dico, i_size, i_lbd, 300] / float(nbRept)])
                deltay = 100 * (results_gap[i_dico, i_size, i_lbd, 300] \
                    - results_gap[i_dico, i_size, i_lbd, 350]) / float(nbRept)
                angle = np.arccos(deltax / np.sqrt(deltax**2 + deltay**2))
                trans_angle = plt.gca().transData.transform_angles(np.array((np.rad2deg(angle),)), \
                    line.reshape((1, 2)))[0]
                # ax.text(.5, 1.1 * 100. * results_gap[i_dico, i_size, i_lbd, 300] / float(nbRept), \
                #     '$\\lambda=' + str(lbd) + '$', \
                #     fontsize=fontsize, color=sequence_color[i_lbd, :], \
                #     rotation=-trans_angle, rotation_mode='anchor', \
                #     bbox=dict(facecolor='white', edgecolor='white', pad=0., alpha=.8))

            ax.set_xlabel('$r_0$', fontsize=fontsize+1)
            ax.set_ylabel('% of detection', fontsize=fontsize+1)

            labels = ['$\\lambda / \\lambda_{\\max}=' + str(lbd) + '$' for lbd in listLbd]
            ax.legend([(sequence_color[0+i, :], sequence_color[3+i, :]) for i in range(3)], 
                labels,
                handler_map={tuple: AnyObjectHandler()},
                fontsize=fontsize, frameon=False
                )

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize-2) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize-2) 

            for spine in ax.spines.values():
                if spine.spine_type in ['top', 'right']:
                    spine.set_visible(False)

            if args.save and dico == SAVE_DIC_1:
                name = '/exp_radius_dic' + dico + '_sizeM' + str(m) +'N' + str(n) + '.pdf'
                f.savefig(FOLDER + name, dpi=None, facecolor='w', \
                    edgecolor='w', orientation='portrait', \
                    papertype=None, format=None, transparent=False, \
                    bbox_inches='tight', pad_inches=0., \
                    frameon=None, metadata=None)
                
            else:
                pass
                plt.show()


def figs_one_lambda():

    for i_lbd in range(len(listLbd)):
        lbd = float(listLbd[i_lbd])

        print(lbd)

        for i_size in range(len(listM)):
            m = int(listM[i_size])
            n = int(listN[i_size])

            f, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=figsize)

            for i_dico in range(len(listDico)):
                dico = str(listDico[i_dico])
                if dico == 'pos_rand':
                    dico = "Uniform"
                elif dico == 'norm':
                    dico = "Gaussian"
                elif dico == 'dct':
                    dico = "DCT"
                elif dico == 'top':
                    dico = "Toeplitz"

                # ax.set_title("Dico: " + dico \
                #     + " size: "+ str([m,n])\
                #     + " nbRet: " + str(nbRept))

                if i_dico == 0:
                    title = "positive dictionary"
                elif i_dico == 1:
                    title = "Gaussian dictionary"

                #ax.set_title(title)

                ax.plot(range_radius, 100. * results_gap[i_dico, i_size, i_lbd, :] \
                    / float(nbRept), \
                    color=blue_colors[i_dico, :], linewidth=LINEWIDTH, label=dico)

                if lbd == .1:
                    ax.set_xlim([range_radius[0], .8])
                elif lbd == .2:
                    ax.set_xlim([range_radius[0], 1.25])
                else:
                    ax.set_xlim([range_radius[0], range_radius[-1]])
                ax.set_ylim([0, 100])

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize-2) 
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize-2) 

                ax.set_xlabel('$r_0$', fontsize=fontsize+1)
                ax.set_ylabel('% of detection', fontsize=fontsize+1)

                for spine in ax.spines.values():
                    if spine.spine_type in ['top', 'right']:
                        spine.set_visible(False)

            ax.legend(fontsize=fontsize, frameon=False)

            if args.save and lbd == SAVE_LBD_2:
                print("ici")
                name = '/exp_radius_lbd' + str(lbd).replace('.', '') + '_sizeM' \
                + str(m) +'N' + str(n) + '.pdf'
                f.savefig(FOLDER + name, dpi=None, facecolor='w', \
                    edgecolor='w', orientation='portrait', \
                    papertype=None, format=None, transparent=False, \
                    bbox_inches='tight', pad_inches=0., \
                    frameon=None, metadata=None)

            else:
                # print("ici")
                plt.show()


if __name__ == '__main__':

    figs_one_dic()
    figs_one_lambda()
