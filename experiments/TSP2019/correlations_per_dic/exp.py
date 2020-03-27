import numpy as np
import random, argparse

import matplotlib.pyplot as plt

from safesqueezing.dictionaries import sample_dictionary

parser=argparse.ArgumentParser()
parser.add_argument('--save', help='save figure', action="store_true")
args=parser.parse_args()

sizes = [i for i in range(2, 51)] + [60, 70, 75, 80, 90, 100]
n_sizes = len(sizes)
n_rep = 15000

exp_inner_Gauss = np.zeros(n_sizes)
exp_inner_Unif = np.zeros(n_sizes)
exp_inner_DCT = np.zeros(n_sizes)
exp_inner_Top = np.zeros(n_sizes)

colors = np.array([
   [228,26,28],
   [55,126,184],
   [77,175,74],
   [152,78,163]
]) / 255.

gen_norm = lambda m: sample_dictionary('norm', m, 1, True)
gen_unif = lambda m: sample_dictionary('pos_rand', m, 1, True)

for i in range(n_sizes):
   m = sizes[i]
   n = int(1.5*m)

   #Gaussian
   exp_inner_Gauss[i] = np.mean(np.array(
      [gen_norm(m).T @ gen_norm(m) for t in range(n_rep)]
   ))

   #Unif
   exp_inner_Unif[i] = np.mean(np.array(
      [gen_unif(m).T @ gen_unif(m) for t in range(n_rep)]
   ))

   # Toeplitz
   dic_top = sample_dictionary('top', m, n, True)

   for l in range(n_rep):
      #DCT
      dic1 = sample_dictionary('dct', m, n, True)
      [i1, i2] = random.sample(range(0, m), 2)
      exp_inner_DCT[i] += dic1[:, i1].T @ dic1[:, i2] / float(n_rep)

      # # Toeplitz
      [i1, i2] = random.sample(range(0, m), 2)
      exp_inner_Top[i] += dic_top[:, i1] @ dic_top[:, i2] / float(n_rep)
      

r = .6
f, ax = plt.subplots(1, 1, figsize=(r*16,r*9))
ax.plot(np.array(sizes), np.abs(exp_inner_Gauss), linewidth=2, \
   label='Gaussian', color=colors[0, :])
ax.plot(np.array(sizes), exp_inner_Unif, linewidth=2, \
   label='Uniform', color=colors[1, :])
ax.plot(np.array(sizes), np.abs(exp_inner_DCT), linewidth=2, \
   label='DCT', color=colors[2, :])
ax.plot(np.array(sizes), exp_inner_Top, linewidth=2, \
   label='Toeplitz', color=colors[3, :])

ax.legend(fontsize=16, loc='lower right')

ax.set_title("Monte-Carlo estimation of the correlation between two columns", fontsize=17)

ax.set_ylabel("$\\mathbb{E}[a_i, a_j]$", fontsize=16)
ax.set_xlabel("dimension $m$",fontsize=16)

if args.save:
   f.savefig('corr_per_dic.pdf', \
      dpi=None, facecolor='w', \
      edgecolor='w', orientation='portrait', \
      papertype=None, format=None, transparent=False, \
      bbox_inches='tight', pad_inches=0.1, \
      metadata=None)

else:
   plt.show()