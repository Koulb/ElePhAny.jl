import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib import cbook
import numpy as np
import pandas as pd

import re 
import argparse

plt.rcParams.update({'font.size': 22})
plt.rcParams['figure.figsize'] = [20, 10]

#Electron-phonon coupling with quadro
parser = argparse.ArgumentParser(description='Plot electron-phonon coupling data.')
parser.add_argument('--path_to_data', type=str, default='./', help='Path to the data directory')
parser.add_argument('--iband1', type=int, default=4, help='First band index')
parser.add_argument('--iband2', type=int, default=4, help='Second band index')
args = parser.parse_args()

path_to_data = args.path_to_data
iband1 = args.iband1
iband2 = args.iband2

g_dict_epw = {'{}        {}        1'.format(iband1,iband2):[],
              '{}        {}        2'.format(iband1,iband2):[],
              '{}        {}        3'.format(iband1,iband2):[],
              '{}        {}        4'.format(iband1,iband2):[],
              '{}        {}        5'.format(iband1,iband2):[],
              '{}        {}        6'.format(iband1,iband2):[]}

g_dict_epw_frozen = {'{}        {}        1'.format(iband1,iband2):[],
                     '{}        {}        2'.format(iband1,iband2):[],
                     '{}        {}        3'.format(iband1,iband2):[],
                     '{}        {}        4'.format(iband1,iband2):[],
                     '{}        {}        5'.format(iband1,iband2):[],
                     '{}        {}        6'.format(iband1,iband2):[]}

#EPW
with open(path_to_data+'epw1.out', 'r') as f:
    for line in f.readlines():
        for branch, temp_list in g_dict_epw.items():
            if branch in line:
                temp_list.append(float(line.split()[-1]))
k_epw = np.linspace(0, 1, len(g_dict_epw['{}        {}        1'.format(iband1,iband2)]))

for branch, values in g_dict_epw.items():
    plt.plot(k_epw,values, linewidth=1, alpha=0.5,color='blue')

#EPW frozen
with open(path_to_data+'epw2.out', 'r') as f:
    for line in f.readlines():
        for branch, temp_list in g_dict_epw_frozen.items():
            if branch in line:
                temp_list.append(float(line.split()[-1]))
k_epw = np.linspace(0, 1, len(g_dict_epw_frozen['{}        {}        1'.format(iband1,iband2)]))

for branch, values in g_dict_epw_frozen.items():
    plt.plot(k_epw,values, 's', linewidth=1, alpha=0.5,color='red')
    #break    

plt.axvline(0.5, linewidth=0.75, color='k', alpha=0.5)
ticks = [0, 0.5, 1]
labels = ['X', 'G', 'L']
plt.xticks(ticks= ticks, labels=labels)

plt.plot(min(k_epw)-1,0, color='blue',label='EPW-DFPT')
plt.plot(min(k_epw)-1,0, 's', color='red',label='EPW-FD')
plt.xlim([0, 1])
plt.legend()

plt.ylabel(r"|g|$_{avg}$(meV)")

plt.title('Electron-phonon coupling')
plt.savefig('el_ph_coupling.png')
plt.show()