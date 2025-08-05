import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams.update({'font.size': 22})
plt.rcParams['figure.figsize'] = [20, 10]

#Comparing results
#### Electron-phonons
if len(sys.argv) > 1:
    path_to_data = sys.argv[1]
else:
    path_to_data = './'
check = '------------------------------------------------------------------------------'

def determine_q_point_cart(path_to_in, ik):
    result = [0.0, 0.0, 0.0]
    count = 1
    with open(f"{path_to_in}/scf.out", "r") as file:
        lines = file.readlines()
        for line in lines:
            if "        k(" in line:
                if ik == count:
                    parts = line.split()
                    # The k-point coordinates are typically at positions 5, 6, 7 (0-based: 4,5,6)
                    # Remove any trailing characters like ',' or ')'
                    coords = [part.rstrip(',)') for part in parts[4:7]]
                    result = [float(coord) for coord in coords]
                    break
                else:
                    count += 1
    return result

#EPW
data_epw1 = []
phononns_epw1 = []
with open(path_to_data+'epw1.out', 'r') as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        if check in line:
            start_index = index + 1
            for index, line2 in enumerate(lines[start_index:]):
                if check in line2:
                    break 
                elif (len(line2.split()) == 7)and 'iq' not in line2 and 'ik' not in line2 and 'ibnd' not in line2:
                    #print(line2)
                    frequency = float(line2.split()[5])
                    phononns_epw1.append(frequency)
                    if abs(frequency) > 1e-4 :
                        # print(frequency,float(line2.split()[6]))
                        data_epw1.append(float(line2.split()[6]))  
                         

data_epw2 = []
phononns_epw2 = []
with open(path_to_data+'epw2.out', 'r') as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        if check in line:
            start_index = index + 1
            for index, line2 in enumerate(lines[start_index:]):
                if check in line2:
                    break 
                elif (len(line2.split()) == 7)and 'iq' not in line2 and 'ik' not in line2 and 'ibnd' not in line2:
                    #print(line2)
                    frequency = float(line2.split()[5])
                    phononns_epw2.append(frequency)
                    if abs(frequency) > 1e-4 :
                        data_epw2.append(float(line2.split()[6]))         

array_epw2 = np.array(data_epw2)
array_epw1 = np.array(data_epw1)

fig, axes = plt.subplots(2, 1)

axes[0].plot(array_epw2 - array_epw1, '-', c='blue')

ylabel1=r'g$^{EPW}_{DFPT}$ - g$^{EPW}_{FD}$, meV'
ylabel2=r'|g$^{EPW}_{DFPT}$ - g$^{EPW}_{FD}$|, meV'
axes[0].set_ylabel(ylabel1)
axes[0].set_xlim([0-10, len(array_epw1)+10])
axes[0].plot(-1,0, color = 'red', label = 'iq = [1..8]')

axes[0].legend()

axes[1].plot(np.sort(np.abs(array_epw2 - array_epw1)), '-', c='blue')

axes[1].set_xlabel('Reduced index I:{i,j,m,ik,iq}')
axes[1].set_ylabel(ylabel2)
axes[1].set_xlim([0-10, len(array_epw1)+10])
axes[1].set_title('Sorted by absolute value')

threshold_colors = ['red', 'yellow', 'green']
threshold_array = [1.0, 0.5, 0.1]
for threshold_ind, threshold in enumerate(threshold_array):
    ind = np.abs(array_epw2 - array_epw1) < threshold
    threshold_number = np.sum(ind) / len(array_epw1)
    label_1 = r'Number of |δg| < {:3.2} meV: {:4.4} %'.format(threshold, threshold_number * 100)
    rect = plt.Rectangle((0, 0), threshold_number * len(array_epw2), threshold, facecolor=threshold_colors[threshold_ind], alpha=0.4, label=label_1)
    axes[1].add_patch(rect)
axes[1].legend(loc='upper left')
axes[1].set_yscale('log')
plt.savefig('epw_frozen_comp.png')

plt.show()

plt.show()
        