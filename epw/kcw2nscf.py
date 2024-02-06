import re
import xml.etree.ElementTree as ET
import shutil


#Specifying files for eigenvalues transition

path_to_kcw = '/scratch/apolyukhin/scripts/q-e/new_ewp_tests/si_final/kc_kcw.save/'
path_to_save = '/scratch/apolyukhin/scripts/q-e/new_ewp_tests/si_final/si_k8_converge_temp/si.save/'
path_to_out = '/scratch/apolyukhin/scripts/q-e/new_ewp_tests/si_final/si_k8_kcw_full/si.save/'

path_kp = path_to_kcw+'data-file-schema.xml'
path_qe = path_to_save +'data-file-schema.xml'
path_out = path_to_out+'data-file-schema.xml'

nks = 8**3 

#Saving first 2 lines for fix in the end 
with open(path_qe) as f:
    lines = f.readlines()

replace_lines = [lines[0],lines[1],lines[2],lines[-1]]

tree_qe = ET.parse(path_qe)
root_qe = tree_qe.getroot()

tree_kp = ET.parse(path_kp)
root_kp = tree_kp.getroot()

#Saving energies
E_kp_array = []

for ks_energies in tree_kp.iter('ks_energies'):
    k_point = ks_energies[0].text
    energies =  [val.replace('e-', 'E-0') for val in  re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?",ks_energies[2].text)]
    energies = energies[0:len(energies)//2]
    E_kp_array.append({'k_point': k_point, 'energies' : energies})

#Changing eneregies in target file 
for index_k, ks_energies in enumerate(tree_qe.iter('ks_energies')):
    k_point = ks_energies[0].text
    energies_qe_raw = ks_energies[2].text
    energies_qe = [val for val in  re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?",ks_energies[2].text)]
    energies_kp = E_kp_array[index_k]['energies']
    
    for index_e, energy_kp in enumerate(energies_kp):
        energies_qe_raw = energies_qe_raw.replace(energies_qe[index_e],energy_kp,1) 
    
    energies_qe[0:len(energies_kp)] = energies_kp
    ks_energies[2].text = energies_qe_raw
    
    # print('kpoint')
    # print(k_point)
    # print('energies')
    # print(energies_qe)
    
#Save results 
tree_qe.write(path_out, encoding='UTF-8', xml_declaration=True, short_empty_elements=False)

#Aditional change of 2 first and 2 last lines for complete agreement
with open(path_out) as f:
    lines_kp = f.readlines()

lines_kp[0]  = replace_lines[0]
lines_kp[1]  = replace_lines[1]   
lines_kp[2]  = replace_lines[2]   
lines_kp[-1] = replace_lines[3]      

with open(path_out, "w") as f:
    f.writelines(lines_kp)

# Copy files from path_to_kcw to path_to_out with desired names
for i in range(nks):
    src_file = path_to_kcw + f'wfcup{i+1}.dat'
    dest_file = path_to_out + f'wfc{i+1}.dat'
    shutil.copy(src_file, dest_file)

# shutil.copy(path_to_kcw + 'charge-density.dat', path_to_out)

####TODO
#2 need to add k_points comparison and break if grids are inconsistent
