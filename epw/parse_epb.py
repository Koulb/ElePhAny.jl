import numpy as np
from scipy.io import FortranFile
import xml.etree.ElementTree as ET
from ase.io import read
import ase
import scipy as sc
import os
import sys
import argparse

# Helper function to read parameter from command line, else use default
def get_param(idx, default, cast_func=int):
    try:
        return cast_func(sys.argv[idx])
    except (IndexError, ValueError):
        return default

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

def parse_eigenval(path_to_calc, ik):

    tree_qe = ET.parse(path_to_calc)
    root_qe = tree_qe.getroot()
    eigenval = []
    count = 1

    for ks_energies in tree_qe.iter('ks_energies'):
        if count == ik:
            k_point = ks_energies[0].text
            eigenval = [float(val) for val in
                        re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", ks_energies[2].text)]
            break
        else:
            count += 1
    return np.array(eigenval)

#Initial parameters from command line arguments
# Usage: python parse_epb.py path_to_epw path_to_frozen nbndep mesh nat
parser = argparse.ArgumentParser(description="Parse and modify EPB files with frozen phonon data.")
parser.add_argument("--path_to_epw", default="./", type=str, help="Path to EPW directory")
parser.add_argument("--path_to_frozen", default="./displacements/", type=str, help="Path to frozen phonon data")
parser.add_argument("--nbnd", default=4, type=int, help="Number of bands")
parser.add_argument("--nbndep", default=4, type=int, help="Number of electron bands for EP")
parser.add_argument("--mesh", default=4, type=int, help="Mesh size")
parser.add_argument("--nat", default=2, type=int, help="Number of atoms")
parser.add_argument("--epb_name", default="si.epb1", type=str, help="EPB file name")

args = parser.parse_args()

path_to_epw     = args.path_to_epw
path_to_frozen  = args.path_to_frozen
nbnd            = args.nbnd
nbndep          = args.nbndep
mesh            = args.mesh
nat             = args.nat
epb_name        = args.epb_name

nbnd            = nbndep  # If you want nbnd = nbndep, otherwise add another argument
nbndep_skip     = nbnd - nbndep

nks   = mesh**3 
nqtot = mesh**3

nmodes = 3 * nat

#in reality you need to do double transpose for some unknown reasons (example is xqc and epmatq)
dtypes2 = [         '<i',                                                        #nqc
          np.dtype(('<d', (nqtot,3))),                              # xqc
          np.dtype(('<d', (nks,nbnd))),                                          # et_loc
          np.dtype(('<c16', (nqtot, nmodes, nmodes))),                           # dynq
          np.dtype(('<c16', (nqtot, nmodes, nks, nbndep, nbndep))), # epmatq
          np.dtype(('<d', (3, 3, nat))),                                         #zstar
          np.dtype(('<d', (3, 3)))                                               #epsi
         ]

#loading the data from .epb file
file_path = path_to_epw+epb_name

file = FortranFile(file_path,'r')
epb_data = file.read_record(*dtypes2)
xqc = epb_data[1]
g_arr = epb_data[4].T
et_loc = epb_data[2]
dynq= epb_data[3].T

# Get the reciprocal lattice vectors
path_to_scf = path_to_epw+'scf.out'
atoms = read(path_to_scf)
lattice_constant = atoms.get_cell_lengths_and_angles()[0]
reciprocal_vectors = atoms.cell.reciprocal().T *lattice_constant

# Get the k-points
kpoints =  atoms.calc.kpts
a_factor = 2*np.abs(reciprocal_vectors[0,0])
real_vectors = np.linalg.inv(reciprocal_vectors) /a_factor

## creating the reshufling list to correctrly parse q points
q_ph   = xqc
q_nscf = [determine_q_point_cart(path_to_epw+'/scf.out', ik) for ik in range(1, nqtot + 1)]

iq_ph_list =[]

for i_ph in range(len(q_ph)):
    for i_nscf in range(len(q_nscf)):
        q_nscf_crystal = real_vectors @ q_nscf[i_nscf]
        q_ph_crystal  = real_vectors @ q_ph[i_ph]
        check = [False, False, False]
        delta_q_all = np.abs(q_nscf_crystal-q_ph_crystal)
        for ind_q, delta_q in enumerate(delta_q_all):
            if (np.isclose(delta_q, 0, atol = 1e-5) or np.isclose(delta_q,1, atol = 1e-5)):
                check[ind_q] = True 

        if np.all(check):
            iq_ph_list.append(i_nscf)
            break

#reading the frozen phonon data for electron-phonon matrix elements
## WITH RESHUFLING
## WITH change of i and j
## gathering frozen phonon data in the same format as in the .epb file
g_frozen = np.zeros((nbndep,nbndep,nks,nmodes,nqtot),dtype=complex)
sum = 0
for ik in range(1,nqtot+1):
    for iq_ph,iq_nscf in enumerate(iq_ph_list):
        path_to_file = path_to_frozen + 'epw/braket_list_rotated_{}_{}'.format(ik,iq_nscf+1)
        data = np.loadtxt(path_to_file)
        for line in data:
            iat    = int(line[0])
            i_cart = int(line[1])
            im     = 3*(iat-1) + i_cart
            i      = int(line[3])
            j      = int(line[2])
            sum    += 1
            if (i > nbndep_skip) and (j > nbndep_skip) and (i <= nbndep+nbndep_skip) and (j <= nbndep+nbndep_skip):
                g_frozen[i-nbndep_skip-1, j-nbndep_skip-1, ik-1, im-1, iq_ph] = line[4] - 1j * line[5]


#enforcement of acoustic sum rule
# g_frozen[:,:,:,0:3,0] = 0.0

#Now need to read dyn mat and save the in .epb files as well        
dynq_frozen = np.zeros((nmodes,nmodes,nqtot),dtype=complex)

Rydberg = ase.units.Rydberg
Bohr = ase.units.Bohr

#need to  have correct reshuffling of iq points like in ep case + multiple by correct factor 
for iq_ph,iq_nscf in enumerate(iq_ph_list):    
    dynq_frozen_data = np.loadtxt(path_to_frozen+'dyn_mat/dyn_mat{}'.format(iq_nscf+1))
    dynq_frozen[:,:,iq_ph]  = dynq_frozen_data[:,0::2] + 1j*dynq_frozen_data[:,1::2]

# updating the eigenvalues
et_loc_frozen = np.zeros_like(et_loc)
# print(et_loc)
for ik in range(nks):
    e_values = parse_eigenval(path_to_frozen+'scf_0/tmp/scf.save/data-file-schema.xml', ik+1)
    et_loc_frozen[ik,:] = 2 * e_values #Ha to Ry 

#writing the update g array to the binary file .epb
file_path = path_to_epw+epb_name
file = FortranFile(file_path,'w')
epb_data_list = list(epb_data)

epb_data_list[4] = g_frozen.T
epb_data_list[3] = dynq_frozen.T
epb_data_list[2] = et_loc_frozen #et_loc

new_epb_data = tuple(epb_data_list)

file.write_record(*new_epb_data)
file.close()
