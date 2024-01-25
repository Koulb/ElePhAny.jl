import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.io import FortranFile
import xml.etree.ElementTree as ET
from ase.io import read
import ase
import scipy as sc
import phonopy

#Initial parameters
path_to_epw = '/scratch/apolyukhin/scripts/q-e/new_ewp_tests/si_test_2/'
path_to_q_points = '/home/apolyukhin/Notebooks/q_data/'
path_to_frozen = '/scratch/apolyukhin/julia_tests/qe_inputs_small_test/displacements/'#CHANGE TO _ibrav

nbnd = 4
nbndep = 4

nks = 2**3 
nqtot = 2**3

nmodes =  6
nat = 2

#in reality you need to do double transpose for some unknown reasons (example is xqc and epmatq)
dtypes2 = [         '<i',                                                        #nqc
          np.dtype(('<d', (nqtot,3))),                              # xqc
          np.dtype(('<d', (nks,nbnd))),                                          # et_loc
        #   np.dtype(('<c16', (nmodes, nmodes, nqtot))),                           # dynq
          np.dtype(('<c16', (nqtot, nmodes, nmodes))),                           # dynq
          np.dtype(('<c16', (nqtot, nmodes, nks, nbndep, nbndep))), # epmatq
          np.dtype(('<d', (3, 3, nat))),                                         #zstar
          np.dtype(('<d', (3, 3)))                                               #epsi
         ]

#loading the data from .epb file
file_path = path_to_epw+'si.epb1'

file = FortranFile(file_path,'r')
epb_data = file.read_record(*dtypes2)
xqc = epb_data[1]
g_arr = epb_data[4].T
eval = epb_data[2]
dynq= epb_data[3].T

# Get the reciprocal lattice vectors
path_to_scf = path_to_epw+'nscf.out'
atoms = read(path_to_scf)
lattice_constant = atoms.get_cell_lengths_and_angles()[0]
reciprocal_vectors = atoms.cell.reciprocal().T *lattice_constant
# Get the k-points
kpoints =  atoms.calc.kpts
a_factor = 2*np.abs(reciprocal_vectors[0,0])

real_vectors = np.linalg.inv(reciprocal_vectors) /a_factor

## creating the reshufling list to correctrly parse q points
q_ph   = np.loadtxt(path_to_q_points+'/q_ph_2.txt')
q_nscf = np.loadtxt(path_to_q_points+'/q_nscf_2.txt')
iq_ph_list =[]

for i_ph in range(len(q_ph)):
    for i_nscf in range(len(q_nscf)):
        q_nscf_crystal = real_vectors @ q_nscf[i_nscf]
        q_ph_crystal  = real_vectors @ q_ph[i_ph]
        check = [False, False, False]
        delta_q_all = np.abs(q_nscf_crystal-q_ph_crystal)
        for ind_q, delta_q in enumerate(delta_q_all):
            if (np.isclose(delta_q, 0) or np.isclose(delta_q,1)):
                check[ind_q] = True 

        if np.all(check):
            iq_ph_list.append(i_nscf)
            break

#reding the frozen phonon data for electron-phonon matrix elements
## WITH RESHUFLING
## WITH change of i and j
## gathering frozen phonon data in the same format as in the .epb file
g_frozen = np.zeros((nbndep,nbndep,nks,nmodes,nqtot),dtype=complex)
sum = 0
for ik in range(1,nqtot+1):
    for iq_ph,iq_nscf in enumerate(iq_ph_list):
        path_to_file = path_to_frozen + 'epw/braket_list_rotated_{}_{}'.format(ik,iq_nscf+1)
        # print(path_to_file)
        data = np.loadtxt(path_to_file)
        for line in data:
            iat    = int(line[0])
            i_cart = int(line[1])
            im     = 3*(iat-1) + i_cart
            i      = int(line[3])
            j      = int(line[2])
            sum    += 1
            g_frozen[i-1, j-1, ik-1, im-1, iq_ph] = line[4] - 1j * line[5]

#enforcement of acoustic sum rule
# g_frozen[:,:,:,0:3,0] = 0.0

#Now need to read dyn mat and save the in .epb files as well        
dynq_frozen = np.zeros((nmodes,nmodes,nqtot),dtype=complex)

## DELETE AT THE END
# dynq_frozen = dynq

# path_to_phonon= '/scratch/apolyukhin/julia_tests/qe_inputs_small_test/displacements/phonopy_params.yaml'
# phonon = phonopy.load(path_to_phonon)
Rydberg = ase.units.Rydberg
Bohr = ase.units.Bohr

# iq_temp = 0
# print(iq_ph_list[iq_temp])
# print(real_vectors @q_nscf[iq_ph_list[iq_temp]])
# print(real_vectors @q_ph[iq_temp])
# dyn_test = phonon.get_dynamical_matrix_at_q(real_vectors @q_ph[iq_temp])
# print(dyn_test*Rydberg/Bohr)
# eigvals_frozen, eigvecs_frozen = sc.linalg.eig(dyn_test)
# frequencies = np.sort(np.sqrt(np.abs(eigvals_frozen.real)) * np.sign(eigvals_frozen.real))
# print('eigvals_frozen frozen = ', frequencies*3634.873744)
# print(dynq_frozen[:,:,iq_temp])
# dynq_frozen[:,:,0] = dyn_test*Rydberg/Bohr
# print(dynq_frozen[:,:,0])
# dynq_frozen_data_tst = np.loadtxt(path_to_frozen+'dyn_mat/dyn_mat{}'.format(1))#*(Bohr/Rydberg)
# dynq_tst  = dynq_frozen_data_tst[:,0::2] + 1j*dynq_frozen_data_tst[:,1::2]
# print('Frozen \n', dynq_frozen_data_tst)
# print('DFPT \n',dynq[:,:,0])
# print('ratio \n', dynq_tst/dynq[:,:,0])
# exit(3)
# eigvals_frozen, eigvecs_frozen = sc.linalg.eig(dyn_test)
# frequencies = np.sort(np.sqrt(np.abs(eigvals_frozen.real)) * np.sign(eigvals_frozen.real))
# print('eigvals_frozen frozen = ', frequencies)

# DELETE AT THE END

for iq_ph,iq_nscf in enumerate(iq_ph_list):
    #need to  have correct reshuffling of iq points like in ep case + multiple by correct factor 
    dynq_frozen_data = np.loadtxt(path_to_frozen+'dyn_mat/dyn_mat{}'.format(iq_nscf+1))
    dynq_frozen[:,:,iq_ph]  = dynq_frozen_data[:,0::2] + 1j*dynq_frozen_data[:,1::2]
    
    # print(np.real(np.round(dynq[:,:,iq],3)))
    # eigvals_dfpt, eigvecs_dfpt = sc.linalg.eig(dynq[:,:,iq])
    # frequencies = np.sort(np.sqrt(np.abs(eigvals_dfpt.real)) * np.sign(eigvals_dfpt.real))
    
    # Rydberg = ase.units.Rydberg
    # Bohr = ase.units.Bohr
    # print('Rydberg = ', Rydberg)
    # print('Bohr = ', Bohr)
    # print(Bohr/Rydberg*0.364)
    # # print('eigenvals dfpt scales = ', frequencies)
    # print('eigenvals dfpt = ', frequencies)
    
    # eigvals_frozen, eigvecsv = sc.linalg.eig((dynq_frozen[:,:,iq].T))
    # print(np.real(dynq_frozen[:,:,iq]))
    # print('eigenvals frozen = ', eigvals_frozen)
    # exit(3)

#writing the update g array to the binary file .epb
file_path = path_to_epw+'si.epb1'
file = FortranFile(file_path,'w')
epb_data_list = list(epb_data)
epb_data_list[4] = g_frozen.T
epb_data_list[3] = dynq_frozen.T#/0.91546028
new_epb_data = tuple(epb_data_list)

file.write_record(*new_epb_data)
file.close()