# Examples

This example demonstrates the calculation of electron-phonon matrix elements for Silicon on a $\mathbf{k}=\mathbf{q}=2\times2\times2$ grid using the projectability approach with PBE and HSE functionals.  

## Silicon with PBE

The first step is loading the ElePhAny module together with PythonCall, as we are using the ASE and Phonopy libraries to handle the creation of Quantum Espresso inputs and phonon calculations, respectively.

```julia
 using ElectronPhonon, PythonCall
```
Then, we need to decide where to create the folder with the corresponding data for the finite difference (FD) calculation. Consider that we set `path_to_calc` to the same path where the `si.jl` script for the calculation is located. We expect the following structure of the calculation:

```julia
path_to_calc = pwd() * "/"
```

```md
.
├── si.jl
├── run.sh
└── displacements
    ├── dyn_mat
    ├── epw
    ├── group_1
    ...
    ├── group_12
    ├── out
    ├── phonopy_params.yaml
    ├── phonopy.yaml
    ├── qpoints.yaml
    ├── scf_0
    └── sc_size.conf
```

Let's break it down piece by piece. Inside `displacement` folder, there are `group_*` folders that contain input/output of the displaced configurations (`scf.in scf.out nscf.in nscf.out`) that we need for further post-processing. 
`scf_0` folders contain information about pristine calculation together with `*jld2` files that are needed for restarting of the code once the post-processing is partially finished. `phonopy_params.yaml` and `phonopy.yaml qpoints.yaml sc_size.conf` contains information related to Phonopy calculation. 
`out` folder contains information about electron-phonon elements calculated with DFPT and FD on the coarse-grid for a given `ik` and `iq` point per file. `epw` and `dyn_mat` folders are needed for further interface with EPW package. 
If run.sh is also present in the root folder, then subsequent executions of QE are going to be done using Slurm, and `run.sh` is going to be copied in all the `group_*` and `scf_0` folders. 

Now we need to set up additional settings for the FD calculation: `abs_disp` is the size of the finite displacement which we are going to apply to atoms (Bohr). `path_to_qe` is an optional path to the Quantum Espresso (QE) source folder, needed only when we want to compare results of FD with DFPT; otherwise, ElePhAny assumes that QE executables are on the path. To access electron-phonon elements, you will need to compile a special version of the ph.x code located in `testsuite/non_epw_comp` of QE.
The following parameter, `mpi_ranks`, is required for the parallel execution of QE. It is ignored if `run.sh` is present and calculations are run using Slurm.
The `use_symm` flag corresponds to whether symmetries of displaced configurations are going to be used or every atom of the unit cell is going to be perturbed.
At last, `sc_size` corresponds to the supercell size, which corresponds to the $\mathbf{q}$ grid.
`k_mesh` corresponds to the k-points sampling of the *supercells*. To have consistency between pristine unit cells and displaced supercells, k-point sampling of the pristine unit cell is a product of $sc\_size \times k\_mesh$.
 
```julia
#Params
abs_disp = 1e-3
path_to_qe= "/home/poliukhin/Soft/sourse/q-e/"
mpi_ranks = 8
use_symm = false
sc_size::Vec3{Int} = [2,2,2]
k_mesh::Vec3{Int}  = [1,1,1]
```

The final step in setting up the calculation is to create the QE input file for the pristine system. It is handled by ASE and Phonopy packages, for which we need to provide `unitcell` and `scf_parameters` python-like dictionaries. 

```julia
# Lattice constant of Silicon
a = 5.43052   in Angstrom

unitcell = Dict(
    :symbols =>  pylist(["Si", "Si"]),
    :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
    [0.0, 0.5 * a, 0.5 * a],
    [-0.5 * a, 0.5 * a, 0.0]]),
    :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
    :masses => pylist([28.08550, 28.08550])
)

# Set up the calculation parameters as a Python dictionary
scf_parameters = Dict(
    :format => "espresso-in",
    :kpts => pytuple((k_mesh[1]*sc_size[1], k_mesh[2]*sc_size[2], k_mesh[3]*sc_size[3])),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/home/poliukhin/Development/frozen_phonons/elph/example/pseudo",
    :ecutwfc => 60,
    :conv_thr =>1.e-13,
    :pseudopotentials => Dict("Si" => "Si.upf"),
    :diagonalization => "ppcg",#david
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true,
    :nbnd => 4,
    :electron_maxstep => 1000
)
```

Once all the settings are set, we can create the model. If use_symm is set to `true`, the code will analyze and find the minimum amount of displacements that is consistent with Phonopy.

```julia
model = create_model(path_to_calc = path_to_calc,
                      abs_disp = abs_disp,
                      path_to_qe = path_to_qe,
                      mpi_ranks = mpi_ranks,
                      sc_size = sc_size,
                      k_mesh  = k_mesh,
                      unitcell = unitcell,
                      scf_parameters = scf_parameters,
                      use_symm = use_symm);
```

## Running code 

Once all the preparation is complete, all subsequent functions will operate directly with the model object. First, we could create the displaced configurations and run related DFT calculations.

```julia 
create_disp_calc!(model; from_scratch = from_scratch)
run_calculations(model)
```

!!! note
    If the calculations are run on the cluster and run.sh is present in the root directory, the calculations must be run sequentially, as `run_calculations(model)` is designed to execute both SCF and NSCF calculations. In this case, use the following functions, running them one by one (scf pristine and displaced, nscf pristine, nscf displaced):

    ```julia 
    create_disp_calc!(model; from_scratch = from_scratch)
    run_disp_calc(model)
    run_nscf_calc(model)
    run_disp_nscf_calc(model)
    ```     

Once all the calculations are done, we could post-process the results. The `prepare_model` function will unfold pristine wave functions to the supercell and save them in the binary '*jld2' format for further post-processing. It will also call Phonopy to create a diagonal force constant matrix to access phonon data. After this, the `create_electrons` and `create_phonons` functions will create and save data for the final electron-phonon matrix element calculations. To save space and computational time `create_electrons` will save braket matixes beetween displaced and undisplaced wave-functions $U^{\pm}_{jn\mathbf{k}} = \langle \psi^{\pm}_{j} \mid \psi_{n\mathbf{k}} \rangle$ instead of wave functions itself.

```julia 
prepare_model(model)
electrons = create_electrons(model)
phonons = create_phonons(model)
```
!!! note
    If the electrons and phonons objects were already created before, we could take advantage of the loading functions and not recalculate $U^{\pm}_{jn\mathbf{k}}$ matrices.
    ```julia 
    electrons = load_electrons(model)
    phonons = load_phonons(model)
    ``` 

!!! tip
    Since part of this code allows manipulating wave functions that QE saves in binary or HDF5 format, it is helpful to present some examples. Considering that we have QE calculation with `tmp/scf.save/` folder, which contains several wave functions `wfc*.dat` or `wfc*.hdf5`, we could read them directly in Julia.

    ```julia 
    miller, psi_list = parse_wf("./tmp/scf.save/wfc1")
    ```

    QE saves wave function in reciprocal space $\psi_{n\mathbf{k}}(\mathbf{r}) = \sum_{G}\psi_{n\mathbf{k}}(\mathbf{G}) e^{i(\mathbf{G} +\mathbf{k})\mathbf{r}}$, where psi_list is a list of wave functions coefficient for every band with $N_{\mathbf{G}}$ coefficients per band that coresponds to $\psi_{n\mathbf{k}}(\mathbf{G})$ and miller is matrix $3xN_{G}$ that allows to resonstruct the $G$ vector of coresponding plane wave coefficient $\psi_{n\mathbf{k}}(\mathbf{G})$ by using (read more wave functions in QE on [Gitlab Wiki page](https://gitlab.com/QEF/q-e/-/wikis/Developers/Format-of-data-files)):
        
    $$\mathbf{G}[i, j, k] = \text{miller}[1,i] \mathbf{b}_1 + \text{miller}[2,i] \mathbf{b}_2 + \text{miller}[3,i] \mathbf{b}_3$$

    Having access to wave function coefficients and Miller indices, we could easily transform the wave function to real or reciprocal space as well as calculate brackets between them.

    ```julia 
    N_fft = determine_fft_grid("./tmp/scf.save/data-file-schema.xml"; use_xml=true)
    psi_list_R = [wf_from_G(miller, psi_G, N_fft) for psi_G in psi_list]
    psi_list_G = [wf_to_G(miller, psi_R, N_fft) for psi_R in psi_list_R]
    braket_G = calculate_braket(psi_list_G[1],psi_list_G[2])
    braket_R = calculate_braket_real(psi_list_R[2], psi_list_R[2])
    ```

Now, the final part of the calculation is the construction of the electron-phonon matrix for every k and q points of interest. 
`electron_phonon_qe` function will try to calculate electron-phonon matrix elements directly in QE so that comparison between DFPT and FD could be made (provided a special version of ph.x is compiled). `plot_ep_coupling` allows simple visualization of the output quantities. 

```julia 
# Electron-phonon matrix elements
ik_list = [i for i in 1:product(k_mesh*sc_size)]
iq_list = [i for i in 1:product(k_mesh*sc_size)]

for ik in ik_list
    for iq in iq_list
        electron_phonon_qe(model, ik, iq)# requires to compile special ph.x in testsuite/non_epw_comp
        electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
        plot_ep_coupling(model, ik, iq)
    end
end
```

Inspecting the `out` folder, we could find the resulting comparison of electron-phonon matrix elements. For example for `ik` = 2, `iq` = 1:

![Comparison of electron-phonon matrix elements](assets/comparison_2_1.png)

## Silicon with Hybrids

Since the proposed approach can be straightforwardly applied to any functional of interest, we could perform the same calculation, for example, for the Hybrid functional. In QE, we could do it by just adding a few additional parameters and following the rest of the example. 

```julia 
scf_parameters_hse = Dict(
    :input_dft => "HSE",
    :nqx1 => 1,
    :nqx2 => 1,
    :nqx3 => 1
)

merge!(scf_parameters, scf_parameters_hse)
```
In case any other method beyond DFT is of interest, one could intervene after the electrons and phonons object is created and change the corresponding eigenvalues and eigenvectors.

## Interface with EPW
TBD
