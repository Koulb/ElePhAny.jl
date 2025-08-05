# Examples of the code usage (TODO: convert to documentation):

### Si/dft/si.jl 
The default usage of the code creates a "displacement" folder containing all the data for the EP calculation. Subsequent steps are: 
- Create displaced supercells (with or without symmetries) 
- Run scf calculations (manually or with a code)
- Unfold unperturbed wave functions to supercell for the latter calculation of brakets
- Calculation of ep matrix elements using projectability formula
- Save data/compare results with DFPT

### Si/dft/si_file.jl
Reading input parameters from pristine unit cell scf.in file and frozen_params.json

### Si/dft_epw/si.jl
Example of the inerface of the ElePhAny with EPW code 

### Si/hse/si_hse.jl
It is the same as DFT, but it uses HSE functionally. Note that whenever 'run.sh' is present in the folder, run_calculation will try to use it to launch calculation with slurm (a simple solution for running on the cluster for now).

### Si/kcw/si_kcw.jl
(This is an old script, needs to be updated). Same as dft, but uses koopmans functionals. The main difference is that it also requires koopmans.json and koopmans_sc.json files for now (TODO: ideally, we want to have some interaction between the code, so we don't need to provide this explicitly), where the first file is unit cell calculation. The latter corresponds to a pristine supercell. Additionally, one could choose the spin channel for debugging purposes (the "dw" channel contains dft eigenvalues).

### gaas/dft/gaas.jl
Same as si/dft but for GaAs # Examples of the code usage (TODO: convert to documentation):







