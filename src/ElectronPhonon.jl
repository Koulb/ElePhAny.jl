module ElectronPhonon

###
using PyCall

const ase = PyNULL()
const ase_io = PyNULL()
const phonopy = PyNULL()
const phonopy_structure_atoms = PyNULL()

function __init__()
    copy!(ase, pyimport("ase"))
    copy!(ase_io, pyimport("ase.io"))
    copy!(phonopy, pyimport("phonopy"))
    copy!(phonopy_structure_atoms, pyimport("phonopy.structure.atoms"))
end

# Write your package code here.
export create_disp_calc, run_disp_calc, save_potential
include("electrons.jl")

export prepare_wave_functions_all
include("wave_function.jl")

export calculate_phonons
include("phonons.jl")

export electron_phonon, electron_phonon_qe
include("electron_phonons.jl")

end
