module ElectronPhonon

###
using PythonCall

const ase = PythonCall.pynew()
const ase_io = PythonCall.pynew()
const phonopy = PythonCall.pynew()
const phonopy_structure_atoms =  PythonCall.pynew()
const np =  PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(ase, pyimport("ase"))
    PythonCall.pycopy!(ase_io, pyimport("ase.io"))
    PythonCall.pycopy!(phonopy, pyimport("phonopy"))
    PythonCall.pycopy!(phonopy_structure_atoms, pyimport("phonopy.structure.atoms"))
    PythonCall.pycopy!(np, pyimport("numpy"))
end

# Write your package code here.
export create_disp_calc, run_disp_calc, save_potential
include("electrons.jl")

export prepare_wave_functions_all
include("wave_function.jl")

export calculate_phonons
include("phonons.jl")

export electron_phonon, electron_phonon_qe, plot_ep_coupling
include("electron_phonons.jl")

end
