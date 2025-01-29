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


export uma_to_ry, cm1_to_ry ,cm1_to_Thz
include("const.jl")

export Electrons, Phonons, create_model, create_model_kcw
include("model.jl")

include("symmetries.jl")

export create_disp_calc, create_disp_calc!, run_disp_calc, save_potential, run_nscf_calc, prepare_eigenvalues, create_electrons
export load_electrons
include("electrons.jl")

export prepare_wave_functions_all, prepare_wave_functions_undisp, prepare_u_matrixes, create_phonons, create_miller_index
include("wave_function.jl")

export calculate_phonons, prepare_phonons, load_phonons
include("phonons.jl")

export run_calculations, prepare_model, electron_phonon, electron_phonon_qe, plot_ep_coupling, get_kpoint_list, fold_kpoint
include("electron_phonons.jl")

export parse_qe_in, parse_frozen_params
include("io.jl")

#cli
include("cli/main.jl")

end
