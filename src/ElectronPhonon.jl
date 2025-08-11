module ElectronPhonon

###
using PythonCall
using MPI  # Add MPI support

const ase = PythonCall.pynew()
const ase_io = PythonCall.pynew()
const phonopy = PythonCall.pynew()
const phonopy_structure_atoms =  PythonCall.pynew()
const np =  PythonCall.pynew()

# Global MPI variables
const MPI_COMM = Ref{MPI.Comm}()
const MPI_RANK = Ref{Int}()
const MPI_SIZE = Ref{Int}()

function __init__()
    # Initialize MPI
    if !MPI.Initialized()
        MPI.Init()
    end
    MPI_COMM[] = MPI.COMM_WORLD
    MPI_RANK[] = MPI.Comm_rank(MPI_COMM[])
    MPI_SIZE[] = MPI.Comm_size(MPI_COMM[])
    
    # Initialize Python modules
    PythonCall.pycopy!(ase, pyimport("ase"))
    PythonCall.pycopy!(ase_io, pyimport("ase.io"))
    PythonCall.pycopy!(phonopy, pyimport("phonopy"))
    PythonCall.pycopy!(phonopy_structure_atoms, pyimport("phonopy.structure.atoms"))
    PythonCall.pycopy!(np, pyimport("numpy"))
    
    # Configure threading for optimal performance
    if MPI_RANK[] == 0
        @info "Initialized with $(Threads.nthreads()) threads and $(MPI_SIZE[]) MPI processes"
    end
end

# MPI utility functions
function mpi_barrier()
    MPI.Barrier(MPI_COMM[])
end

function mpi_bcast!(data, root=0)
    MPI.Bcast!(data, root, MPI_COMM[])
end

function mpi_gather!(sendbuf, recvbuf, root=0)
    MPI.Gather!(sendbuf, recvbuf, root, MPI_COMM[])
end

function mpi_reduce!(sendbuf, recvbuf, op=MPI.SUM, root=0)
    MPI.Reduce!(sendbuf, recvbuf, op, root, MPI_COMM[])
end

function mpi_scatter!(sendbuf, recvbuf, root=0)
    MPI.Scatter!(sendbuf, recvbuf, root, MPI_COMM[])
end

# Get MPI rank and size
mpi_rank() = MPI_RANK[]
mpi_size() = MPI_SIZE[]
is_master() = MPI_RANK[] == 0

export uma_to_ry, cm1_to_ry ,cm1_to_Thz, bohr_to_ang
export Vec3, Mat3
include("const.jl")

export Electrons, Phonons, create_model, create_model_kcw
include("model.jl")

include("symmetries.jl")

export create_disp_calc, create_disp_calc!, run_disp_calc, save_potential, run_nscf_calc, prepare_eigenvalues, create_electrons
export load_electrons, run_disp_nscf_calc, run_nscf_calc, run_disp_calc
include("electrons.jl")

export prepare_wave_functions_all, prepare_wave_functions_undisp, prepare_u_matrixes, create_phonons, create_miller_index
export parse_wf, determine_fft_grid, wf_from_G, wf_to_G, calculate_braket, calculate_braket_real
export wf_from_G_optimized, wf_from_G_list_optimized, determine_phase_optimized, calculate_braket_optimized, calculate_braket_real_optimized
export prepare_wave_functions_to_R_optimized
include("wave_function.jl")

export calculate_phonons, prepare_phonons, load_phonons
include("phonons.jl")

export run_calculations, prepare_model, electron_phonon, electron_phonon_qe, plot_ep_coupling, get_kpoint_list, fold_kpoint
export electron_phonon_mpi, prepare_wave_functions_disp_mpi, run_disp_calc_mpi, prepare_wave_functions_hybrid
export mpi_rank, mpi_size, is_master, mpi_barrier, mpi_bcast!, mpi_gather!, mpi_reduce!, mpi_scatter!
export clear_fft_buffers
include("electron_phonons.jl")

export parse_qe_in, parse_frozen_params
include("io.jl")

#cli
include("cli/main.jl")

end
