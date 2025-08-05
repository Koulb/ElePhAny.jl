abstract type AbstractModel end
abstract type AbstractElectrons end
abstract type AbstractPhonons end
abstract type AbstractSymmetries end

"""
    Symmetries

A structure representing the symmetries of a system.

# Fields
- `ineq_atoms_list::Vector{Int64}`: List of indices of inequivalent atoms in the system.
- `trans_list::Vector{Vector{Float64}}`: List of translation vectors associated with each symmetry operation.
- `rot_list::Vector{Matrix{Float64}}`: List of rotation matrices corresponding to each symmetry operation.
"""
struct Symmetries <: AbstractSymmetries
    ineq_atoms_list::Vector{Int64}
    trans_list::Vector{Vector{Float64}}
    rot_list::Vector{Matrix{Float64}}
    ind_k_list::Vector{Vector{Int}}
end

"""
    ModelQE <: AbstractModel

A mutable struct representing a Quantum ESPRESSO (QE) model configuration.

# Fields
- `path_to_calc::String`: Path to the calculation directory.
- `abs_disp::Float64`: Absolute displacement value used in calculations.
- `path_to_qe::String`: Path to the Quantum ESPRESSO executable.
- `mpi_ranks::Int`: Number of MPI ranks to use for parallel execution.
- `sc_size::Int`: Supercell size.
- `k_mesh::Int`: Number of k-points in the mesh.
- `Ndispalce::Int`: Number of displacements.
- `unitcell::Dict`: Dictionary containing unit cell information.
- `scf_parameters::Dict`: Dictionary of self-consistent field (SCF) parameters.
- `use_symm::Bool`: Whether to use symmetry in calculations.
- `symmetries::Symmetries`: Symmetry information associated with the model.
"""
mutable struct ModelQE <: AbstractModel
    path_to_calc::String
    abs_disp::Float64
    path_to_qe::String
    mpi_ranks::Int
    sc_size::Vector{Int}
    k_mesh::Vector{Int}
    Ndispalce::Int
    unitcell::Dict
    scf_parameters::Dict
    use_symm::Bool
    symmetries::Symmetries
end

"""
    ModelKCW <: AbstractModel


A mutable struct representing a Koopmans model configuration.

# Fields
- `path_to_calc::String`: Path to the calculation directory.
- `spin_channel::String`: Spin channel used in the calculation (e.g., "up", "down").
- `abs_disp::Float64`: Absolute displacement value for phonon calculations.
- `path_to_qe::String`: Path to the Quantum ESPRESSO executable.
- `mpi_ranks::Int`: Number of MPI ranks to use for parallel calculations.
- `sc_size::Int`: Supercell size.
- `k_mesh::Int`: Number of k-points in the mesh.
- `Ndispalce::Int`: Number of displacements.
- `unitcell::Dict`: Dictionary containing unit cell information.
- `scf_parameters::Dict`: Dictionary of self-consistent field (SCF) parameters.
- `use_symm::Bool`: Whether to use symmetry in calculations.
- `symmetries::Symmetries`: Symmetry operations associated with the model.
"""
mutable struct ModelKCW <: AbstractModel
    path_to_calc::String
    spin_channel::String
    abs_disp::Float64
    path_to_qe::String
    mpi_ranks::Int
    sc_size::Vector{Int}
    k_mesh::Vector{Int}
    Ndispalce::Int
    unitcell::Dict
    scf_parameters::Dict
    use_symm::Bool
    symmetries::Symmetries
end

"""
    create_model(; path_to_calc::String = "./",
                   abs_disp::Float64 = 1e-3,
                   path_to_qe::String = "./",
                   mpi_ranks::Int = 1,
                   sc_size::Int = 2,
                   k_mesh::Int = 1,
                   Ndispalce::Int = 0,
                   unitcell::Dict = Dict(),
                   scf_parameters::Dict = Dict(),
                   use_symm::Bool = false,
                   symmetries::Symmetries = Symmetries([],[],[]))

Create and initialize a `ModelQE` object with the specified parameters.

# Keyword Arguments
- `path_to_calc::String = "./"`: Path to the calculation directory.
- `abs_disp::Float64 = 1e-3`: Absolute displacement value for atomic positions.
- `path_to_qe::String = "./"`: Path to the Quantum ESPRESSO executable.
- `mpi_ranks::Int = 1`: Number of MPI ranks to use for calculations.
- `sc_size::Int = 2`: Size of the supercell.
- `k_mesh::Int = 1`: Number of k-points in the mesh.
- `Ndispalce::Int = 0`: Number of displacements to consider.
- `unitcell::Dict = Dict()`: Dictionary containing unit cell information.
- `scf_parameters::Dict = Dict()`: Dictionary of self-consistent field (SCF) parameters.
- `use_symm::Bool = false`: Whether to use symmetries in the calculation.
- `symmetries::Symmetries = Symmetries([],[],[])`: Symmetry operations to use.

# Returns
- `ModelQE`: An initialized `ModelQE` object with the specified settings.

"""
function create_model(;path_to_calc::String = "./",
                      abs_disp::Float64    = 1e-3,
                      path_to_qe::String   = "./",
                      mpi_ranks::Int       = 1,
                      sc_size::Vector{Int}   = [2, 2, 2],
                      k_mesh::Vector{Int}    = [1, 1, 1],
                      Ndispalce::Int       = 0,
                      unitcell::Dict       = Dict(),
                      scf_parameters::Dict = Dict(),
                      use_symm::Bool       = false,
                      symmetries::Symmetries = Symmetries([],[],[],[]))

     if use_symm && any(x -> x > 1, k_mesh)
         @error "Symmetry usage is not implemented for EP supercell calculations with kpoints"
     end

    return ModelQE(path_to_calc, abs_disp, path_to_qe, mpi_ranks, sc_size, k_mesh, Ndispalce, unitcell, scf_parameters, use_symm, symmetries)
end

"""
    create_model_kcw(path_to_calc::String,
                     spin_channel::String,
                     abs_disp::Float64,
                     path_to_qe::String,
                     mpi_ranks::Int,
                     sc_size::Int,
                     k_mesh::Int,
                     Ndispalce::Int,
                     unitcell::Dict,
                     scf_parameters::Dict,
                     use_symm::Bool)

Create and return a `ModelKCW` object with the specified parameters.

# Arguments
- `path_to_calc::String`: Path to the calculation directory.
- `spin_channel::String`: Spin channel identifier (e.g., "up", "down").
- `abs_disp::Float64`: Absolute displacement value.
- `path_to_qe::String`: Path to the Quantum ESPRESSO executable.
- `mpi_ranks::Int`: Number of MPI ranks to use.
- `sc_size::Int`: Supercell size.
- `k_mesh::Int`: Number of k-points in the mesh.
- `Ndispalce::Int`: Number of displacements.
- `unitcell::Dict`: Dictionary containing unit cell information.
- `scf_parameters::Dict`: Dictionary of self-consistent field (SCF) parameters.
- `use_symm::Bool`: Whether to use symmetry in calculations.

# Returns
- `ModelKCW`: An instance of the `ModelKCW` type initialized with the provided parameters.
"""
function create_model_kcw(path_to_calc::String,
                          spin_channel::String,
                          abs_disp::Float64,
                          path_to_qe::String,
                          mpi_ranks::Int,
                          sc_size::Vector{Int},
                          k_mesh::Vector{Int},
                          Ndispalce::Int,
                          unitcell::Dict,
                          scf_parameters::Dict,
                          use_symm::Bool)

    return ModelKCW(path_to_calc, spin_channel, abs_disp, path_to_qe, mpi_ranks, sc_size, k_mesh, Ndispalce, unitcell, scf_parameters, use_symm, symmetries)
end

"""
    create_model_kcw(; path_to_calc::String = "./",
                      spin_channel::String = "up",
                      abs_disp::Float64 = 1e-3,
                      path_to_qe::String = "./",
                      mpi_ranks::Int = 1,
                      sc_size::Int = 2,
                      k_mesh::Int = 1,
                      Ndispalce::Int = 0,
                      unitcell::Dict = Dict(),
                      scf_parameters::Dict = Dict(),
                      use_symm::Bool = false,
                      symmetries::Symmetries = Symmetries([],[],[]))

Create and initialize a `ModelKCW` object with the specified parameters.

# Keyword Arguments
- `path_to_calc::String`: Path to the calculation directory. Defaults to `"./"`.
- `spin_channel::String`: Spin channel to use (e.g., `"up"` or `"down"`). Defaults to `"up"`.
- `abs_disp::Float64`: Absolute displacement value. Defaults to `1e-3`.
- `path_to_qe::String`: Path to Quantum ESPRESSO executables. Defaults to `"./"`.
- `mpi_ranks::Int`: Number of MPI ranks to use. Defaults to `1`.
- `sc_size::Int`: Supercell size. Defaults to `2`.
- `k_mesh::Int`: Number of k-points in the mesh. Defaults to `1`.
- `Ndispalce::Int`: Number of displacements. Defaults to `0`.
- `unitcell::Dict`: Dictionary describing the unit cell. Defaults to an empty `Dict()`.
- `scf_parameters::Dict`: Dictionary of self-consistent field (SCF) parameters. Defaults to an empty `Dict()`.
- `use_symm::Bool`: Whether to use symmetries. Defaults to `false`.
- `symmetries::Symmetries`: Symmetries object. Defaults to `Symmetries([],[],[])`.

# Returns
- `ModelKCW`: An initialized `ModelKCW` object.
"""
function create_model_kcw(;path_to_calc::String = "./",
    spin_channel::String = "up",
    abs_disp::Float64    = 1e-3,
    path_to_qe::String   = "./",
    mpi_ranks::Int       = 1,
    sc_size::Vector{Int}   = [2, 2, 2],
    k_mesh::Vector{Int}    = [1, 1, 1],
    Ndispalce::Int       = 0,
    unitcell::Dict       = Dict(),
    scf_parameters::Dict = Dict(),
    use_symm::Bool       = false,
    symmetries::Symmetries = Symmetries([],[],[]))

    if use_symm && k_mesh > 1
        @error "Symmetry usage is not implemented for supercell calculations with kpoints"
    end

    return ModelKCW(path_to_calc, spin_channel, abs_disp, path_to_qe, mpi_ranks, sc_size, k_mesh, Ndispalce, unitcell, scf_parameters, use_symm, symmetries)
end

"""
    Electrons

A structure representing electronic properties in a model.

# Fields
- `U_list::Array`: List of brakets beetwen displaced(+tau) and undisplaced wavefunctions.
- `V_list::Array`: List of brakets beetwen displaced(-tau) and undisplaced wavefunctions.
- `ϵkᵤ_list::Array`: List of undisplaced electronic energies.
- `ϵₚ_list::Array`: List of displaced(+tau) electronic energies.
- `ϵₚₘ_list::Array`: List of displaced(+tau) electronic energies.
- `k_list::Array`: List of k-points.
"""
struct Electrons <: AbstractElectrons
    U_list::Array{}
    V_list::Array{}
    ϵkᵤ_list::Array{}
    ϵₚ_list::Array{}#Float64
    ϵₚₘ_list::Array{}
    k_list::Array{}
end

"""
    Phonons

A structure representing phonon properties.

# Fields
- `M_phonon::Array{}`: Array transfromation from crystall to cartesian basis.
- `ωₐᵣᵣ_ₗᵢₛₜ::Array{}`: Array of phonon frequencies.
- `εₐᵣᵣ_ₗᵢₛₜ::Array{}`: Array of phonon energies or eigenvalues.
- `mₐᵣᵣ::Array{}`: Array of masses (typically `Float64`).

# Notes
All fields are currently typed as `Array{}`. For improved type safety and performance, consider specifying element types, e.g., `Array{Float64}`.
"""
struct Phonons <: AbstractPhonons
    M_phonon::Array{}
    ωₐᵣᵣ_ₗᵢₛₜ::Array{}
    εₐᵣᵣ_ₗᵢₛₜ::Array{}
    mₐᵣᵣ::Array{}#Float64
end
