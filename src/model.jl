abstract type AbstractModel end
abstract type AbstractElectrons end
abstract type AbstractPhonons end
abstract type AbstractSymmetries end

struct Symmetries <: AbstractSymmetries
    ineq_atoms_list::Vector{Int64}
    trans_list::Vector{Vector{Float64}}
    rot_list::Vector{Matrix{Float64}}
end

mutable struct ModelQE <: AbstractModel
    path_to_calc::String
    abs_disp::Float64
    path_to_qe::String
    mpi_ranks::Int
    sc_size::Vec3{Int}
    k_mesh::Vec3{Int}
    Ndispalce::Int
    unitcell::Dict
    scf_parameters::Dict
    use_symm::Bool
    symmetries::Symmetries
end

mutable struct ModelKCW <: AbstractModel
    path_to_calc::String
    spin_channel::String
    abs_disp::Float64
    path_to_qe::String
    mpi_ranks::Int
    sc_size::Vec3{Int}
    k_mesh::Vec3{Int}
    Ndispalce::Int
    unitcell::Dict
    scf_parameters::Dict
    use_symm::Bool
    symmetries::Symmetries
end

function create_model(;path_to_calc::String = "./",
                      abs_disp::Float64    = 1e-3,
                      path_to_qe::String   = "./",
                      mpi_ranks::Int       = 1,
                      sc_size::Vec3{Int}   = [2, 2, 2],
                      k_mesh::Vec3{Int}    = [1, 1, 1],
                      Ndispalce::Int       = 0,
                      unitcell::Dict       = Dict(),
                      scf_parameters::Dict = Dict(),
                      use_symm::Bool       = false,
                      symmetries::Symmetries = Symmetries([],[],[]))

     if use_symm && any(x -> x > 1, k_mesh)
         @error "Symmetry usage is not implemented for EP supercell calculations with kpoints"
     end

    return ModelQE(path_to_calc, abs_disp, path_to_qe, mpi_ranks, sc_size, k_mesh, Ndispalce, unitcell, scf_parameters, use_symm, symmetries)
end

function create_model_kcw(path_to_calc::String,
                          spin_channel::String,
                          abs_disp::Float64,
                          path_to_qe::String,
                          mpi_ranks::Int,
                          sc_size::Vec3{Int},
                          k_mesh::Vec3{Int},
                          Ndispalce::Int,
                          unitcell::Dict,
                          scf_parameters::Dict,
                          use_symm::Bool)

    return ModelKCW(path_to_calc, spin_channel, abs_disp, path_to_qe, mpi_ranks, sc_size, k_mesh, Ndispalce, unitcell, scf_parameters, use_symm, symmetries)
end

function create_model_kcw(;path_to_calc::String = "./",
    spin_channel::String = "up",
    abs_disp::Float64    = 1e-3,
    path_to_qe::String   = "./",
    mpi_ranks::Int       = 1,
    sc_size::Vec3{Int}   = [2, 2, 2],
    k_mesh::Vec3{Int}    = [1, 1, 1],
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

struct Electrons <: AbstractElectrons
    U_list::Array{}
    V_list::Array{}
    ϵkᵤ_list::Array{}
    ϵₚ_list::Array{}#Float64
    ϵₚₘ_list::Array{}
    k_list::Array{}
end

struct Phonons <: AbstractPhonons
    M_phonon::Array{}
    ωₐᵣᵣ_ₗᵢₛₜ::Array{}
    εₐᵣᵣ_ₗᵢₛₜ::Array{}
    mₐᵣᵣ::Array{}#Float64
end
