abstract type AbstractElectrons end
abstract type AbstractPhonons end
abstract type AbstractModel end


struct ModelQE <: AbstractModel
    path_to_calc::String
    abs_disp::Float64
    directory_path::String
    path_to_qe::String
    mpi_ranks::Int
    mesh::Int
    Ndispalce::Int
    unitcell::Dict
    scf_parameters::Dict
end

function create_model(path_to_calc::String, abs_disp::Float64, directory_path::String, path_to_qe::String, mpi_ranks::Int, mesh::Int, Ndispalce::Int, unitcell::Dict, scf_parameters::Dict)
    return ModelQE(path_to_calc, abs_disp, directory_path, path_to_qe, mpi_ranks, mesh, Ndispalce, unitcell, scf_parameters)
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


