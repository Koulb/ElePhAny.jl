using Test
using ElectronPhonon
using PythonCall

import Base: isapprox

function isapprox(a::Union{Py, PyArray}, b::Union{Py, PyArray}; atol=ElectronPhonon.toleranse_tests)
    a_jl = vec(pyconvert(Vector,a))
    b_jl = vec(pyconvert(Vector,b))

    if length(a_jl) != length(b_jl)
        return false
    end

    result = all(isapprox.(a_jl, b_jl; atol=atol))

    return result
end

@testset "Reading the scf.in file to create inputs for the model" begin
    path_to_scf = "test_data/displacements/scf_0/scf.in"
    unitcell, scf_parameters = ElectronPhonon.parse_qe_in(path_to_scf)

    a = 5.43052  # in Angstrom
    mesh = [2,2,2]

    #Defined slighly different so Python objects could be compared
    unitcell_check = Dict(
        :symbols =>  pylist(["Si", "Si"]),
        :cell =>    pylist([pylist([-0.5 * a, 0.0, 0.5 * a]),
                           pylist([0.0, 0.5 * a, 0.5 * a]),
                           pylist([-0.5 * a, 0.5 * a, 0.0])]),
        :scaled_positions => pylist([[0, 0, 0],[0.75, 0.75, 0.75]]),
        :masses => pylist([28.08550, 28.08550])
    )

    # Set up the calculation parameters as a Python dictionary
    scf_parameters_check = Dict(
        :format => "espresso-in",
        :kpts => pytuple((mesh[1], mesh[2], mesh[3])),
        :calculation =>"scf",
        :prefix => "scf",
        :outdir => "./tmp/",
        :pseudo_dir => "./tmp/scf.save/",
        :ecutwfc => 60,
        :conv_thr =>1.e-13,# 1e-16,# #1.e-20,#5.e-30
        :pseudopotentials => Dict("Si" => "Si.upf"),
        :mixing_mode => "plain",
        :mixing_beta => 0.7,
        :crystal_coordinates => true,
        :verbosity => "high",
        :tstress => false,
        :ibrav => 2,
        :tprnfor => true,
        :nbnd => 4,
        :electron_maxstep => 1000,
        :nat => 2,
        :nosym=> false,
        :noinv=> false
    )


    @test all(pyconvert(Any, unitcell[:symbols] == unitcell_check[:symbols]))
    @test all(pyconvert(Any, unitcell[:masses] == unitcell[:masses]))

    for icell in 0:2
        @test isapprox(unitcell[:cell][icell],unitcell_check[:cell][icell])
    end

    @test isapprox(unitcell[:scaled_positions],unitcell_check[:scaled_positions])

    for iat in 0:1
        @test isapprox(unitcell[:scaled_positions][iat],unitcell_check[:scaled_positions][iat])
    end

    for key in keys(scf_parameters)
        if key == :kpts
            result = pyconvert(Any,scf_parameters[key] == scf_parameters_check[key])
            @test all(result)
        else
            result = scf_parameters[key] == scf_parameters_check[key]
            @test all(result)
        end
    end
end



@testset "Reading frozen_params.json to check finite-difference parameters" begin
    path_to_json = "test_data/frozen_params.json"

    params = ElectronPhonon.parse_frozen_params(path_to_json)

    params_check = Dict(
                       "abs_disp" => 1e-3 ,
                       "path_to_qe"=>"/home/user/soft/qe",
                       "mpi_ranks" => 8,
                       "sc_size" => [2,2,2]
                       )

    for key in keys(params_check)
        result = params[key] == params_check[key]
        @test all(result)
    end

end


@testset "Creating of the model instance from scf.in and frozen_params.json" begin
    path_to_scf = "test_data/displacements/scf_0/scf.in"
    unitcell, scf_parameters = ElectronPhonon.parse_qe_in(path_to_scf)

    path_to_json = "test_data/frozen_params.json"
    frozen_params = ElectronPhonon.parse_frozen_params(path_to_json)

    model = create_model(path_to_calc = frozen_params["path_to_calc"],
                         abs_disp = frozen_params["abs_disp"],
                         path_to_qe = frozen_params["path_to_qe"],
                         mpi_ranks = frozen_params["mpi_ranks"],
                         sc_size  = frozen_params["sc_size"],
                         k_mesh  = frozen_params["k_mesh"],
                         Ndispalce = frozen_params["Ndispalce"],
                         unitcell = unitcell,
                         scf_parameters = scf_parameters)

    #Test that model instance is propely created
    @test model.abs_disp == 1e-3
end


function restore_epb!(path_to_epw::String, path_to_saved_epb::String, epb_name::String)
    src = joinpath(path_to_saved_epb, epb_name)
    dest = joinpath(path_to_epw, epb_name)

    cp(src, dest; force=true)
end


function arrays_close(A, B; rtol=1e-8, atol=1e-12)
    size(A) == size(B) || return false
    return all(isapprox.(A, B; rtol=rtol, atol=atol))
end


@testset "Parsing .epb file" begin
    path_to_epw = "./test_data/"
    path_to_frozen = "./test_data/displacements"
    path_to_saved_epb = "./test_data/save_dir"

    epb_name = "si.epb1"
    nat = 2
    mesh = 8
    mesh_q = 8
    nmodes = 3 * nat
    nbnd = 4
    nbndep = 4

    println("testing only_g = false:-------------------------------")
    only_g = false
    parse_epb(
    path_to_epw, 
    path_to_frozen, 
    nbnd,  
    nbndep,
    mesh, 
    mesh,
    nat, 
    epb_name, 
    only_g
    )

    epb_data = read_fortran_binary_generic(joinpath(path_to_epw, epb_name), mesh_q, mesh, nbnd, nmodes, nbndep, nat)
    _, xqc, et_loc, dynq, gcoupling, _, _ = epb_data


    xqc_expected = [ 0.5, -0.5,  0.5] #1
    et_loc_expected = [-0.0974736 , -0.0974736 ,  0.26530639,  0.26530639]#5
    dynq_expected = -0.030605025398691723 #[1,3,4]
    gcoupling_expected = -0.21944698342+0.12497402507im #[1,5,1,2,3]

    @test size(xqc) == (mesh_q,3)
    @test size(et_loc) == (mesh,nbnd)
    @test size(dynq) == (mesh_q,nmodes,nmodes)
    @test size(gcoupling) == (mesh_q,nmodes,mesh,nbndep,nbndep) 

    @test isapprox(xqc[2,3], xqc_expected[3]; rtol=1e-10, atol=1e-12)
    @test isapprox(et_loc[6,2], et_loc_expected[2]; rtol=1e-6, atol=1e-12)
    @test isapprox(dynq[2,4,5], dynq_expected; rtol=1e-10, atol=1e-12)
    @test isapprox(real(gcoupling[2,6,2,3,4]), real(gcoupling_expected),rtol=1e-10, atol=1e-12)

    restore_epb!(path_to_epw, path_to_saved_epb, epb_name)
    
    println("testing now only_g = true:---------------------------")
    only_g = true
    parse_epb(
    path_to_epw, 
    path_to_frozen, 
    nbnd,  
    nbndep,
    mesh, 
    mesh,
    nat, 
    epb_name, 
    only_g
    )

    altered = read_fortran_binary_generic(joinpath(path_to_epw, epb_name),mesh_q, mesh, nbnd, nmodes, nbndep, nat)
    reference = read_fortran_binary_generic(joinpath(path_to_saved_epb, epb_name),mesh_q, mesh, nbnd, nmodes, nbndep, nat)
    _, xqc_a, et_loc_a, dynq_a, _, _, _ = altered
    _, xqc_r, et_loc_r, dynq_r, _, _, _ = reference

    @test arrays_close(xqc_a, xqc_r)
    @test arrays_close(et_loc_a, et_loc_r)
    @test size(dynq_a) == (mesh_q,nmodes,nmodes)
    @test size(dynq_r) == (mesh_q,nmodes,nmodes)

    restore_epb!(path_to_epw, path_to_saved_epb, epb_name)

    println("finished tests of epb function-----------------------")


end

@testset "Parsing .xml files" begin 
    path_to_data = "./test_data/"
    path_to_kcw = "displacements/scf_0/tmp/scf.save"
    path_to_save = "save_dir"
    path_to_out = "si.save"
    mesh = 8
    nbnd = 4
    type_of_calc = "dft"

    fake2nscf(
        path_to_data,
        path_to_kcw,
        path_to_save,
        path_to_out,
        mesh,
        type_of_calc
    )
    
    et_loc = zeros(Float64, mesh, nbnd)
    energy_dicts = read_kpoint_eigenvals(joinpath(path_to_data, path_to_out, "data-file-schema.xml"),type_of_calc = type_of_calc)
    et_loc_expected = [-0.0974736 , -0.0974736 ,  0.26530639,  0.26530639]
    for ik in 1:mesh
        e_values = energy_dicts[ik]["energies"]
        et_loc[ik, :] = 2 .* e_values  # Ha to Ry
    end

    @test isapprox(et_loc[6,2], et_loc_expected[2]; rtol=1e-6, atol=1e-12)
    @test isapprox(et_loc[6,3], et_loc_expected[3]; rtol=1e-6, atol=1e-12)
    @test isapprox(et_loc[6,4], et_loc_expected[4]; rtol=1e-6, atol=1e-12)
end
