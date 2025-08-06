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
