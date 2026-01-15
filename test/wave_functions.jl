using Test, JLD2, Glob, Logging, ElectronPhonon

@testset "Test brakets calcuations" begin
    ψ1 =  Array{Complex{Float64}}([1.0 + 1im, 0.0 + 1im, 0.0 + 1im, 0.0 + 1im])
    ψ2 =  Array{Complex{Float64}}([0.0 + 1im, 1.0 + 1im, 0.0 + 1im, 0.0 + 1im])
    norm11 = abs(ElectronPhonon.calculate_braket(ψ1, ψ1))
    norm12 = abs(ElectronPhonon.calculate_braket(ψ1, ψ2))

    ψ1_real = Array{Complex{Float64}, 3}(undef, 2, 2, 2)
    ψ1_real[:, :, 1] =  sqrt(2) * [1/2 + 1/2im, -1/2 - 1/2im,
                                  -1/2 - 1/2im,  1/2 + 1/2im]
    ψ1_real[:, :, 2] =  sqrt(2) * [1/2 - 1/2im, -1/2 + 1/2im,
                                  -1/2 + 1/2im,  1/2 - 1/2im]

    ψ2_real = Array{Complex{Float64}, 3}(undef, 2, 2, 2)
    ψ2_real[:, :, 1] =  sqrt(2) * [1/2 - 1/2im, -1/2 + 1/2im,
                                  -1/2 + 1/2im,  1/2 - 1/2im]
    ψ2_real[:, :, 2] = sqrt(2) * [1/2 + 1/2im, -1/2 - 1/2im,
                                 -1/2 - 1/2im,  1/2 + 1/2im]

    norm11 = abs(ElectronPhonon.calculate_braket_real(ψ1_real, ψ1_real))
    norm12 = abs(ElectronPhonon.calculate_braket_real(ψ1_real, ψ2_real))

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test parsing wavefunctions from binaries of QE" begin
    path_tst_data = "test_data/displacements/scf_0/tmp/scf.save/wfc1"

    _, evc_list = ElectronPhonon.parse_wf(path_tst_data)
    norm11 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[2]))

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test parsing wavefunctions from HDF5 of QE" begin
    path_tst_data = "test_data/displacements/scf_0/tmp/scf.save/hdf5/wfc1"

    _, evc_list = ElectronPhonon.parse_wf(path_tst_data)
    norm11 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[2]))

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test transforming wavefunction to real space" begin
    path_tst_data = "test_data/displacements/scf_0/tmp/scf.save/wfc1"
    Nxyz = [36, 36, 36]

    miller, evc_list = ElectronPhonon.parse_wf(path_tst_data)
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]

    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[2]))

    atol = ElectronPhonon.toleranse_tests
    @test norm11 > atol # not enforce brakets in real space to normolize to 1 anymore
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test transforming wavefunction to real space with slow fft" begin
    path_tst_data = "test_data/displacements/scf_0/tmp/scf.save/wfc1"
    miller, evc_list = ElectronPhonon.parse_wf(path_tst_data)
    Nxyz = [20, 20, 20]

    wfc_list = [ElectronPhonon.wf_from_G_slow(miller, evc, Nxyz) for evc in evc_list]

    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[2]))

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test transforming wavefunction back to reciprocal space" begin
    path_tst_data = "test_data/displacements/scf_0/tmp/scf.save/wfc1"
    Nxyz = [36, 36, 36]

    miller, evc_list = ElectronPhonon.parse_wf(path_tst_data)
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    evc_list_new = [ElectronPhonon.wf_to_G(miller, wfc, Nxyz) for wfc in wfc_list]

    norm11 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list_new[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list_new[2]))

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test unfolding wavefunctions in supercell" begin
    path_tst_data = "test_data/displacements/scf_0/tmp/scf.save/wfc1"
    Nxyz = [36, 36, 36]
    sc_size = [2, 2, 2]

    miller, evc_list = ElectronPhonon.parse_wf(path_tst_data)
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    wfc_list_sc = [ElectronPhonon.wf_pc_to_sc(wfc, sc_size) for wfc in wfc_list]

    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list_sc[1], wfc_list_sc[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list_sc[1], wfc_list_sc[2]))

    atol = ElectronPhonon.toleranse_tests
    @test norm11 > atol # not enforce brakets in real space to normolize to 1 anymore
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test determining correct fft grid from scf.out" begin
    path_tst_data = "test_data/displacements/scf_0/scf.out"
    Nxyz = ElectronPhonon.determine_fft_grid(path_tst_data)

    @test isapprox(Nxyz, [36, 36, 36]; atol=1e-14)
end

@testset "Test determining the phase of ik != 0 wavefunction" begin
    path_tst_data = "test_data/displacements/scf_0/"
    ik = 2

    Nxyz = ElectronPhonon.determine_fft_grid(path_tst_data * "scf.out")
    miller, evc_list = ElectronPhonon.parse_wf(path_tst_data * "tmp/scf.save/wfc2")
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    q_point = ElectronPhonon.determine_q_point(path_tst_data, ik)
    phase = ElectronPhonon.determine_phase(q_point, Nxyz)
    wfc_list_phase = [wfc .* phase for wfc in wfc_list]

    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list_phase[1], wfc_list_phase[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list_phase[1], wfc_list_phase[2]))

    atol = ElectronPhonon.toleranse_tests
    @test norm11 > atol # not enforce brakets in real space to normolize to 1 anymore
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end

@testset "Test unfolding undisplaced wavefunction to supercell" begin
    path_tst_data = "test_data/displacements/"
    ik = 2
    sc_size = [2,2,2]
    Nxyz = ElectronPhonon.determine_fft_grid(path_tst_data * "scf_0/scf.out")

    miller, evc_list = ElectronPhonon.parse_wf(path_tst_data * "/scf_0/tmp/scf.save/wfc$ik")
    miller_sc, _ = ElectronPhonon.parse_wf(path_tst_data * "/group_1/tmp/scf.save/wfc1")

    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    wfc_list_sc = [ElectronPhonon.wf_pc_to_sc(wfc, sc_size) for wfc in wfc_list]

    q_point = ElectronPhonon.determine_q_point(path_tst_data * "scf_0/", ik; sc_size=sc_size)
    phase = ElectronPhonon.determine_phase(q_point, sc_size.*Nxyz)

    wfc_list_phase = [wfc .* phase for wfc in wfc_list_sc]
    evc_list_new = [ElectronPhonon.wf_to_G(miller_sc, wfc, sc_size.*Nxyz) for wfc in wfc_list_phase]

    norm11 = abs(ElectronPhonon.calculate_braket(evc_list_new[1], evc_list_new[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list_new[1], evc_list_new[2]))

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)

end


@testset "Test unfolding undisplaced wavefunction to supercell in automated way" begin
    path_tst_data = "test_data/displacements/"
    ik = 2
    sc_size = [2,2,2]
    global_logger(ConsoleLogger(Warn))
    prepare_wave_functions_undisp(path_tst_data, sc_size)

    ψkᵤ_list = load(path_tst_data*"/scf_0/g_list_sc_$ik.jld2")
    ψkᵤ = [ψkᵤ_list["wfc$iband"] for iband in 1:length(ψkᵤ_list)]

    norm11 = abs(ElectronPhonon.calculate_braket(ψkᵤ[1], ψkᵤ[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(ψkᵤ[1], ψkᵤ[2]))

    # Use the Glob package to match all necessary   files in the directory
    files_to_delete1 = glob("g_list_sc*", joinpath(path_tst_data, "scf_0"))
    files_to_delete2 = glob("wfc_list*", joinpath(path_tst_data, "scf_0"))
    files_to_delete  =reduce(vcat, (files_to_delete1, files_to_delete2))


    # Delete each file
    for file in files_to_delete
        rm(file)
    end

    @test isapprox(norm11, 1.0; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(norm12, 0.0; atol=ElectronPhonon.toleranse_tests)
end
