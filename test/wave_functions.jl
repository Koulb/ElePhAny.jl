using Test
using JLD2
using Glob
using Logging 

using ElectronPhonon

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
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test parsing wavefunctions from binaries of QE" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/wfc1.dat"

    _, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data)
    norm11 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[2]))
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test parsing wavefunctions from binaries of QE" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/wfc1.dat"

    _, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data)
    norm11 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list[2]))
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test transforming wavefunction to real space" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/wfc1.dat"
    Nxyz = 36

    miller, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data)
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]

    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[2]))
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test transforming wavefunction to real space with slow fft" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/wfc1.dat"
    miller, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data)
    Nxyz = 20

    wfc_list = [ElectronPhonon.wf_from_G_slow(miller, evc, Nxyz) for evc in evc_list]

    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list[1], wfc_list[2]))
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test transforming wavefunction back to reciprocal space" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/wfc1.dat"
    Nxyz = 36

    miller, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data)
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    evc_list_new = [ElectronPhonon.wf_to_G(miller, wfc, Nxyz) for wfc in wfc_list]
    
    norm11 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list_new[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list[1], evc_list_new[2])) 
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test unfolding wavefunctions in supercell" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/wfc1.dat"
    Nxyz = 36

    miller, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data)
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    wfc_list_sc = [ElectronPhonon.wf_pc_to_sc(wfc, 2) for wfc in wfc_list]
    
    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list_sc[1], wfc_list_sc[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list_sc[1], wfc_list_sc[2])) 
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test determining correct fft grid from scf.out" begin
    path_tst_data = "test_data/scf_0/scf.out"
    Nxyz = ElectronPhonon.determine_fft_grid(path_tst_data)

    @test isapprox(Nxyz, 36; atol=1e-14) 
end

@testset "Test determining the phase of ik != 0 wavefunction" begin
    path_tst_data = "test_data/scf_0/"
    ik = 2

    Nxyz = ElectronPhonon.determine_fft_grid(path_tst_data * "scf.out")
    miller, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data * "tmp/scf.save/wfc2.dat")
    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    q_point = ElectronPhonon.determine_q_point(path_tst_data, ik)
    phase = ElectronPhonon.determine_phase(q_point, Nxyz)
    wfc_list_phase = [wfc .* phase for wfc in wfc_list]
 
    norm11 = abs(ElectronPhonon.calculate_braket_real(wfc_list_phase[1], wfc_list_phase[1]))
    norm12 = abs(ElectronPhonon.calculate_braket_real(wfc_list_phase[1], wfc_list_phase[2])) 
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end

@testset "Test unfolding undisplaced wavefunction to supercell" begin
    path_tst_data = "test_data/"
    ik = 2
    mesh = 2
    Nxyz = ElectronPhonon.determine_fft_grid(path_tst_data * "scf_0/scf.out")

    miller, evc_list = ElectronPhonon.parse_fortan_bin(path_tst_data * "/scf_0/tmp/scf.save/wfc$ik.dat")
    miller_sc, _ = ElectronPhonon.parse_fortan_bin(path_tst_data * "/group_1/tmp/scf.save/wfc1.dat") 

    wfc_list = [ElectronPhonon.wf_from_G(miller, evc, Nxyz) for evc in evc_list]
    wfc_list_sc = [ElectronPhonon.wf_pc_to_sc(wfc, mesh) for wfc in wfc_list]

    q_point = ElectronPhonon.determine_q_point(path_tst_data * "scf_0/", ik; mesh=mesh)    
    phase = ElectronPhonon.determine_phase(q_point, mesh*Nxyz)

    wfc_list_phase = [wfc .* phase for wfc in wfc_list_sc]
    evc_list_new = [ElectronPhonon.wf_to_G(miller_sc, wfc, mesh*Nxyz) for wfc in wfc_list_phase]

    norm11 = abs(ElectronPhonon.calculate_braket(evc_list_new[1], evc_list_new[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(evc_list_new[1], evc_list_new[2])) 
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)

end


@testset "Test unfolding undisplaced wavefunction to supercell in automated way" begin
    path_tst_data = "test_data/"
    ik = 2
    mesh = 2
    global_logger(ConsoleLogger(Warn))
    prepare_wave_functions_undisp(path_tst_data, mesh)

    ψkᵤ_list = load(path_tst_data*"/scf_0/g_list_sc_$ik.jld2")
    ψkᵤ = [ψkᵤ_list["wfc$iband"] for iband in 1:length(ψkᵤ_list)]
 
    norm11 = abs(ElectronPhonon.calculate_braket(ψkᵤ[1], ψkᵤ[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(ψkᵤ[1], ψkᵤ[2]))  

    # Use the Glob package to match all .jld2 files in the directory
    files_to_delete = glob("*.jld2", joinpath(path_tst_data, "scf_0"))
    
    # Delete each file
    for file in files_to_delete
        rm(file)
    end

    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12, 0.0; atol=1e-14)
end
