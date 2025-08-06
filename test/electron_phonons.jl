using Test, PythonCall, Logging, ElectronPhonon

@testset "Test parsing electron-phonon data from QE" begin
    path_tst_data = "test_data/displacements/"
    nbands = 4
    nat = 2

    elph_dfpt = ElectronPhonon.parse_ph(path_tst_data*"scf_0/ph.out", nbands, 3*nat)
    ωₐᵣᵣ_DFPT, _ = ElectronPhonon.parse_qe_ph(path_tst_data*"scf_0/dyn1", nat)
    # println(ωₐᵣᵣ_DFPT)
    # println(elph_dfpt[2,:,3])
    elph_dfpt_test = ComplexF64[0.044206963440000004 + 0.0im, 0.044206963440000004 + 0.0im, 5.002807184e-10 + 0.0im, 5.002807184e-10 + 0.0im]
    ωₐᵣᵣ_DFPT_test = [-57.728206 -57.728206 369.259079 420.354152 479.315539 479.315539]

    @test all(isapprox.(ωₐᵣᵣ_DFPT_test, ωₐᵣᵣ_DFPT; atol=ElectronPhonon.toleranse_tests))
    @test all(isapprox.(elph_dfpt_test, elph_dfpt[2,:,3]; atol=ElectronPhonon.toleranse_tests))

end

@testset "Test calculating brakets using projectability approach" begin
    path_tst_data = "test_data/"
    path_to_qe = ""
    sc_size = [2, 2, 2]
    k_mesh  = [1, 1, 1]
    Ndispalce = 12
    mpi_ranks = 1
    use_symm = false

    ik = 2   #[0.0 0.0 0.5]
    iq = 3   #[0.0 0.5 0.0]
    abs_disp = 0.001
    nat = 2
    global_logger(ConsoleLogger(Warn))

    unitcell = Dict(
        :symbols =>  pylist(["Si", "Si"]),
        :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
        :masses => pylist([28.08550, 28.08550])
    )

    scf_parameters = Dict(
        :nbnd => 4
    )

    model = create_model(path_to_calc = path_tst_data,
                         abs_disp = abs_disp,
                         path_to_qe = path_to_qe,
                         mpi_ranks = mpi_ranks,
                         sc_size = sc_size,
                         k_mesh = k_mesh,
                         Ndispalce = Ndispalce,
                         unitcell = unitcell,
                         scf_parameters = scf_parameters,
                         use_symm = use_symm)

    electrons = ElectronPhonon.load_electrons(model)
    phonons = ElectronPhonon.load_phonons(model)

    brakets = ElectronPhonon.electron_phonon(model, ik, iq, electrons, phonons; save_epw = true) #save_epw = true
    brakets_test = [-0.08929173722566114 + 0.01718059867889217im, 0.04280449975955811 + 0.015614864644048581im]

    @test isapprox(brakets[1][2][3,4], brakets_test[1]; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(brakets[2][1][4,3], brakets_test[2]; atol=ElectronPhonon.toleranse_tests)

end

@testset "Test calculating electron-phonon matrix elements and comparing with DFPT" begin
    path_tst_data = "test_data/"
    path_to_qe = ""
    sc_size = [2, 2, 2]
    Ndispalce = 12
    mpi_ranks = 1
    k_mesh = [1, 1, 1]
    use_symm = false
    ik = 2   #[0.0 0.0 0.5]
    iq = 3   #[0.0 0.5 0.0]
    abs_disp = 0.001
    nat = 2
    global_logger(ConsoleLogger(Warn))

    unitcell = Dict(
        :symbols =>  pylist(["Si", "Si"]),
        :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
        :masses => pylist([28.08550, 28.08550])
    )

    scf_parameters = Dict(
        :nbnd => 4
    )

    model = create_model(path_to_calc = path_tst_data,
                         abs_disp = abs_disp,
                         path_to_qe = path_to_qe,
                         mpi_ranks = mpi_ranks,
                         sc_size = sc_size,
                         k_mesh = k_mesh,
                         Ndispalce = Ndispalce,
                         unitcell = unitcell,
                         scf_parameters = scf_parameters,
                         use_symm = use_symm)

    electrons = ElectronPhonon.load_electrons(model)
    phonons = ElectronPhonon.load_phonons(model)

    elph_frozen = ElectronPhonon.electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
    elph_dfpt   = ElectronPhonon.parse_ph(path_tst_data*"displacements/scf_0/ph.out", scf_parameters[:nbnd], 3*nat)

    @test all(isapprox.(elph_frozen, elph_dfpt; atol=1e-3))

end
