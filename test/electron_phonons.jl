using Test, PythonCall, Logging, ElectronPhonon

@testset "Test calculating brakets using projectability approach" begin
    path_tst_data = "test_data/"
    path_to_qe = ""
    mesh = 2 
    Ndispalce = 12
    mpi_ranks = 1   
    ik = 2   #[0.0 0.0 0.5]
    iq = 3   #[0.0 0.5 0.0]
    abs_disp = 0.001
    nat = 2
    global_logger(ConsoleLogger(Warn))

    unitcell = Dict(
        :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
        :masses => pylist([28.08550, 28.08550])
    )

    scf_parameters = Dict(
        :nbnd => 4
    )

    model = create_model(path_tst_data, abs_disp, path_to_qe, mpi_ranks, mesh, Ndispalce, unitcell, scf_parameters)

    electrons = ElectronPhonon.load_electrons(model)
    phonons = ElectronPhonon.load_phonons(model)

    brakets = ElectronPhonon.electron_phonon(model, ik, iq, electrons, phonons; save_epw = true) #save_epw = true
    brakets_test = [-0.027596679111611726 + 0.06331498639687654im, 0.0020688309462811513 + 0.011649138869732529im]

    @test isapprox(brakets[1][2][3,4], brakets_test[1]; atol=ElectronPhonon.toleranse_tests)
    @test isapprox(brakets[2][1][4,3], brakets_test[2]; atol=ElectronPhonon.toleranse_tests)

end

@testset "Test calculating electron-phonon matrix elements and comparing with DFPT" begin
    path_tst_data = "test_data/"
    path_to_qe = ""
    mesh = 2 
    Ndispalce = 12
    mpi_ranks = 1   
    ik = 2   #[0.0 0.0 0.5]
    iq = 3   #[0.0 0.5 0.0]
    abs_disp = 0.001
    nat = 2
    global_logger(ConsoleLogger(Warn))

    unitcell = Dict(
        :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
        :masses => pylist([28.08550, 28.08550])
    )

    scf_parameters = Dict(
        :nbnd => 4
    )

    model = create_model(path_tst_data, abs_disp, path_to_qe, mpi_ranks, mesh, Ndispalce, unitcell, scf_parameters)

    electrons = ElectronPhonon.load_electrons(model)
    phonons = ElectronPhonon.load_phonons(model)

    elph_frozen = ElectronPhonon.electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
    elph_dfpt   = ElectronPhonon.parse_ph(path_tst_data*"displacements/scf_0/ph.out", scf_parameters[:nbnd], 3*nat)

    @test all(isapprox.(elph_frozen, elph_dfpt; atol=1e-3))

end