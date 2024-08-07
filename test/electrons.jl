using Test
using ElectronPhonon

@testset "Test parsing eigenvalues from xml of QE" begin
    path_tst_data = "test_data/scf_0/tmp/scf.save/data-file-schema.xml"
    energy = ElectronPhonon.read_qe_xml(path_tst_data)[:eigenvalues][1]
    check = [-5.49066814, 6.5309217, 6.5309217, 6.5309217]
    @test all(isapprox.(energy, check; atol=1e7*ElectronPhonon.toleranse_tests))
end

@testset "Test parsing the k-points from kpoints.dat file" begin
    path_tst_data = "test_data/scf_0/"
    kpoints = ElectronPhonon.get_kpoint_list(path_tst_data)
    
    check = [[0.0, 0.0, 0.0], 
             [0.0, 0.0, 0.5], 
             [0.0, 0.5, 0.0], 
             [0.0, 0.5, 0.5], 
             [0.5, 0.0, 0.0], 
             [0.5, 0.0, 0.5], 
             [0.5, 0.5, 0.0], 
             [0.5, 0.5, 0.5]]

    @test all(isapprox.(check, kpoints; atol=ElectronPhonon.toleranse_tests))
end

@testset "Test folding k-point in 1st BZ" begin
    path_tst_data = "test_data/scf_0/"
    ik = 2   #[0.0 0.0 0.5]
    iq = 4   #[0.0 0.5 0.5]

    kpoints = ElectronPhonon.get_kpoint_list(path_tst_data)
    ikq = ElectronPhonon.fold_kpoint(ik,iq,kpoints) #[0.0 0.5 0.0]
    kq_point = kpoints[ikq]
    kq_point_check = kpoints[3]

    @test all(isapprox.(kq_point_check, kq_point; atol=ElectronPhonon.toleranse_tests))
end

##TODO: add some test on IO of QE
