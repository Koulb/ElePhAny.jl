using Test
using ElectronPhonon

@testset "Test parsing eigenvalues from xml of QE" begin
    path_tst_data = "test_data/data-file-schema.xml"
    energy = ElectronPhonon.read_qe_xml(path_tst_data)[:eigenvalues][1]

    check = [-5.490708, 6.530881, 6.530881, 6.530881]
    @test all(isapprox.(energy, check; atol=1e-6))
end
