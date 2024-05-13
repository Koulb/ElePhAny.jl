using Test
using ElectronPhonon

@testset "Test parsing wavefunctions from binaries of QE" begin
    path_tst_data = "test_data/wfc1.dat"
    _, wfc = ElectronPhonon.parse_fortan_bin(path_tst_data)
    norm11 = abs(ElectronPhonon.calculate_braket(wfc[1], wfc[1]))
    norm12 = abs(ElectronPhonon.calculate_braket(wfc[1], wfc[2]))
    
    @test isapprox(norm11, 1.0; atol=1e-14) 
    @test isapprox(norm12,0.0; atol=1e-14)
end
