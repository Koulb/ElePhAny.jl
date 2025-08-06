using ElectronPhonon, PythonCall, Glob, Test

@testset "Test determining q_point in crystall coordinates" begin
  path_tst_data = "test_data/displacements/scf_0/"
  iq = 2

  q_point =  ElectronPhonon.determine_q_point(path_tst_data, iq; sc_size=1)
  q_point_check = [0.0, 0.0, 0.5]

  @test all(isapprox.(q_point, q_point_check; atol=ElectronPhonon.toleranse_tests))
end


@testset "Test determining q_point in cartesian coordinates" begin
  path_tst_data = "test_data/displacements/scf_0/"
  iq = 2

  q_point =  ElectronPhonon.determine_q_point_cart(path_tst_data, iq)
  q_point_check = [-0.5, 0.5, -0.5]

  @test all(isapprox.(q_point_check, q_point; atol=ElectronPhonon.toleranse_tests))
end


@testset "Test reading forces from xml file of QE" begin
  path_tst_data = "test_data/displacements/group_1/tmp/scf.save/data-file-schema.xml"

  forces =  ElectronPhonon.read_forces_xml(path_tst_data)
  forces_check = [0.00018426817994402487 1.8604406887485266e-7 -0.00018426817994402487;
                 -4.605458202823222e-6   4.001150202445474e-8   4.605458202823222e-6;
                  2.801757918018952e-5   1.0805770557766172e-5 -6.230884701512572e-6;
                  6.24966252599373e-6   -1.0861113776274836e-5 -2.79865490726555e-5;
                  6.230884701512572e-6   1.080577055776617e-5  -2.801757918018952e-5;
                  2.79865490726555e-5   -1.0861113776274836e-5 -6.24966252599373e-6;
                  1.70620756248411e-5    2.612162094876132e-8  -1.70620756248411e-5;
                 -7.71645614848067e-6    3.33921329959602e-8    7.71645614848067e-6;
                 -6.837333591554268e-6   1.2759429238718131e-5  6.837333591554268e-6;
                 -6.900812718359392e-6  -1.28567062396875e-5    6.900812718359392e-6;
                  5.949950923159774e-6   3.953404285704598e-8  -5.94880795526296e-6;
                 -3.339279622056244e-5   3.20481654582033e-8    3.33910000001958e-5;
                  5.948807955262962e-6   3.953404285704598e-8  -5.949950923159774e-6;
                 -3.33910000001958e-5    3.20481654582033e-8    3.339279622056244e-5;
                 -9.451353817540662e-5  -6.115253962499595e-5   9.451353817540665e-5;
                 -9.435629487025762e-5   6.093176932150812e-5   9.43562948702576e-5]

  @test all(isapprox.(forces_check, forces; atol=ElectronPhonon.toleranse_tests))
end


@testset "Test creating displaced supercells with phonopy" begin
  path_tst_data = "test_data/displacements/scf_0/"
  path_tst_xml  = "test_data/phonopy_params.yaml"
  abs_disp  = 0.001
  sc_size      = [2,2,2]
  a = 5.43052  # in Angstrom
  use_sym = false

  unitcell = Dict(
      :symbols =>  pylist(["Si", "Si"]),
      :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
      [0.0, 0.5 * a, 0.5 * a],
      [-0.5 * a, 0.5 * a, 0.0]]),
      :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
      :masses => pylist([28.08550, 28.08550])
  )

  ElectronPhonon.dislpaced_unitecells(path_tst_data, unitcell,abs_disp, sc_size, use_sym)

  phonon_params = ElectronPhonon.phonopy.load(path_tst_data*"phonopy_params.yaml")
  phonon_params_check = ElectronPhonon.phonopy.load(path_tst_xml)

  displacemnts = pyconvert(Vector{Vector{Float64}}, phonon_params.displacements)
  displacemnts_check = pyconvert(Vector{Vector{Float64}}, phonon_params_check.displacements)

  @test all(isapprox.(displacemnts_check, displacemnts; atol=ElectronPhonon.toleranse_tests))

  # Delete the created files
  rm(path_tst_data*"phonopy_params.yaml")

end


@testset "Test running phonon calculation with phonopy" begin
  path_tst_data = "test_data/displacements/"
  path_tst_xml  = "test_data/phonopy_params.yaml"
  path_tst_group = "test_data/displacements/group_1/"
  abs_disp  = 0.001
  sc_size   = [2,2,2]
  k_mesh    = [1,1,1]
  use_sym   = false
  Ndispalce = 12
  a = 5.43052  # in Angstrom

  unitcell = Dict(
      :symbols =>  pylist(["Si", "Si"]),
      :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
      [0.0, 0.5 * a, 0.5 * a],
      [-0.5 * a, 0.5 * a, 0.0]]),
      :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
      :masses => pylist([28.08550, 28.08550])
  )

  force =  ElectronPhonon.read_forces_xml(path_tst_group * "tmp/scf.save/data-file-schema.xml")
  number_atoms = length(unitcell[:symbols])*prod(sc_size)
  forces = Array{Float64}(undef, Ndispalce, number_atoms, 3)
  for i_disp in 1:Ndispalce
    forces[i_disp,:,:] = force * (-1)^(i_disp) # to have different forces
  end

  ElectronPhonon.prepare_phonons_data(path_tst_data, unitcell,abs_disp, sc_size, k_mesh, use_sym, forces; save_dynq=false)
  # Delete created files
  rm(path_tst_data*"sc_size.conf")
  rm(path_tst_data*"phonopy.yaml")
  rm(path_tst_data*"qpoints.yaml")

  phonon_params = ElectronPhonon.phonopy.load(path_tst_data*"/phonopy_params.yaml")
  phonon_params_check = ElectronPhonon.phonopy.load(path_tst_xml)

  dynq = pyconvert(Matrix{ComplexF64}, phonon_params.get_dynamical_matrix_at_q([0.5, 0.5, 0.0]))
  dynq_check = [-0.004044118682382613 - 0.0im -2.5092357697586932e-20 + 0.0im 3.377817382367472e-20 + 0.0im 2.756622812621252e-19 + 0.0im 3.4286908625220056e-36 + 0.00045539799282151973im 6.857381725044011e-36 - 2.4706321425316367e-19im;
                -2.5092357697586932e-20 - 0.0im -0.004044118682382611 - 0.0im 4.555228012792705e-19 + 0.0im 1.0286072587566016e-35 + 0.00045539799282151973im 2.756622812621252e-19 + 0.0im -2.0572145175132035e-35 - 2.4706321425316367e-19im;
                 3.377817382367472e-20 - 0.0im 4.555228012792705e-19 - 0.0im -0.006962951662729665 - 0.0im -1.3714763450088023e-35 + 0.0im -3.428690862522006e-35 - 2.4706321425316367e-19im 2.756622812621251e-19 + 3.705948213797455e-19im;
                 2.756622812621252e-19 - 0.0im 1.0286072587566016e-35 - 0.00045539799282151973im -1.3714763450088023e-35 - 0.0im 0.004954914668025651 - 0.0im -3.6341695944221465e-20 + 0.0im -1.6080823466770856e-19 + 0.0im;
                 3.4286908625220056e-36 - 0.00045539799282151973im 2.756622812621252e-19 - 0.0im -3.428690862522006e-35 + 2.4706321425316367e-19im -3.6341695944221465e-20 - 0.0im 0.00495491466802565 - 0.0im -7.720725445411372e-21 + 0.0im;
                 6.857381725044011e-36 + 2.4706321425316367e-19im -2.0572145175132035e-35 + 2.4706321425316367e-19im 2.756622812621251e-19 - 3.705948213797455e-19im -1.6080823466770856e-19 - 0.0im -7.720725445411372e-21 - 0.0im -1.5249416540281821e-5 - 0.0im]

  @test all(isapprox.(dynq_check, dynq; atol=1e6*ElectronPhonon.toleranse_tests))

end
