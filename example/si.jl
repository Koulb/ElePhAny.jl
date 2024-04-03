using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads

# Example usage
path_to_calc = "/scratch/apolyukhin/julia_tests/si_tst/"
abs_disp = 1e-3  #1e-4 #0.0005

println("Displacement: $abs_disp")
directory_path = "$path_to_calc"#
path_to_qe= "/home/apolyukhin/Soft/sourse/q-e/"
mpi_ranks = 10

#Params
mesh = 2 #important to change !
Ndispalce = 12

# Lattice constant of Silicon
a = 5.43052  # in Angstrom

unitcell = Dict(
    :symbols =>  pylist(["Si", "Si"]),
    :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
    [0.0, 0.5 * a, 0.5 * a],
    [-0.5 * a, 0.5 * a, 0.0]]),
    :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
    :masses => pylist([28.08550, 28.08550])
)

# Set up the calculation parameters as a Python dictionary
scf_parameters = Dict(
    :format => "espresso-in",
    :kpts => pytuple((mesh, mesh, mesh)),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/home/apolyukhin/Development/frozen_phonons/elph/example/pseudo",
    :ecutwfc => 80,
    :conv_thr =>1.e-16,# 1e-16,# #1.e-20,#5.e-30
    :pseudopotentials => Dict("Si" => "Si.upf"),
    # :diagonalization => "ppcg",#"ppcg",#"ppcg",#david
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true,
    :nbnd => 14,
    :electron_maxstep => 1000,
    :nosym=> false,
    :noinv=> false
    # :input_dft => "HSE",
    # :nqx1 => 1,
    # :nqx2 => 1,
    # :nqx3 => 1
)

# So for the HSE we need to do scf with nq1=nq2=nq3=2 and provide the kpoints in the scf.in file (check if nsf is an option)
# For the supercell nq1=nq2=nq3=1 to be consitnent ?

model = create_model(path_to_calc, abs_disp, directory_path, path_to_qe, mpi_ranks, mesh, Ndispalce, unitcell, scf_parameters)
run_model(model)

# electrons = create_electrons(model)

# ### Phonons calculation
# phonons = create_phonons(model)

# # #### Electron-phonon matrix elements
# ik_list = [i for i in 1:mesh^3] ##[1,2]##
# iq_list = [i for i in 1:mesh^3] ##[1,2]##

# progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

# println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
# for ik in ik_list #@threads
#     for iq in iq_list
#         # electron_phonon_qe(model, ik, iq)
#         electron_phonon(model, ik, iq, electrons, phonons; save_epw = true)
#         # plot_ep_coupling(model, ik, iq)
#         next!(progress)
#     end
# end