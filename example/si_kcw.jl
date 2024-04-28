using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads

# Example usage
path_to_calc = "/exports/work/poliukhi/koopmans_scripts/projectability_silicon_new_4/convergence/0.001/"
spin_channel = "up"
abs_disp = 1e-3  #1e-4 #0.0005

println("Displacement: $abs_disp")
path_to_qe = "/home/poliukhi/soft/q-e/"
mpi_ranks = 1

#Params
mesh      = 4 #important to change !
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
    :pseudo_dir => "/home/poliukhi/pseudo",
    :ecutwfc => 60,
    :conv_thr =>1.e-13,# 1e-16,# #1.e-20,#5.e-30
    :pseudopotentials => Dict("Si" => "Si.upf"),
    #:diagonalization => "ppcg",#"ppcg",#"ppcg",#david
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true,
    :nbnd => 20,
    :electron_maxstep => 1000,
    :nosym=> false,
    :noinv=> false
)


model = create_model_kcw(path_to_calc, spin_channel, abs_disp, path_to_qe, mpi_ranks, mesh, Ndispalce, unitcell, scf_parameters)
# create_disp_calc(model)
# prepare_model(model)

electrons = create_electrons(model)
phonons   = create_phonons(model)

###### Electron-phonon matrix elements
ik_list = [i for i in 1:mesh^3] ##[1,2]##
iq_list = [i for i in 1:mesh^3] ##[1,2]##

progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
for ik in ik_list #@threads
    for iq in iq_list
        electron_phonon(model, ik, iq, electrons, phonons; save_epw = true)
        next!(progress)
    end
end
