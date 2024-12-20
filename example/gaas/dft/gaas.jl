using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads

# Example usage
path_to_calc = pwd()*"/"
abs_disp = 5e-4

println("Displacement: $abs_disp")
directory_path = "$path_to_calc"#
path_to_qe = "/home/apolyukhin/Soft/sourse/q-e/"
mpi_ranks  = 10

#Params
sc_size = 2

# Lattice constant of GaAs
a = 5.65325  # in Angstrom

unitcell = Dict(
    :symbols =>  pylist(["Ga", "As"]),
    :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
    [0.0, 0.5 * a, 0.5 * a],
    [-0.5 * a, 0.5 * a, 0.0]]),
    :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
    :masses => pylist([69.723, 74.921595])
)

# Set up the calculation parameters as a Python dictionary
scf_parameters = Dict(
    :format => "espresso-in",
    :kpts => pytuple((sc_size, sc_size, sc_size)),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/scratch/apolyukhin/scripts/q-e/new_ewp_tests/Pseudo",
    :ecutwfc => 80,
    :conv_thr =>1.e-13,
    :pseudopotentials => Dict("Ga" => "Ga_ONCV_PBE-1.0.upf", "As" => "As_ONCV_PBE-1.0.upf"),
    :diagonalization => "ppcg",#,#david
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true,
    :nbnd => 12,
    :electron_maxstep => 1000,
    :nosym=> false,
    :noinv=> false
)

use_symm = true

model = create_model(path_to_calc = path_to_calc,
                      abs_disp = abs_disp,
                      path_to_qe = path_to_qe,
                      mpi_ranks = mpi_ranks,
                      sc_size = sc_size,
                      Ndispalce = 2,
                      unitcell = unitcell,
                      scf_parameters = scf_parameters,
                      use_symm = use_symm)

create_disp_calc!(model; from_scratch = false)
run_calculations(model)
prepare_model(model)

electrons = create_electrons(model)
phonons = create_phonons(model)

# Loading option instead of calculation
electrons = load_electrons(model)
phonons = load_phonons(model)

# Electron-phonon matrix elements
ik_list = [1]
iq_list = [3]

progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
for ik in ik_list
    for iq in iq_list
        electron_phonon_qe(model, ik, iq)
        electron_phonon(model, ik, iq, electrons, phonons;)
        plot_ep_coupling(model, ik, iq; nbnd_max = 9)
        next!(progress)
    end
end
