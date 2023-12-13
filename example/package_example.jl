using ElectronPhonon, PythonCall

# Example usage
directory_path = "/home/apolyukhin/Development/julia_tests/qe_inputs_ibrav/"#
path_to_kmesh = "/home/apolyukhin/Soft/sourse/q-e/W90/utility/"
mpi_ranks = 10

#Params
ik = 1
iq = 2
mesh = 4 #important to change !
abs_disp = 0.01 
Ndispalce = 12

# Lattice constant of Silicon
a = 5.43052  # in Angstrom
angstrom_to_bohr = 1.88973 # Need to understand why

unitcell = Dict(
    :symbols =>  pylist(["Si", "Si"]),
    :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
    [0.0, 0.5 * a, 0.5 * a],
    [-0.5 * a, 0.5 * a, 0.0]]),
    :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
    :masses => pylist([28.08550,28.08550])
)

# Set up the calculation parameters as a Python dictionary
scf_parameters = Dict(
    :format => "espresso-in",
    :kpts => pytuple((mesh, mesh, mesh)),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/home/apolyukhin/Development/frozen_phonons/elph/example/pseudo",
    :ecutwfc => 60,
    :conv_thr => 1.6e-8,
    :pseudopotentials => Dict("Si" => "Si.upf"),
    :diagonalization => "david",
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true
)


# Electrons calculation
Ndispalce = create_disp_calc(directory_path, unitcell, scf_parameters, abs_disp, mesh; from_scratch = true)
run_disp_calc(directory_path*"displacements/", Ndispalce, mpi_ranks)
run_nscf_calc(directory_path, unitcell, scf_parameters, mesh, path_to_kmesh, mpi_ranks)
save_potential(directory_path*"displacements/", Ndispalce, mesh)

prepare_wave_functions_undisp(directory_path*"displacements/", ik, iq, mesh)
### prepare_wave_functions_all(directory_path*"displacements/", ik, iq, mesh, Ndispalce)

## Phonons calculation
calculate_phonons(directory_path*"displacements/",unitcell, abs_disp, Ndispalce, mesh, iq)

# Electron-phonon matrix elements
electron_phonon_qe(directory_path*"displacements/", ik, iq, mpi_ranks)
electron_phonon(directory_path*"displacements/", abs_disp, Ndispalce, ik, iq, mesh)
plot_ep_coupling(directory_path*"displacements/")
