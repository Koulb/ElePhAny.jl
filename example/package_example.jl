using ElectronPhonon, PythonCall, ProgressMeter

# Example usage
# list_of_disp = ["0005", "001", "0025", "005", "01", "025", "05"]
# abs_disp = parse(Float64, "0."*abs_disp_str)

path_to_calc = "/scratch/apolyukhin/julia_tests/qe_inputs/"
abs_disp = 0.001

println("Displacement: $abs_disp")
directory_path = "$path_to_calc"#
path_to_kmesh = "/home/apolyukhin/Soft/sourse/q-e/W90/utility/"
mpi_ranks = 10

# path_to_kcw = "/scratch/apolyukhin/scripts/koopmans/koopmans_scripts/in/projectability_silicon_kcw_crs_sc_new/0.001/"
# kcw_chanel = "up"

#Params
mesh = 2 #important to change !
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
    :ecutwfc => 80,
    :conv_thr => 1.e-13,
    :pseudopotentials => Dict("Si" => "Si.upf"),
    :diagonalization => "david",
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true#,
    #:nosym=> false,
    #:input_dft => "HSE"
)
#So for the HSE we need to do scf with nq1=nq2=nq3=2 and provide the kpoints in the scf.in file (check if nsf is an option)
#For the supercell nq1=nq2=nq3=1 to be consitnent ?


# # ## Electrons calculation
Ndispalce = create_disp_calc(directory_path, unitcell, scf_parameters, abs_disp, mesh; from_scratch = true)
run_disp_calc(directory_path*"displacements/", Ndispalce, mpi_ranks)
run_nscf_calc(directory_path, unitcell, scf_parameters, mesh, path_to_kmesh, mpi_ranks)
save_potential(directory_path*"displacements/", Ndispalce, mesh)


prepare_wave_functions_undisp(directory_path*"displacements/", mesh)#; path_to_kcw=path_to_kcw,kcw_chanel=kcw_chanel

### Phonons calculation
calculate_phonons(directory_path*"displacements/",unitcell, abs_disp, Ndispalce, mesh)

## Electron-phonon matrix elements
ik_list = [i for i in 1:mesh^3]#[1,2]##[1]#[1,8]##[1]
iq_list = [i for i in 1:mesh^3]#[1,2]##[1]#[1,6,8]#

progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
for ik in ik_list
    for iq in iq_list
        electron_phonon_qe(directory_path*"displacements/", ik, iq, mpi_ranks)
        electron_phonon(directory_path*"displacements/", abs_disp, Ndispalce, ik, iq, mesh)#;save_epw = true# path_to_kcw=path_to_kcw,kcw_chanel=kcw_chanel,
        plot_ep_coupling(directory_path*"displacements/"; ik, iq)
        next!(progress)
    end
end