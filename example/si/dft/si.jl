using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads

#flags
create = true
from_scratch = false
run = false
prepare = false
calc_ep = true

# Example usage
path_to_calc = pwd() * "/"
abs_disp = 1e-3

println("Displacement: $abs_disp")
directory_path = "$path_to_calc"#
path_to_qe= "/home/poliukhin/Soft/sourse/q-e/"
mpi_ranks = 8

#Params
sc_size::Vec3{Int} = [1,1,1]
k_mesh::Vec3{Int}  = [2,2,2]

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
    :kpts => pytuple((k_mesh[1]*sc_size[1], k_mesh[2]*sc_size[2], k_mesh[3]*sc_size[3])),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/home/poliukhin/Development/frozen_phonons/elph/example/pseudo",
    :ecutwfc => 60,
    :conv_thr =>1.e-13,# 1e-16,# #1.e-20,#5.e-30
    :pseudopotentials => Dict("Si" => "Si.upf"),
    :diagonalization => "ppcg",#"ppcg",#"ppcg",#david
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true,
    :nbnd => 4,
    :electron_maxstep => 1000,
    :nosym=> true,
    :noinv=> true
)

use_symm = false

model = create_model(path_to_calc = path_to_calc,
                      abs_disp = abs_disp,
                      path_to_qe = path_to_qe,
                      mpi_ranks = mpi_ranks,
                      sc_size = sc_size,
                      k_mesh  = k_mesh,
                      unitcell = unitcell,
                      scf_parameters = scf_parameters,
                      use_symm = use_symm)


if create
    create_disp_calc!(model; from_scratch = from_scratch)
end

if run
    run_calculations(model)
end


if prepare
    prepare_model(model)
    electrons = create_electrons(model)
    phonons = create_phonons(model)
end

if calc_ep
    # Loading option instead of calculation
    electrons = load_electrons(model)
    phonons = load_phonons(model)

    # Electron-phonon matrix elements
    ik_list = [1,2,3,4,5,6,7,8]#[1,2,3,4] # [i for i in 1:sc_size^3] ##[1,2]##
    iq_list = [1]#[1,2,3,4] # [i for i in 1:sc_size^3] ##[1,2]##

    progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

    println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
    for ik in ik_list #@threads
        for iq in iq_list
            electron_phonon_qe(model, ik, iq)# requires to compile special ph.x in testsuite/non_epw_comp
            electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
            plot_ep_coupling(model, ik, iq, nbnd_max = 8)
            next!(progress)
        end
    end
end
