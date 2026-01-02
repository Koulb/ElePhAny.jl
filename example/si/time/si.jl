using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads#CUDA, 
using Dates

#flags
create = true
from_scratch = false
run = false
prepare = true
calc_ep = true

# Example usage
path_to_calc = pwd() * "/"
abs_disp = 1e-3

println("Displacement: $abs_disp")
directory_path = "$path_to_calc"#
path_to_qe= "/home/apolyukhin/Development/q-e_tmp/"
mpi_ranks = 8

#Params
sc_size = [1,1,1]
k_mesh = [4,4,4]
nbnd = 10

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
    :pseudo_dir => "/home/apolyukhin/scripts/electron_phonon/si_2/displacements/scf_0/tmp/scf.save",
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
    :nbnd => nbnd,
    :nosym => true,
    :noinv => true,
    :electron_maxstep => 1000
)

use_symm = true

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

prepare_duration = 0.0
calc_ep_duration = 0.0
prepare_start = nothing
calc_start = nothing

if prepare
    prepare_model(model)
    prepare_start = now()
    t0 = time()
    electrons = create_electrons(model)
    prepare_duration = time() - t0
    phonons = create_phonons(model)
end

if calc_ep
    calc_start = now()
    t0 = time()
    # Loading option instead of calculation
    electrons = load_electrons(model)
    phonons = load_phonons(model)

    # Electron-phonon matrix elements
    ik_list = [1,2,3]
    iq_list = [1]

    # progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

    println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
    for ik in ik_list
        for iq in iq_list
            electron_phonon_qe(model, ik, iq)
            electron_phonon(model, ik, iq, electrons, phonons;)
            # plot_ep_coupling(model, ik, iq, nbnd_max = 8)
            # next!(progress)
        end
    end
    calc_ep_duration = time() - t0
end

# Dump timings and run timestamp to file for later checks
timings_file = joinpath(path_to_calc, "timings.txt")
open(timings_file, "a") do io
    println(io, "Run recorded at: $(now())")
    println(io, "# Bands: $(nbnd) ik_list length: $(length(ik_list))  iq_list length: $(length(iq_list))")

    if prepare_start !== nothing
        println(io, "prepare duration: $(round(prepare_duration, digits=3)) s")
    else
        println(io, "prepare: skipped")
    end
    if calc_start !== nothing
        println(io, "calc_ep duration: $(round(calc_ep_duration, digits=3)) s")
    else
        println(io, "calc_ep: skipped")
    end
    println(io, "-"^60)
end
