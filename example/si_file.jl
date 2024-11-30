using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads

# Example usage
path_to_calc = "/scratch/apolyukhin/julia_tests/si_tst/"
abs_disp = 1e-3  #1e-4 #0.0005

# println("Displacement: $abs_disp")
# directory_path = "$path_to_calc"#
path_to_qe= "/home/apolyukhin/Soft/sourse/q-e/"
mpi_ranks = 8

# #Params
mesh = 2 #important to change !
Ndispalce = 12

unitcell, scf_parameters = ElectronPhonon.parse_qe_in("$path_to_calc/scf.in")

# a = 5.43052  # in Angstrom

# unitcell_old = Dict(
#     :symbols =>  pylist(["Si", "Si"]),
#     :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
#     [0.0, 0.5 * a, 0.5 * a],
#     [-0.5 * a, 0.5 * a, 0.0]]),
#     :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
#     :masses => pylist([28.08550, 28.08550])
# )


# # Set up the calculation parameters as a Python dictionary
# scf_parameters_old = Dict(
#     :format => "espresso-in",
#     :kpts => pytuple((mesh, mesh, mesh)),
#     :calculation =>"scf",
#     :prefix => "scf",
#     :outdir => "./tmp/",
#     :pseudo_dir => "/home/apolyukhin/Development/frozen_phonons/elph/example/pseudo",
#     :ecutwfc => 60,
#     :conv_thr =>1.e-13,# 1e-16,# #1.e-20,#5.e-30
#     :pseudopotentials => Dict("Si" => "Si.upf"),
#    # :diagonalization => "ppcg",#"ppcg",#"ppcg",#david
#     :mixing_mode => "plain",
#     :mixing_beta => 0.7,
#     :crystal_coordinates => true,
#     :verbosity => "high",
#     :tstress => false,
#     :ibrav => 2,
#     :tprnfor => true,
#     :nbnd => 10,
#     :electron_maxstep => 1000,
#     :nosym=> true,
#     :noinv=> true#,
#     # :input_dft => "HSE",
#     # :nqx1 => 1,
#     # :nqx2 => 1,
#     # :nqx3 => 1
# )

# function compare_dicts(dict1::Dict, dict2::Dict)
#     # Collect all unique keys from both dictionaries
#     all_keys = union(keys(dict1), keys(dict2))

#     for key in all_keys
#         # Check if the key is present in both dictionaries
#         if haskey(dict1, key) && haskey(dict2, key)
#             # If both keys exist, compare their values
#             # if dict1[key] != dict2[key]
#             #     println("Difference found for key '$key':")
#             #     println("  dict1[$key] = $(dict1[key])")
#             #     println("  dict2[$key] = $(dict2[key])")
#             # end
#             println("key '$key':")
#             println("  dict1[$key] = $(dict1[key])")
#             println("  dict2[$key] = $(dict2[key])")
#         else
#             # If a key is only in one dictionary
#             println("Key '$key' is missing in one of the dictionaries:")
#             if haskey(dict1, key)
#                 println("  dict1[$key] = $(dict1[key])")
#             else
#                 println("  dict1[$key] is missing")
#             end
#             if haskey(dict2, key)
#                 println("  dict2[$key] = $(dict2[key])")
#             else
#                 println("  dict2[$key] is missing")
#             end
#         end
#     end
# end

# Example usage
# compare_dicts(scf_parameters,scf_parameters_old)


#compare if unitcell and unitcell_old are the same field by field



# exit(3)


# So for the HSE we need to do scf with nq1=nq2=nq3=2 and provide the kpoints in the scf.in file (check if nsf is an option)
# For the supercell nq1=nq2=nq3=1 to be consitnent ?
# use_symm = false

model = create_model(path_to_calc, abs_disp, path_to_qe, mpi_ranks, mesh, Ndispalce, unitcell, scf_parameters)

# model = create_model(path_to_calc = path_to_calc,
#                       abs_disp = abs_disp,
#                       path_to_qe = path_to_qe,
#                       mpi_ranks = mpi_ranks,
#                       sc_size = sc_size,
#                       Ndispalce = 1,
#                       unitcell = unitcell,
#                       scf_parameters = scf_parameters,
#                       use_symm = use_symm)


create_disp_calc(model; from_scratch = false)
# run_calculations(model)
# prepare_model(model)

# electrons = create_electrons(model)
# # # ### Phonons calculation
# phonons = create_phonons(model)

# # Loading option instead of calculation
# electrons = load_electrons(model)
# phonons = load_phonons(model)

# # # # #### Electron-phonon matrix elements
# ik_list = [2]#[1,2,3] # [i for i in 1:sc_size^3] ##[1,2]##
# iq_list = [2]#[1,2,3] # [i for i in 1:sc_size^3] ##[1,2]##

# progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

# println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
# for ik in ik_list #@threads
#     for iq in iq_list
#         # electron_phonon_qe(model, ik, iq)
#         electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
#         plot_ep_coupling(model, ik, iq, nbnd_max = 8)
#         next!(progress)
#     end
# end
