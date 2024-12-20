using ElectronPhonon, JSON3, PythonCall, ProgressMeter

#read frozen_params.json and parse it
frozen_params = parse_frozen_params("frozen_params.json")
unitcell, scf_parameters = parse_qe_in("scf.in")

model = create_model(path_to_calc = pwd()*"/",
                     abs_disp = frozen_params["abs_disp"],
                     path_to_qe = frozen_params["path_to_qe"],
                     mpi_ranks = frozen_params["mpi_ranks"],
                     sc_size = frozen_params["sc_size"],
                     use_symm = frozen_params["use_symm"],
                     unitcell = unitcell,
                     scf_parameters = scf_parameters)

create_disp_calc!(model; from_scratch = true)
run_calculations(model)
prepare_model(model)

electrons = create_electrons(model)
phonons = create_phonons(model)

# Loading option instead of calculation
# electrons = load_electrons(model)
# phonons = load_phonons(model)

# Electron-phonon matrix elements
ik_list = [1,2,3]# [i for i in 1:sc_size^3]
iq_list = [1,2,3]# [i for i in 1:sc_size^3]

progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
for ik in ik_list
    for iq in iq_list
        electron_phonon_qe(model, ik, iq)
        electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
        plot_ep_coupling(model, ik, iq)
        next!(progress)
    end
end
