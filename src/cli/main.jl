using Comonicon

"""

# Options

- '--qe_in_file <arg>': path to the Quantum Espresso input file
- '--frozen_params_file <arg>': path to the frozen parameters file

# Flags

- `-c, --create_disp`: create the displacement files
- `-f, --from_scratch`: create
- `-r, --run_calculations`: run the calculations
- `-p, --prepare_model`: prepare the model
- `-l, --load_data`: load the data
- `-e, --ep_calculations`: perform calculations for electron-phonon matrix elements
- `-s, --save_epw`: save the electron-phonon matrix elements in EPW format

"""

# /home/apolyukhin/.julia/bin/epjl

@main function epjl(; qe_in_file::String="scf.in",
                      frozen_params_file::String="frozen_params.json",
                      create_disp::Bool=false,
                      from_scratch::Bool=false,
                      run_calc::Bool=false,
                      prepare::Bool=false,
                      load_data::Bool=false,
                      ep_calculations::Bool=false,
                      save_epw::Bool=false,
                    )

    println("Running epjl")

    #read frozen_params.json and parse it
    frozen_params = parse_frozen_params(frozen_params_file)
    unitcell, scf_parameters = parse_qe_in(qe_in_file)
    sc_size  = frozen_params["sc_size"]

    model = create_model(path_to_calc = frozen_params["path_to_calc"],
                         abs_disp = frozen_params["abs_disp"],
                         path_to_qe = frozen_params["path_to_qe"],
                         mpi_ranks = frozen_params["mpi_ranks"],
                         sc_size  = sc_size,
                         k_mesh  = frozen_params["k_mesh"],
                         Ndispalce = frozen_params["Ndispalce"],
                         use_symm = frozen_params["use_symm"],
                         unitcell = unitcell,
                         scf_parameters = scf_parameters)

    if create_disp
        create_disp_calc!(model; from_scratch = from_scratch)
    end

    if run_calc
        run_calculations(model)
    end

    local electrons, phonons

    if prepare
        prepare_model(model)
        electrons = create_electrons(model)
        phonons = create_phonons(model)
    end

    if load_data
        electrons = load_electrons(model)
        phonons = load_phonons(model)
    end

    if ep_calculations
        ####### Electron-phonon matrix elements
        ik_list =  [i for i in 1:sc_size^3] ##[1,2]##
        iq_list =  [i for i in 1:sc_size^3] ##[1,2]##

        # progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

        println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
        for ik in ik_list #@threads
            for iq in iq_list
                if save_epw
                    electron_phonon(model, ik, iq, electrons, phonons; save_epw = save_epw)
                    println("ik = $ik, iq = $iq done")
                else
                    electron_phonon_qe(model, ik, iq)
                    electron_phonon(model, ik, iq, electrons, phonons; save_epw = save_epw)
                    plot_ep_coupling(model, ik, iq)
                    println("ik = $ik, iq = $iq done")
                    # next!(progress)
                end
            end
        end
    end

end
