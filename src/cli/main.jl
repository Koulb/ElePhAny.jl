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

Comonicon.@main function epjl(; qe_in_file::String="scf.in",
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
    sc_size::Vec3{Int}  = frozen_params["sc_size"]
    k_mesh::Vec3{Int}  = frozen_params["k_mesh"]

    model = create_model(path_to_calc = frozen_params["path_to_calc"],
                         abs_disp = frozen_params["abs_disp"],
                         path_to_qe = frozen_params["path_to_qe"],
                         mpi_ranks = frozen_params["mpi_ranks"],
                         sc_size  = sc_size,
                         k_mesh  = k_mesh,
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

# MPI-optimized main function
function main_mpi()
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Initialize model
    model = create_model_from_args(args)
    
    # Check if we should use MPI
    use_mpi = mpi_size() > 1
    
    if is_master()
        @info "Starting calculation with $(mpi_size()) MPI processes and $(Threads.nthreads()) threads per process"
    end
    
    # Prepare model (only master does this)
    if is_master()
        prepare_model(model)
    end
    mpi_barrier()
    
    # Load electrons and phonons
    electrons = load_electrons(model)
    phonons = load_phonons(model)
    
    # Broadcast data to all ranks
    if use_mpi
        # Broadcast model parameters
        mpi_bcast!(model.sc_size)
        mpi_bcast!(model.k_mesh)
        mpi_bcast!(model.Ndispalce)
        
        # Broadcast electron and phonon data
        # Note: This is simplified - in practice you'd need to serialize/deserialize
        # the complex data structures for MPI communication
    end
    
    if args.ep_calculations
        # Distribute k-point and q-point calculations across MPI ranks
        sc_size = model.sc_size[1]  # Assuming cubic supercell
        ik_list = collect(1:sc_size^3)
        iq_list = collect(1:sc_size^3)
        
        rank = mpi_rank()
        size = mpi_size()
        
        # Distribute (ik, iq) pairs across ranks
        all_pairs = [(ik, iq) for ik in ik_list for iq in iq_list]
        local_pairs = all_pairs[rank+1:size:end]
        
        if is_master()
            println("Calculating electron-phonon matrix elements for $(length(all_pairs)) points across $(size) MPI processes:")
        end
        
        for (ik, iq) in local_pairs
            if args.save_epw
                electron_phonon_mpi(model, ik, iq, electrons, phonons; save_epw=args.save_epw)
                if is_master()
                    println("Rank $rank: ik = $ik, iq = $iq done")
                end
            else
                electron_phonon_qe(model, ik, iq)
                electron_phonon_mpi(model, ik, iq, electrons, phonons; save_epw=args.save_epw)
                if is_master()
                    plot_ep_coupling(model, ik, iq)
                    println("Rank $rank: ik = $ik, iq = $iq done")
                end
            end
        end
        
        mpi_barrier()
        
        if is_master()
            println("All electron-phonon calculations completed")
        end
    end
end

# Hybrid MPI + threading wave function preparation
function prepare_wave_functions_hybrid(path_to_in::String, sc_size::Vector{Int}; k_mesh::Vector{Int} = [1,1,1])
    rank = mpi_rank()
    size = mpi_size()
    
    total_kpoints = prod(sc_size) * prod(k_mesh)
    local_kpoints = rank+1:size:total_kpoints
    
    if is_master()
        @info "Preparing wave functions for $total_kpoints k-points across $size MPI processes"
    end
    
    for (local_idx, ik) in enumerate(local_kpoints)
        file_path = path_to_in*"/scf_0/"
        
        # Use optimized wave function preparation
        prepare_wave_functions_to_R_optimized(file_path; ik=ik, chunk_size=5)
        prepare_unfold_to_sc(file_path, sc_size, ik)
        prepare_wave_functions_to_G(path_to_in; ik=ik)
        
        if is_master()
            @info "Rank $rank: ik = $ik/$(length(local_kpoints)) is ready"
        end
    end
    
    mpi_barrier()
end

# MPI-optimized displacement calculations
function run_disp_calc_mpi(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0)
    rank = mpi_rank()
    size = mpi_size()
    
    # Distribute displacements across MPI ranks
    local_disps = rank+1:size:Ndispalce
    
    if is_master()
        println("Running scf_0:")
    end
    
    # Only master runs scf_0
    if is_master()
        if isfile(path_to_in*"scf_0/"*"run.sh")
            run_scf_cluster(path_to_in*"scf_0/")
        else
            run_scf(path_to_in*"scf_0/", mpi_ranks)
        end
    end
    
    mpi_barrier()
    
    # Distribute displacement calculations
    for i_disp in local_disps
        if is_master()
            println("Rank $rank: Running displacement # $i_disp:")
        end
        
        dir_name = "group_"*string(i_disp)*"/"
        if isfile(path_to_in*dir_name*"run.sh")
            run_scf_cluster(path_to_in*dir_name)
        else
            run_scf(path_to_in*dir_name, mpi_ranks)
        end
    end
    
    mpi_barrier()
    return true
end
