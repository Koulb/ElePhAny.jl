#!/usr/bin/env julia

"""
MPI Example for ElectronPhonon.jl

This example demonstrates how to use the MPI parallelization features
for large-scale electron-phonon calculations.

Usage:
    mpirun -np 4 julia --project=. mpi_example.jl
"""

using ElectronPhonon

function main()
    # Check if we're running with MPI
    if mpi_size() == 1
        @warn "Running with only 1 MPI process. For better performance, use: mpirun -np 4 julia --project=. mpi_example.jl"
    end
    
    if is_master()
        println("=== ElectronPhonon.jl MPI Example ===")
        println("MPI Processes: $(mpi_size())")
        println("Threads per process: $(Threads.nthreads())")
        println("Total parallel workers: $(mpi_size() * Threads.nthreads())")
    end
    
    # Example parameters (modify as needed)
    path_to_calc = "./example_calculation/"
    abs_disp = 1e-3
    sc_size = [2, 2, 2]
    k_mesh = [1, 1, 1]
    
    # Create model
    model = create_model(
        path_to_calc = path_to_calc,
        abs_disp = abs_disp,
        sc_size = sc_size,
        k_mesh = k_mesh,
        use_symm = true
    )
    
    if is_master()
        println("Model created with:")
        println("  Supercell size: $(model.sc_size)")
        println("  K-mesh: $(model.k_mesh)")
        println("  Displacements: $(model.Ndispalce)")
    end
    
    # Prepare model (only master does this)
    if is_master()
        println("Preparing model...")
        prepare_model(model)
    end
    mpi_barrier()
    
    # Load data
    if is_master()
        println("Loading electrons and phonons...")
    end
    electrons = load_electrons(model)
    phonons = load_phonons(model)
    mpi_barrier()
    
    # Example: Calculate electron-phonon matrix elements for a few k-points
    if is_master()
        println("Calculating electron-phonon matrix elements...")
    end
    
    # Distribute k-point calculations across MPI ranks
    total_kpoints = prod(sc_size)
    local_kpoints = mpi_rank() + 1:mpi_size():total_kpoints
    
    for ik in local_kpoints
        for iq in 1:total_kpoints
            if is_master()
                println("Rank $(mpi_rank()): Processing (ik=$ik, iq=$iq)")
            end
            
            # Use MPI-optimized calculation
            result = electron_phonon_mpi(
                model.path_to_calc*"displacements/",
                model.abs_disp,
                length(model.unitcell[:symbols]),
                ik, iq,
                model.sc_size,
                model.k_mesh,
                electrons.ϵkᵤ_list,
                electrons.ϵₚ_list,
                electrons.ϵₚₘ_list,
                electrons.k_list,
                electrons.U_list,
                electrons.V_list,
                phonons.M_phonon,
                phonons.ωₐᵣᵣ_ₗᵢₛₜ,
                phonons.εₐᵣᵣ_ₗᵢₛₜ,
                phonons.mₐᵣᵣ;
                save_epw = false,
                save_qeraman = false,
                phonons_dfpt = true
            )
            
            if is_master() && result !== nothing
                println("Rank $(mpi_rank()): Completed (ik=$ik, iq=$iq)")
            end
        end
    end
    
    mpi_barrier()
    
    # Example: Optimized wave function preparation
    if is_master()
        println("Preparing wave functions with hybrid MPI + threading...")
    end
    
    prepare_wave_functions_hybrid(
        model.path_to_calc*"displacements/",
        model.sc_size;
        k_mesh = model.k_mesh
    )
    
    # Example: Memory-optimized processing
    if is_master()
        println("Demonstrating memory-optimized processing...")
    end
    
    # Process wave functions in streaming mode
    for ik in local_kpoints
        if is_master()
            println("Rank $(mpi_rank()): Streaming processing for k-point $ik")
        end
        
        process_wave_functions_streaming(
            model.path_to_calc*"displacements/",
            ik;
            batch_size = 5
        )
    end
    
    # Clear FFT buffers
    clear_fft_buffers()
    
    mpi_barrier()
    
    if is_master()
        println("=== Example completed successfully ===")
        println("Check the output files in $(model.path_to_calc)displacements/")
    end
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
