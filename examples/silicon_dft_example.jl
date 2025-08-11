#!/usr/bin/env julia

"""
Silicon (Si) DFT Example with Optimized Routines

This example demonstrates a complete electron-phonon calculation for silicon
using the new HPC-optimized routines, including:
- MPI parallelization for large-scale calculations
- Optimized wave function transformations
- Memory-efficient processing
- Hybrid MPI + threading approach

Usage:
    # Single node with threading
    julia --project=. -t auto silicon_dft_example.jl
    
    # Multi-node with MPI
    mpirun -np 4 julia --project=. -t auto silicon_dft_example.jl
"""

using ElectronPhonon
using LinearAlgebra
using Printf

function setup_silicon_model()
    """Setup silicon crystal structure and calculation parameters"""
    
    if is_master()
        println("=== Silicon DFT Electron-Phonon Calculation ===")
        println("Setting up silicon crystal structure...")
    end
    
    # Silicon crystal structure (diamond structure)
    # Lattice constant: 5.43 Å
    lattice_constant = 5.43  # Å
    
    # Primitive cell vectors (in Å)
    cell_vectors = [
        [lattice_constant/2, lattice_constant/2, 0.0],
        [0.0, lattice_constant/2, lattice_constant/2],
        [lattice_constant/2, 0.0, lattice_constant/2]
    ]
    
    # Atomic positions (in fractional coordinates)
    # Two Si atoms in primitive cell
    atomic_positions = [
        [0.0, 0.0, 0.0],      # Si at origin
        [0.25, 0.25, 0.25]    # Si at (1/4, 1/4, 1/4)
    ]
    
    # Atomic species
    atomic_species = ["Si", "Si"]
    
    # Calculation parameters
    sc_size = [2, 2, 2]  # 2x2x2 supercell
    k_mesh = [8, 8, 8]   # 8x8x8 k-point mesh
    q_mesh = [4, 4, 4]   # 4x4x4 q-point mesh for phonons
    
    # Energy cutoffs
    ecutwfc = 30.0  # Wave function cutoff (Ry)
    ecutrho = 120.0 # Charge density cutoff (Ry)
    
    # Displacement amplitude for frozen-phonon
    abs_disp = 0.01  # Å
    
    # Create model structure
    model = Dict(
        "lattice_constant" => lattice_constant,
        "cell_vectors" => cell_vectors,
        "atomic_positions" => atomic_positions,
        "atomic_species" => atomic_species,
        "sc_size" => sc_size,
        "k_mesh" => k_mesh,
        "q_mesh" => q_mesh,
        "ecutwfc" => ecutwfc,
        "ecutrho" => ecutrho,
        "abs_disp" => abs_disp,
        "pseudo_dir" => "./pseudos",
        "outdir" => "./si_calculation"
    )
    
    return model
end

function create_qe_input(model)
    """Create Quantum ESPRESSO input files for silicon"""
    
    if is_master()
        println("Creating Quantum ESPRESSO input files...")
    end
    
    # Create output directory
    mkpath(model["outdir"])
    
    # SCF calculation input
    scf_input = """
&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = 'si'
    outdir = '$(model["outdir"])'
    pseudo_dir = '$(model["pseudo_dir"])'
    verbosity = 'high'
/
&SYSTEM
    ibrav = 0
    nat = 2
    ntyp = 1
    ecutwfc = $(model["ecutwfc"])
    ecutrho = $(model["ecutrho"])
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.01
    nbnd = 20
/
&ELECTRONS
    diagonalization = 'david'
    mixing_mode = 'plain'
    mixing_beta = 0.7
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS {crystal}
Si  0.000000000  0.000000000  0.000000000
Si  0.250000000  0.250000000  0.250000000
CELL_PARAMETERS {angstrom}
$(model["cell_vectors"][1][1])  $(model["cell_vectors"][1][2])  $(model["cell_vectors"][1][3])
$(model["cell_vectors"][2][1])  $(model["cell_vectors"][2][2])  $(model["cell_vectors"][2][3])
$(model["cell_vectors"][3][1])  $(model["cell_vectors"][3][2])  $(model["cell_vectors"][3][3])
K_POINTS {automatic}
$(model["k_mesh"][1]) $(model["k_mesh"][2]) $(model["k_mesh"][3]) 0 0 0
"""
    
    # Write SCF input
    write("$(model["outdir"])/si.scf.in", scf_input)
    
    # Phonon calculation input
    phonon_input = """
&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = 'si'
    outdir = '$(model["outdir"])'
    pseudo_dir = '$(model["pseudo_dir"])'
    verbosity = 'high'
/
&SYSTEM
    ibrav = 0
    nat = 16
    ntyp = 1
    ecutwfc = $(model["ecutwfc"])
    ecutrho = $(model["ecutrho"])
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.01
    nbnd = 80
/
&ELECTRONS
    diagonalization = 'david'
    mixing_mode = 'plain'
    mixing_beta = 0.7
    conv_thr = 1.0d-8
/
ATOMIC_SPECIES
Si  28.0855  Si.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS {crystal}
Si  0.000000000  0.000000000  0.000000000
Si  0.250000000  0.250000000  0.250000000
Si  0.500000000  0.000000000  0.000000000
Si  0.750000000  0.250000000  0.250000000
Si  0.000000000  0.500000000  0.000000000
Si  0.250000000  0.750000000  0.250000000
Si  0.500000000  0.500000000  0.000000000
Si  0.750000000  0.750000000  0.250000000
Si  0.000000000  0.000000000  0.500000000
Si  0.250000000  0.250000000  0.750000000
Si  0.500000000  0.000000000  0.500000000
Si  0.750000000  0.250000000  0.750000000
Si  0.000000000  0.500000000  0.500000000
Si  0.250000000  0.750000000  0.750000000
Si  0.500000000  0.500000000  0.500000000
Si  0.750000000  0.750000000  0.750000000
CELL_PARAMETERS {angstrom}
$(model["cell_vectors"][1][1]*2)  $(model["cell_vectors"][1][2]*2)  $(model["cell_vectors"][1][3]*2)
$(model["cell_vectors"][2][1]*2)  $(model["cell_vectors"][2][2]*2)  $(model["cell_vectors"][2][3]*2)
$(model["cell_vectors"][3][1]*2)  $(model["cell_vectors"][3][2]*2)  $(model["cell_vectors"][3][3]*2)
K_POINTS {automatic}
$(model["k_mesh"][1]) $(model["k_mesh"][2]) $(model["k_mesh"][3]) 0 0 0
"""
    
    # Write phonon input
    write("$(model["outdir"])/si.phonon.in", phonon_input)
    
    if is_master()
        println("Input files created in $(model["outdir"])/")
    end
end

function run_silicon_calculation(model)
    """Run the complete silicon electron-phonon calculation"""
    
    if is_master()
        println("Starting silicon electron-phonon calculation...")
        println("MPI Processes: $(mpi_size())")
        println("Threads per process: $(Threads.nthreads())")
        println("Total parallel workers: $(mpi_size() * Threads.nthreads())")
    end
    
    # Step 1: Prepare model and create input files
    if is_master()
        create_qe_input(model)
    end
    mpi_barrier()
    
    # Step 2: Run SCF calculation (only on master)
    if is_master()
        println("Running SCF calculation...")
        # In a real calculation, you would run QE here
        # run(`pw.x -in $(model["outdir"])/si.scf.in > $(model["outdir"])/si.scf.out`)
        println("SCF calculation completed (simulated)")
    end
    mpi_barrier()
    
    # Step 3: Generate k-point and q-point lists
    if is_master()
        println("Generating k-point and q-point meshes...")
    end
    
    # Generate k-point mesh
    k_list = generate_k_mesh(model["k_mesh"])
    q_list = generate_q_mesh(model["q_mesh"])
    
    if is_master()
        println("Generated $(length(k_list)) k-points and $(length(q_list)) q-points")
    end
    
    # Step 4: Calculate electron energies and wave functions
    if is_master()
        println("Calculating electron properties...")
    end
    
    # Simulate electron energies (in a real calculation, these would come from QE)
    nbands = 20
    ϵkᵤ_list = []
    U_list = []
    
    for ik in 1:length(k_list)
        # Simulate band energies (realistic for silicon)
        ϵk = zeros(nbands)
        for ib in 1:nbands
            if ib <= 4  # Valence bands
                ϵk[ib] = -2.0 - 0.1 * (ib - 1)  # eV
            else  # Conduction bands
                ϵk[ib] = 1.0 + 0.2 * (ib - 5)   # eV
            end
        end
        push!(ϵkᵤ_list, ϵk)
        
        # Simulate wave function coefficients
        U = [randn(ComplexF64) for _ in 1:nbands]
        push!(U_list, U)
    end
    
    # Step 5: Calculate phonon properties
    if is_master()
        println("Calculating phonon properties...")
    end
    
    # Simulate phonon frequencies (realistic for silicon)
    Nat = 2  # atoms in primitive cell
    nmodes = 3 * Nat
    ωₐᵣᵣ_ₗᵢₛₜ = []
    εₐᵣᵣ_ₗᵢₛₜ = []
    
    for iq in 1:length(q_list)
        ωₐᵣᵣ = zeros(nmodes)
        εₐᵣᵣ = zeros(ComplexF64, nmodes, 3, Nat)
        
        # Simulate phonon frequencies (acoustic and optical modes)
        for imode in 1:nmodes
            if imode <= 3  # Acoustic modes
                ωₐᵣᵣ[imode] = 0.1 + 0.05 * imode  # THz
            else  # Optical modes
                ωₐᵣᵣ[imode] = 15.0 + 0.5 * (imode - 4)  # THz
            end
            
            # Simulate polarization vectors
            for α in 1:3, iat in 1:Nat
                εₐᵣᵣ[imode, α, iat] = randn(ComplexF64)
            end
        end
        
        push!(ωₐᵣᵣ_ₗᵢₛₜ, ωₐᵣᵣ)
        push!(εₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ)
    end
    
    # Step 6: Run electron-phonon calculation with MPI
    if is_master()
        println("Running electron-phonon matrix element calculations...")
    end
    
    # Select specific k-points and q-points for demonstration
    ik_selected = 1
    iq_selected = 1
    
    # Run MPI-optimized electron-phonon calculation
    results = electron_phonon_mpi(
        model["outdir"],
        model["abs_disp"],
        Nat,
        ik_selected,
        iq_selected,
        model["sc_size"],
        model["k_mesh"],
        ϵkᵤ_list,
        ϵkᵤ_list,  # Use same for simplicity
        ϵkᵤ_list,  # Use same for simplicity
        k_list,
        U_list,
        U_list,    # Use same for simplicity
        zeros(ComplexF64, nbands, nbands, nmodes),  # Placeholder
        ωₐᵣᵣ_ₗᵢₛₜ,
        εₐᵣᵣ_ₗᵢₛₜ,
        ones(Nat);  # Atomic masses
        save_epw = true,
        save_qeraman = false,
        phonons_dfpt = true
    )
    
    # Step 7: Analyze results
    if is_master()
        println("Analysis completed!")
        println("Results saved in $(model["outdir"])/")
        
        # Print some key results
        if haskey(results, "g_epw")
            println("Electron-phonon matrix elements calculated successfully")
            println("Matrix element shape: $(size(results["g_epw"]))")
        end
    end
    
    return results
end

function analyze_silicon_results(results, model)
    """Analyze and visualize silicon electron-phonon results"""
    
    if !is_master()
        return
    end
    
    println("\n=== Silicon Electron-Phonon Analysis ===")
    
    # Calculate average electron-phonon coupling strength
    if haskey(results, "g_epw")
        g_avg = mean(abs.(results["g_epw"]))
        g_max = maximum(abs.(results["g_epw"]))
        println("Average electron-phonon coupling: $(@sprintf("%.4f", g_avg)) eV")
        println("Maximum electron-phonon coupling: $(@sprintf("%.4f", g_max)) eV")
    end
    
    # Calculate Eliashberg function (if available)
    if haskey(results, "α²F")
        println("Eliashberg function calculated")
        println("λ = $(@sprintf("%.4f", results["λ"]))")
    end
    
    # Save results to file
    output_file = "$(model["outdir"])/si_results.txt"
    open(output_file, "w") do io
        println(io, "Silicon Electron-Phonon Results")
        println(io, "===============================")
        println(io, "Lattice constant: $(model["lattice_constant"]) Å")
        println(io, "Supercell size: $(model["sc_size"])")
        println(io, "k-point mesh: $(model["k_mesh"])")
        println(io, "q-point mesh: $(model["q_mesh"])")
        println(io, "Displacement amplitude: $(model["abs_disp"]) Å")
        println(io, "")
        println(io, "Calculation Parameters:")
        println(io, "MPI processes: $(mpi_size())")
        println(io, "Threads per process: $(Threads.nthreads())")
        println(io, "Total parallel workers: $(mpi_size() * Threads.nthreads())")
    end
    
    println("Results saved to $output_file")
end

function generate_k_mesh(k_mesh)
    """Generate k-point mesh"""
    k_list = []
    for i in 1:k_mesh[1], j in 1:k_mesh[2], k in 1:k_mesh[3]
        kx = (i - 1) / k_mesh[1]
        ky = (j - 1) / k_mesh[2]
        kz = (k - 1) / k_mesh[3]
        push!(k_list, [kx, ky, kz])
    end
    return k_list
end

function generate_q_mesh(q_mesh)
    """Generate q-point mesh"""
    q_list = []
    for i in 1:q_mesh[1], j in 1:q_mesh[2], k in 1:q_mesh[3]
        qx = (i - 1) / q_mesh[1]
        qy = (j - 1) / q_mesh[2]
        qz = (k - 1) / q_mesh[3]
        push!(q_list, [qx, qy, qz])
    end
    return q_list
end

function main()
    """Main function for silicon DFT example"""
    
    # Setup silicon model
    model = setup_silicon_model()
    
    # Run calculation
    results = run_silicon_calculation(model)
    
    # Analyze results
    analyze_silicon_results(results, model)
    
    # Clean up FFT buffers
    clear_fft_buffers()
    
    if is_master()
        println("\n=== Silicon DFT Example Completed ===")
        println("Check $(model["outdir"])/ for output files")
    end
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
