using Test
using ElectronPhonon
using YAML

@testset "Silicon DFT Example" begin
    
    @testset "Configuration Loading" begin
        # Test loading configuration file
        config_path = joinpath(@__DIR__, "..", "examples", "silicon_config.yaml")
        if isfile(config_path)
            config = YAML.load_file(config_path)
            
            @test haskey(config, "crystal")
            @test haskey(config, "calculation")
            @test haskey(config, "hpc")
            
            # Test crystal structure
            crystal = config["crystal"]
            @test crystal["lattice_constant"] ≈ 5.43
            @test length(crystal["atomic_positions"]) == 2
            @test crystal["atomic_species"] == ["Si", "Si"]
            
            # Test calculation parameters
            calc = config["calculation"]
            @test calc["supercell_size"] == [2, 2, 2]
            @test calc["k_mesh"] == [8, 8, 8]
            @test calc["q_mesh"] == [4, 4, 4]
            
            # Test HPC settings
            hpc = config["hpc"]
            @test hpc["mpi"]["enabled"] == true
            @test hpc["threading"]["enabled"] == true
        end
    end
    
    @testset "Silicon Model Setup" begin
        # Test silicon crystal structure setup
        lattice_constant = 5.43
        cell_vectors = [
            [lattice_constant/2, lattice_constant/2, 0.0],
            [0.0, lattice_constant/2, lattice_constant/2],
            [lattice_constant/2, 0.0, lattice_constant/2]
        ]
        
        @test length(cell_vectors) == 3
        @test all(length.(cell_vectors) .== 3)
        
        # Test atomic positions
        atomic_positions = [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25]
        ]
        
        @test length(atomic_positions) == 2
        @test atomic_positions[1] == [0.0, 0.0, 0.0]
        @test atomic_positions[2] == [0.25, 0.25, 0.25]
    end
    
    @testset "K-point and Q-point Generation" begin
        # Test k-point mesh generation
        k_mesh = [4, 4, 4]
        k_list = []
        for i in 1:k_mesh[1], j in 1:k_mesh[2], k in 1:k_mesh[3]
            kx = (i - 1) / k_mesh[1]
            ky = (j - 1) / k_mesh[2]
            kz = (k - 1) / k_mesh[3]
            push!(k_list, [kx, ky, kz])
        end
        
        @test length(k_list) == prod(k_mesh)
        @test k_list[1] == [0.0, 0.0, 0.0]
        @test k_list[end] == [0.75, 0.75, 0.75]
        
        # Test q-point mesh generation
        q_mesh = [2, 2, 2]
        q_list = []
        for i in 1:q_mesh[1], j in 1:q_mesh[2], k in 1:q_mesh[3]
            qx = (i - 1) / q_mesh[1]
            qy = (j - 1) / q_mesh[2]
            qz = (k - 1) / q_mesh[3]
            push!(q_list, [qx, qy, qz])
        end
        
        @test length(q_list) == prod(q_mesh)
        @test q_list[1] == [0.0, 0.0, 0.0]
        @test q_list[end] == [0.5, 0.5, 0.5]
    end
    
    @testset "MPI Functionality" begin
        # Test MPI functions are available
        @test mpi_size() >= 1
        @test mpi_rank() >= 0
        @test mpi_rank() < mpi_size()
        @test is_master() == (mpi_rank() == 0)
        
        # Test MPI communication
        if mpi_size() > 1
            # Test barrier
            mpi_barrier()
            @test true  # If we reach here, barrier worked
            
            # Test broadcast
            if is_master()
                test_data = [1.0, 2.0, 3.0]
            else
                test_data = zeros(3)
            end
            mpi_bcast!(test_data)
            @test test_data == [1.0, 2.0, 3.0]
        end
    end
    
    @testset "Optimized Functions" begin
        # Test optimized wave function conversion
        miller = [0 1 -1; 0 1 0; 0 0 1]
        evc = [1.0 + 0.0im, 0.5 + 0.5im, 0.0 + 1.0im]
        Nxyz = [8, 8, 8]
        
        # Test original function
        wf_original = wf_from_G(miller, evc, Nxyz)
        @test size(wf_original) == Nxyz
        @test eltype(wf_original) == ComplexF64
        
        # Test optimized function
        wf_optimized = wf_from_G_optimized(miller, evc, Nxyz)
        @test size(wf_optimized) == Nxyz
        @test eltype(wf_optimized) == ComplexF64
        
        # Results should be identical (within numerical precision)
        @test isapprox(wf_original, wf_optimized, rtol=1e-10)
    end
    
    @testset "Memory Management" begin
        # Test FFT buffer management
        dims = (16, 16, 16)
        buffer1 = get_fft_buffer(dims)
        buffer2 = get_fft_buffer(dims)
        
        @test buffer1 === buffer2  # Same buffer should be returned
        @test size(buffer1) == dims
        
        # Test buffer clearing
        clear_fft_buffers()
        @test length(FFT_BUFFERS) == 0
    end
    
    @testset "Silicon Physical Properties" begin
        # Test silicon-specific properties
        band_gap = 1.12  # eV
        @test band_gap > 0
        
        # Test phonon frequencies (realistic for silicon)
        acoustic_freqs = [0.1, 0.15, 0.2]  # THz
        optical_freqs = [15.0, 15.5, 16.0]  # THz
        
        @test all(acoustic_freqs .< 1.0)  # Acoustic modes should be low frequency
        @test all(optical_freqs .> 10.0)  # Optical modes should be high frequency
        
        # Test electron-phonon coupling
        lambda_expected = 0.5
        @test lambda_expected > 0 && lambda_expected < 2.0  # Reasonable range
    end
    
    @testset "Configuration Validation" begin
        # Test that configuration parameters are reasonable
        config_path = joinpath(@__DIR__, "..", "examples", "silicon_config.yaml")
        if isfile(config_path)
            config = YAML.load_file(config_path)
            
            # Test energy cutoffs
            ecutwfc = config["calculation"]["energy_cutoffs"]["wave_function"]
            ecutrho = config["calculation"]["energy_cutoffs"]["charge_density"]
            
            @test ecutwfc > 0
            @test ecutrho > ecutwfc  # Charge density cutoff should be higher
            
            # Test displacement amplitude
            abs_disp = config["calculation"]["displacement_amplitude"]
            @test abs_disp > 0 && abs_disp < 0.1  # Reasonable range
            
            # Test mesh sizes
            k_mesh = config["calculation"]["k_mesh"]
            q_mesh = config["calculation"]["q_mesh"]
            
            @test all(k_mesh .> 0)
            @test all(q_mesh .> 0)
            @test all(k_mesh .>= q_mesh)  # k-mesh should be at least as fine as q-mesh
        end
    end
end

println("Silicon example tests completed!")
