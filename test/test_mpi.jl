using Test
using ElectronPhonon

@testset "MPI Functionality" begin
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
    
    wf_orig = wf_from_G(miller, evc, Nxyz)
    wf_opt = wf_from_G_optimized(miller, evc, Nxyz)
    
    @test size(wf_orig) == size(wf_opt)
    @test isapprox(wf_orig, wf_opt, rtol=1e-10)
    
    # Test optimized braket calculation
    bra = rand(ComplexF64, 10)
    ket = rand(ComplexF64, 10)
    
    result_orig = calculate_braket(bra, ket)
    result_opt = calculate_braket_optimized(bra, ket)
    
    @test isapprox(result_orig, result_opt, rtol=1e-10)
    
    # Test optimized phase determination
    q_point = [0.1, 0.2, 0.3]
    phase_orig = determine_phase(q_point, Nxyz)
    phase_opt = determine_phase_optimized(q_point, Nxyz)
    
    @test size(phase_orig) == size(phase_opt)
    @test isapprox(phase_orig, phase_opt, rtol=1e-10)
end

@testset "Memory Management" begin
    # Test FFT buffer management
    initial_buffers = length(FFT_BUFFERS)
    
    # Use some buffers
    get_fft_buffer([4, 4, 4])
    get_fft_buffer([8, 8, 8])
    
    @test length(FFT_BUFFERS) > initial_buffers
    
    # Clear buffers
    clear_fft_buffers()
    @test length(FFT_BUFFERS) == 0
end

@testset "I/O Optimizations" begin
    # Test optimized wave function saving
    test_data = Dict("wfc1" => rand(ComplexF64, 4, 4, 4))
    test_path = "./test_output"
    
    # Create test directory
    mkpath(test_path)
    
    # Test saving with compression
    save_wave_functions_optimized(test_path, test_data, 1; compress=true)
    @test isfile("$test_path/wfc_list_1.jld2")
    
    # Clean up
    rm(test_path, recursive=true)
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    @testset "All Tests" begin
        include("test_mpi.jl")
    end
end
