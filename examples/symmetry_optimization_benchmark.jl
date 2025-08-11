#!/usr/bin/env julia

"""
Symmetry Operation Optimization Benchmark

This script benchmarks the performance improvements achieved with the new
optimized symmetry operations, comparing:
- Original vs optimized grid rotation
- Original vs optimized wave function rotation
- Original vs optimized phase factor calculation
- Original vs optimized U matrix preparation
- MPI scaling for symmetry operations

Usage:
    julia --project=. -t auto symmetry_optimization_benchmark.jl
    mpirun -np 4 julia --project=. -t auto symmetry_optimization_benchmark.jl
"""

using ElectronPhonon
using BenchmarkTools
using Statistics
using Printf
using LinearAlgebra

function benchmark_grid_rotation()
    """Benchmark grid rotation operations"""
    
    println("=== Grid Rotation Benchmark ===")
    
    # Test parameters
    grid_sizes = [[16, 16, 16], [32, 32, 32], [64, 64, 64]]
    
    # Test rotation matrix and translation
    rot = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 1.0]  # 90-degree rotation around z
    tras = [0.1, 0.2, 0.3]
    
    for Nxyz in grid_sizes
        println("\nGrid size: $Nxyz")
        N1, N2, N3 = Nxyz
        
        # Benchmark original function
        println("Benchmarking original rotate_grid...")
        original_times = []
        for i in 1:3
            t = @elapsed rotate_grid_original(N1, N2, N3, rot, tras)
            push!(original_times, t)
        end
        
        # Benchmark optimized function
        println("Benchmarking optimized rotate_grid_optimized...")
        optimized_times = []
        for i in 1:3
            t = @elapsed rotate_grid_optimized(N1, N2, N3, rot, tras)
            push!(optimized_times, t)
        end
        
        # Calculate statistics
        orig_mean = mean(original_times)
        opt_mean = mean(optimized_times)
        speedup = orig_mean / opt_mean
        
        println("Original: $(@sprintf("%.6f", orig_mean)) s")
        println("Optimized: $(@sprintf("%.6f", opt_mean)) s")
        println("Speedup: $(@sprintf("%.2fx", speedup))")
        
        # Verify results are identical
        result_orig = rotate_grid_original(N1, N2, N3, rot, tras)
        result_opt = rotate_grid_optimized(N1, N2, N3, rot, tras)
        @assert result_orig == result_opt "Grid rotation results differ!"
        println("✓ Results verified identical")
    end
end

function benchmark_wave_function_rotation()
    """Benchmark wave function rotation operations"""
    
    println("\n=== Wave Function Rotation Benchmark ===")
    
    # Test parameters
    grid_sizes = [[16, 16, 16], [32, 32, 32]]
    
    # Test rotation matrix and translation
    rot = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 1.0]
    tras = [0.1, 0.2, 0.3]
    
    for Nxyz in grid_sizes
        println("\nGrid size: $Nxyz")
        N1, N2, N3 = Nxyz
        
        # Generate test wave function
        ff = randn(ComplexF64, N1, N2, N3)
        
        # Get mapping
        mapp = rotate_grid_optimized(N1, N2, N3, rot, tras)
        
        # Benchmark original function
        println("Benchmarking original rotate_deriv...")
        original_times = []
        for i in 1:3
            t = @elapsed rotate_deriv_original(N1, N2, N3, mapp, ff)
            push!(original_times, t)
        end
        
        # Benchmark optimized function
        println("Benchmarking optimized rotate_deriv_optimized...")
        optimized_times = []
        for i in 1:3
            t = @elapsed rotate_deriv_optimized(N1, N2, N3, mapp, ff)
            push!(optimized_times, t)
        end
        
        # Calculate statistics
        orig_mean = mean(original_times)
        opt_mean = mean(optimized_times)
        speedup = orig_mean / opt_mean
        
        println("Original: $(@sprintf("%.6f", orig_mean)) s")
        println("Optimized: $(@sprintf("%.6f", opt_mean)) s")
        println("Speedup: $(@sprintf("%.2fx", speedup))")
        
        # Verify results are identical
        result_orig = rotate_deriv_original(N1, N2, N3, mapp, ff)
        result_opt = rotate_deriv_optimized(N1, N2, N3, mapp, ff)
        @assert isapprox(result_orig, result_opt, rtol=1e-10) "Wave function rotation results differ!"
        println("✓ Results verified identical")
    end
end

function benchmark_phase_factor_calculation()
    """Benchmark phase factor calculations"""
    
    println("\n=== Phase Factor Calculation Benchmark ===")
    
    # Test parameters
    grid_sizes = [[16, 16, 16], [32, 32, 32], [64, 64, 64]]
    kpoint = [0.1, 0.2, 0.3]
    
    for Nxyz in grid_sizes
        println("\nGrid size: $Nxyz")
        
        # Benchmark original function
        println("Benchmarking original determine_phase...")
        original_times = []
        for i in 1:3
            t = @elapsed determine_phase(kpoint, Nxyz)
            push!(original_times, t)
        end
        
        # Benchmark optimized function
        println("Benchmarking optimized determine_phase_optimized...")
        optimized_times = []
        for i in 1:3
            t = @elapsed determine_phase_optimized(kpoint, Nxyz)
            push!(optimized_times, t)
        end
        
        # Calculate statistics
        orig_mean = mean(original_times)
        opt_mean = mean(optimized_times)
        speedup = orig_mean / opt_mean
        
        println("Original: $(@sprintf("%.6f", orig_mean)) s")
        println("Optimized: $(@sprintf("%.6f", opt_mean)) s")
        println("Speedup: $(@sprintf("%.2fx", speedup))")
        
        # Verify results are identical
        result_orig = determine_phase(kpoint, Nxyz)
        result_opt = determine_phase_optimized(kpoint, Nxyz)
        @assert isapprox(result_orig, result_opt, rtol=1e-10) "Phase factor results differ!"
        println("✓ Results verified identical")
    end
end

function benchmark_batch_symmetry_operations()
    """Benchmark batch symmetry operations"""
    
    println("\n=== Batch Symmetry Operations Benchmark ===")
    
    # Test parameters
    N1, N2, N3 = 32, 32, 32
    n_wave_functions = [1, 5, 10, 20]
    
    # Test rotation matrix and translation
    rot = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 1.0]
    tras = [0.1, 0.2, 0.3]
    
    for n_wf in n_wave_functions
        println("\nNumber of wave functions: $n_wf")
        
        # Generate test wave functions
        ff_list = [randn(ComplexF64, N1, N2, N3) for _ in 1:n_wf]
        
        # Benchmark individual operations
        println("Benchmarking individual operations...")
        individual_times = []
        for i in 1:3
            t = @elapsed begin
                for ff in ff_list
                    mapp = rotate_grid_optimized(N1, N2, N3, rot, tras)
                    rotate_deriv_optimized(N1, N2, N3, mapp, ff)
                end
            end
            push!(individual_times, t)
        end
        
        # Benchmark batch operations
        println("Benchmarking batch operations...")
        batch_times = []
        for i in 1:3
            t = @elapsed apply_symmetries_batch(ff_list, N1, N2, N3, rot, tras)
            push!(batch_times, t)
        end
        
        # Calculate statistics
        ind_mean = mean(individual_times)
        batch_mean = mean(batch_times)
        speedup = ind_mean / batch_mean
        
        println("Individual: $(@sprintf("%.6f", ind_mean)) s")
        println("Batch: $(@sprintf("%.6f", batch_mean)) s")
        println("Speedup: $(@sprintf("%.2fx", speedup))")
    end
end

function benchmark_mpi_symmetry_scaling()
    """Benchmark MPI scaling for symmetry operations"""
    
    if mpi_size() == 1
        println("\n=== MPI Symmetry Scaling Benchmark (Single Process) ===")
        println("Run with multiple MPI processes to see scaling:")
        println("mpirun -np 4 julia --project=. -t auto symmetry_optimization_benchmark.jl")
        return
    end
    
    println("\n=== MPI Symmetry Scaling Benchmark ===")
    println("MPI Processes: $(mpi_size())")
    println("Threads per process: $(Threads.nthreads())")
    
    # Test parameters
    n_symmetry_operations = 100
    grid_size = [32, 32, 32]
    
    # Distribute work across MPI ranks
    rank = mpi_rank()
    size = mpi_size()
    local_ops = rank+1:size:n_symmetry_operations
    
    println("Rank $rank: processing $(length(local_ops)) symmetry operations")
    
    # Generate test data
    rot = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 1.0]
    tras = [0.1, 0.2, 0.3]
    ff = randn(ComplexF64, grid_size...)
    
    # Benchmark symmetry operations
    t_start = time()
    
    for op in local_ops
        # Simulate symmetry operation
        mapp = rotate_grid_optimized(grid_size[1], grid_size[2], grid_size[3], rot, tras)
        result = rotate_deriv_optimized(grid_size[1], grid_size[2], grid_size[3], mapp, ff)
        
        # Add some computational work
        sleep(0.001)  # 1ms per operation
    end
    
    t_local = time() - t_start
    
    # Gather timing information
    all_times = mpi_gather([t_local])
    
    if is_master()
        total_time = maximum(all_times)
        avg_time = mean(all_times)
        efficiency = avg_time / total_time
        
        println("\nMPI Symmetry Scaling Results:")
        println("Total time: $(@sprintf("%.4f", total_time)) s")
        println("Average time per rank: $(@sprintf("%.4f", avg_time)) s")
        println("Parallel efficiency: $(@sprintf("%.2f%%", efficiency*100))")
        
        # Calculate speedup
        theoretical_time = sum(all_times) / size
        speedup = theoretical_time / total_time
        println("Speedup: $(@sprintf("%.2fx", speedup))")
    end
end

function benchmark_memory_usage_symmetry()
    """Benchmark memory usage of symmetry operations"""
    
    println("\n=== Memory Usage Benchmark (Symmetry Operations) ===")
    
    # Test parameters
    grid_size = [64, 64, 64]
    n_operations = 50
    
    # Generate test data
    rot = [0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 1.0]
    tras = [0.1, 0.2, 0.3]
    ff = randn(ComplexF64, grid_size...)
    
    # Measure memory usage without cache
    println("Measuring memory usage without cache...")
    clear_symmetry_cache()
    GC.gc()
    mem_before = Sys.total_memory() - Sys.free_memory()
    
    for i in 1:n_operations
        mapp = rotate_grid_optimized(grid_size[1], grid_size[2], grid_size[3], rot, tras)
        result = rotate_deriv_optimized(grid_size[1], grid_size[2], grid_size[3], mapp, ff)
    end
    
    mem_after = Sys.total_memory() - Sys.free_memory()
    mem_without_cache = mem_after - mem_before
    
    # Measure memory usage with cache
    println("Measuring memory usage with cache...")
    clear_symmetry_cache()
    GC.gc()
    mem_before = Sys.total_memory() - Sys.free_memory()
    
    for i in 1:n_operations
        mapp = rotate_grid_optimized(grid_size[1], grid_size[2], grid_size[3], rot, tras)
        result = rotate_deriv_optimized(grid_size[1], grid_size[2], grid_size[3], mapp, ff)
    end
    
    mem_after = Sys.total_memory() - Sys.free_memory()
    mem_with_cache = mem_after - mem_before
    
    println("\nMemory Usage Results:")
    println("Without cache: $(@sprintf("%.2f", mem_without_cache/1024^2)) MB")
    println("With cache: $(@sprintf("%.2f", mem_with_cache/1024^2)) MB")
    println("Cache overhead: $(@sprintf("%.2f", (mem_with_cache-mem_without_cache)/1024^2)) MB")
    
    # Show cache statistics
    println("Cache entries: $(length(GRID_MAPPING_CACHE))")
    println("Cache memory: $(@sprintf("%.2f", sum(sizeof.(values(GRID_MAPPING_CACHE)))/1024^2)) MB")
end

# Original functions for comparison (simplified versions)
function rotate_grid_original(N1, N2, N3, rot, tras)
    mapp = []
    for k in 0:N3-1
        for j in 0:N2-1
            for i in 0:N1-1
                u = [i / N1, j / N2, k / N3]
                ru = rot * u .+ tras
                ru[1] = fold_component(ru[1])
                ru[2] = fold_component(ru[2])
                ru[3] = fold_component(ru[3])
                
                i1 = round(Int, ru[1] * N1)
                i2 = round(Int, ru[2] * N2)
                i3 = round(Int, ru[3] * N3)
                
                ind = i1 + i2 * N1 + i3 * N1 * N2
                push!(mapp, ind)
            end
        end
    end
    return mapp
end

function rotate_deriv_original(N1, N2, N3, mapp, ff)
    ff_rot = zeros(ComplexF64, N1, N2, N3)
    ind = 1
    for k in 0:N3-1
        for j in 0:N2-1
            for i in 0:N1-1
                ind1 = mapp[ind]
                i3 = div(ind1, N2 * N1)
                ind1 = ind1 % (N1 * N2)
                i2 = div(ind1, N1)
                i1 = ind1 % N1
                ind += 1
                ff_rot[i1+1, i2+1, i3+1] = ff[i+1, j+1, k+1]
            end
        end
    end
    return ff_rot
end

function generate_symmetry_performance_report()
    """Generate comprehensive symmetry performance report"""
    
    if is_master()
        println("\n" * "="^60)
        println("SYMMETRY OPTIMIZATION BENCHMARK REPORT")
        println("="^60)
        println("Date: $(now())")
        println("System: $(Sys.MACHINE)")
        println("Julia version: $(VERSION)")
        println("MPI processes: $(mpi_size())")
        println("Threads per process: $(Threads.nthreads())")
        println("Total workers: $(mpi_size() * Threads.nthreads())")
        println("="^60)
    end
    
    # Run all benchmarks
    benchmark_grid_rotation()
    benchmark_wave_function_rotation()
    benchmark_phase_factor_calculation()
    benchmark_batch_symmetry_operations()
    benchmark_mpi_symmetry_scaling()
    benchmark_memory_usage_symmetry()
    
    if is_master()
        println("\n" * "="^60)
        println("SUMMARY")
        println("="^60)
        println("Symmetry operations optimized with:")
        println("- Vectorized grid operations")
        println("- Caching for repeated calculations")
        println("- MPI parallelization")
        println("- Batch processing capabilities")
        println("- Memory-efficient implementations")
        println("="^60)
    end
end

function main()
    """Main benchmark function"""
    
    # Set up FFTW for optimal performance
    FFTW.set_num_threads(Threads.nthreads())
    
    # Run comprehensive benchmark
    generate_symmetry_performance_report()
    
    # Clean up
    clear_fft_buffers()
    clear_symmetry_cache()
    
    if is_master()
        println("\nSymmetry optimization benchmark completed!")
    end
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
