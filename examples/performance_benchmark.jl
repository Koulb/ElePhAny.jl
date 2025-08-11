#!/usr/bin/env julia

"""
Performance Benchmark for Silicon DFT Calculations

This script benchmarks the performance improvements achieved with the new
HPC-optimized routines, comparing:
- Original vs optimized wave function transformations
- Single-threaded vs multi-threaded performance
- Single-node vs MPI parallelization
- Memory usage and efficiency

Usage:
    julia --project=. -t auto performance_benchmark.jl
    mpirun -np 4 julia --project=. -t auto performance_benchmark.jl
"""

using ElectronPhonon
using BenchmarkTools
using Statistics
using Printf

function benchmark_wave_function_transformations()
    """Benchmark wave function transformation routines"""
    
    println("=== Wave Function Transformation Benchmark ===")
    
    # Test parameters
    Nxyz = [32, 32, 32]  # Grid size
    nbands = 20
    nk = 10
    
    println("Grid size: $Nxyz")
    println("Number of bands: $nbands")
    println("Number of k-points: $nk")
    
    # Generate test data
    miller_list = []
    evc_list = []
    
    for ik in 1:nk
        # Random Miller indices
        miller = rand(Int32, 3, 100)
        push!(miller_list, miller)
        
        # Random wave function coefficients
        evc = randn(ComplexF64, 100)
        push!(evc_list, evc)
    end
    
    # Benchmark original routine
    println("\nBenchmarking original wf_from_G...")
    original_times = []
    for i in 1:5  # Multiple runs for statistics
        t = @elapsed for ik in 1:nk
            wf_from_G(miller_list[ik], evc_list[ik], Nxyz)
        end
        push!(original_times, t)
    end
    
    # Benchmark optimized routine
    println("Benchmarking optimized wf_from_G_optimized...")
    optimized_times = []
    for i in 1:5
        t = @elapsed for ik in 1:nk
            wf_from_G_optimized(miller_list[ik], evc_list[ik], Nxyz)
        end
        push!(optimized_times, t)
    end
    
    # Calculate statistics
    orig_mean = mean(original_times)
    orig_std = std(original_times)
    opt_mean = mean(optimized_times)
    opt_std = std(optimized_times)
    speedup = orig_mean / opt_mean
    
    println("\nResults:")
    println("Original: $(@sprintf("%.4f ± %.4f", orig_mean, orig_std)) s")
    println("Optimized: $(@sprintf("%.4f ± %.4f", opt_mean, opt_std)) s")
    println("Speedup: $(@sprintf("%.2fx", speedup))")
    
    return speedup
end

function benchmark_memory_usage()
    """Benchmark memory usage of different approaches"""
    
    println("\n=== Memory Usage Benchmark ===")
    
    # Test parameters
    Nxyz = [64, 64, 64]
    nbands = 50
    nk = 20
    
    println("Grid size: $Nxyz")
    println("Number of bands: $nbands")
    println("Number of k-points: $nk")
    
    # Generate test data
    miller_list = []
    evc_list = []
    
    for ik in 1:nk
        miller = rand(Int32, 3, 200)
        evc = randn(ComplexF64, 200)
        push!(miller_list, miller)
        push!(evc_list, evc)
    end
    
    # Measure memory usage with original approach
    println("\nMeasuring memory usage with original approach...")
    GC.gc()
    mem_before = Sys.total_memory() - Sys.free_memory()
    
    for ik in 1:nk
        wf_from_G(miller_list[ik], evc_list[ik], Nxyz)
    end
    
    mem_after = Sys.total_memory() - Sys.free_memory()
    mem_original = mem_after - mem_before
    
    # Measure memory usage with optimized approach
    println("Measuring memory usage with optimized approach...")
    clear_fft_buffers()
    GC.gc()
    mem_before = Sys.total_memory() - Sys.free_memory()
    
    for ik in 1:nk
        wf_from_G_optimized(miller_list[ik], evc_list[ik], Nxyz)
    end
    
    mem_after = Sys.total_memory() - Sys.free_memory()
    mem_optimized = mem_after - mem_before
    
    println("\nMemory Usage Results:")
    println("Original: $(@sprintf("%.2f", mem_original/1024^2)) MB")
    println("Optimized: $(@sprintf("%.2f", mem_optimized/1024^2)) MB")
    println("Memory reduction: $(@sprintf("%.1f%%", (1-mem_optimized/mem_original)*100))")
    
    return mem_optimized / mem_original
end

function benchmark_mpi_scaling()
    """Benchmark MPI scaling performance"""
    
    if mpi_size() == 1
        println("\n=== MPI Scaling Benchmark (Single Process) ===")
        println("Run with multiple MPI processes to see scaling:")
        println("mpirun -np 4 julia --project=. -t auto performance_benchmark.jl")
        return
    end
    
    println("\n=== MPI Scaling Benchmark ===")
    println("MPI Processes: $(mpi_size())")
    println("Threads per process: $(Threads.nthreads())")
    
    # Test parameters
    n_displacements = 100
    n_kpoints = 50
    
    # Distribute work across MPI ranks
    rank = mpi_rank()
    size = mpi_size()
    
    # Each rank processes a subset
    my_displacements = rank+1:size:n_displacements
    my_kpoints = rank+1:size:n_kpoints
    
    println("Rank $rank: processing $(length(my_displacements)) displacements, $(length(my_kpoints)) k-points")
    
    # Simulate work
    t_start = time()
    
    # Simulate displacement calculations
    for disp in my_displacements
        # Simulate some computational work
        sleep(0.001)  # 1ms per displacement
    end
    
    # Simulate k-point calculations
    for k in my_kpoints
        # Simulate some computational work
        sleep(0.002)  # 2ms per k-point
    end
    
    t_local = time() - t_start
    
    # Gather timing information
    all_times = mpi_gather([t_local])
    
    if is_master()
        total_time = maximum(all_times)
        avg_time = mean(all_times)
        efficiency = avg_time / total_time
        
        println("\nMPI Scaling Results:")
        println("Total time: $(@sprintf("%.4f", total_time)) s")
        println("Average time per rank: $(@sprintf("%.4f", avg_time)) s")
        println("Parallel efficiency: $(@sprintf("%.2f%%", efficiency*100))")
        
        # Calculate speedup
        theoretical_time = sum(all_times) / size
        speedup = theoretical_time / total_time
        println("Speedup: $(@sprintf("%.2fx", speedup))")
    end
end

function benchmark_hybrid_parallelism()
    """Benchmark hybrid MPI + threading performance"""
    
    println("\n=== Hybrid MPI + Threading Benchmark ===")
    
    # Test parameters
    n_operations = 1000
    
    # Distribute across MPI ranks
    rank = mpi_rank()
    size = mpi_size()
    my_ops = rank+1:size:n_operations
    
    println("Rank $rank: processing $(length(my_ops)) operations with $(Threads.nthreads()) threads")
    
    # Benchmark with threading
    t_start = time()
    
    results = zeros(length(my_ops))
    
    @threads for i in 1:length(my_ops)
        op_idx = my_ops[i]
        # Simulate computational work
        result = 0.0
        for j in 1:1000
            result += sin(j * op_idx) * cos(j * op_idx)
        end
        results[i] = result
    end
    
    t_local = time() - t_start
    
    # Gather results
    all_times = mpi_gather([t_local])
    all_results = mpi_gather(results)
    
    if is_master()
        total_time = maximum(all_times)
        total_workers = size * Threads.nthreads()
        
        println("\nHybrid Parallelism Results:")
        println("Total workers: $total_workers (MPI: $size × Threads: $(Threads.nthreads()))")
        println("Total time: $(@sprintf("%.4f", total_time)) s")
        println("Operations per second: $(@sprintf("%.0f", n_operations/total_time))")
        println("Operations per worker: $(@sprintf("%.1f", n_operations/total_workers))")
    end
end

function benchmark_fft_performance()
    """Benchmark FFT performance with different configurations"""
    
    println("\n=== FFT Performance Benchmark ===")
    
    # Test different grid sizes
    grid_sizes = [[16, 16, 16], [32, 32, 32], [64, 64, 64]]
    
    for Nxyz in grid_sizes
        println("\nGrid size: $Nxyz")
        
        # Generate test data
        data = randn(ComplexF64, Nxyz...)
        
        # Benchmark FFT
        fft_time = @elapsed fft(data)
        ifft_time = @elapsed ifft(data)
        
        println("FFT time: $(@sprintf("%.6f", fft_time)) s")
        println("IFFT time: $(@sprintf("%.6f", ifft_time)) s")
        println("Total FFT+IFFT: $(@sprintf("%.6f", fft_time + ifft_time)) s")
        
        # Calculate throughput
        grid_points = prod(Nxyz)
        throughput = grid_points / (fft_time + ifft_time)
        println("Throughput: $(@sprintf("%.0f", throughput)) points/s")
    end
end

function generate_performance_report()
    """Generate comprehensive performance report"""
    
    if is_master()
        println("\n" * "="^60)
        println("PERFORMANCE BENCHMARK REPORT")
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
    wave_speedup = benchmark_wave_function_transformations()
    memory_ratio = benchmark_memory_usage()
    benchmark_mpi_scaling()
    benchmark_hybrid_parallelism()
    benchmark_fft_performance()
    
    if is_master()
        println("\n" * "="^60)
        println("SUMMARY")
        println("="^60)
        println("Wave function transformation speedup: $(@sprintf("%.2fx", wave_speedup))")
        println("Memory usage reduction: $(@sprintf("%.1f%%", (1-memory_ratio)*100))")
        println("="^60)
    end
end

function main()
    """Main benchmark function"""
    
    # Set up FFTW for optimal performance
    FFTW.set_num_threads(Threads.nthreads())
    
    # Run comprehensive benchmark
    generate_performance_report()
    
    # Clean up
    clear_fft_buffers()
    
    if is_master()
        println("\nBenchmark completed!")
    end
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
