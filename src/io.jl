using JSON3
using BlockArrays, LinearAlgebra, Mmap

"""
    parse_frozen_params(path_to_json)

Parses a JSON file containing parameters for a frozen phonon calculation and merges them with default values.

# Arguments
- `path_to_json::AbstractString`: Path to the JSON file containing parameter overrides.

# Returns
- `Dict{String, Any}`: A dictionary with the merged parameters, where defaults are used unless overridden by the JSON file.
"""
function parse_frozen_params(path_to_json)
    frozen_params_default = Dict(
        "path_to_calc" =>  pwd()*"/",
        "abs_disp" => 1e-3,
        "mpi_ranks" => 8,
        "sc_size" => Vector{Int}([1,1,1]),
        "k_mesh" => Vector{Int}([1,1,1]),
        "Ndispalce" => 12
    )

    frozen_params_new = JSON3.read(path_to_json)
    frozen_params_new = Dict(string(k) => v for (k, v) in frozen_params_new)

    for (key, value) in frozen_params_new
        frozen_params_default[key] = value
        if key == "sc_size" || key == "k_mesh"
            frozen_params_default[key] = Vector{Int}(value)
        end
    end

    return frozen_params_default
end

#Inspired from DFControl.jl

const NeedleType = Union{AbstractString, AbstractChar, Regex}

"""
    parse_qe_in(path_to_scf::String)

Parses a Quantum ESPRESSO SCF input file and extracts structural and calculation parameters.

# Arguments
- `path_to_scf::String`: Path to the Quantum ESPRESSO SCF input file.

# Returns
- `unitcell::Dict`: A dictionary containing unit cell information:
    - `:symbols`: Chemical symbols of the atoms.
    - `:cell`: Lattice vectors of the unit cell.
    - `:scaled_positions`: Atomic positions in fractional coordinates.
    - `:masses`: Atomic masses.
- `scf_parameters::Dict`: A dictionary containing parsed SCF calculation parameters, including:
    - `:format`: The format of the input file (set to `"espresso-in"`).
    - `:crystal_coordinates`: Boolean indicating if crystal coordinates are used.
    - Additional parameters parsed from the input file.
"""
function parse_qe_in(path_to_scf::String)
    ase_data = ase_io.read(path_to_scf)

    unitcell = Dict(
        :symbols => ase_data.get_chemical_symbols(),
        :cell => pylist(ase_data.cell.array),
        :scaled_positions => ase_data.get_scaled_positions(),
        :masses => ase_data.get_masses()
    )

    #now goes the parsing of the scf.in file for the other parameters
    scf_parameters = Dict{Symbol, Any}(:format => "espresso-in",
                                       :crystal_coordinates => true)
    parse_file(path_to_scf, QE_PW_PARSE_FUNCTIONS, out = scf_parameters)

    return unitcell, scf_parameters
end

"""
    getfirst(f::Function, A)

Returns the first element `el` in the collection `A` for which the predicate function `f(el)` returns `true`.
If no such element is found, returns `nothing`.

# Arguments
- `f::Function`: A predicate function that takes an element of `A` and returns a `Bool`.
- `A`: A collection to search through.
"""
function getfirst(f::Function, A)
    for el in A
        if f(el)
            return el
        end
    end
end

"""
    parse_file(f::AbstractString, parse_funcs::Vector{<:Pair{NeedleType, Any}}; out = Dict{Symbol, Any}())

Parses the file at path `f` line by line, applying parsing functions specified in `parse_funcs` to lines that match given patterns.

# Arguments
- `f::AbstractString`: Path to the file to be parsed.
- `parse_funcs::Vector{<:Pair{NeedleType, Any}}`: A vector of pairs, where each pair consists of a pattern (`NeedleType`) to search for in each line, and a parsing function to apply when the pattern is found. The parsing function should accept three arguments: the output dictionary, the current line, and the file handle.
- `out::Dict{Symbol, Any}` (optional): An output dictionary to store parsed results. Defaults to an empty dictionary.

# Returns
- `Dict{Symbol, Any}`: The dictionary containing parsed results.
"""
function parse_file(f::AbstractString, parse_funcs::Vector{<:Pair{NeedleType, Any}}; out = Dict{Symbol, Any}())
    lc = 0
    open(f, "r") do file
        while !eof(file)
            line = strip(readline(file))
            lc += 1
            if isempty(line)
                continue
            end
            # Iterate over the pairs in QE_PW_PARSE_FUNCTIONS to find matching lines
            for pf in parse_funcs
                if occursin(pf.first, line)
                    # try
                        pf.second(out, line, file)
                    # catch e
                        # @warn "File corruption or parsing error detected executing parse function $(pf.second) in file $f at line $lc: \"$line\".\nTrying to continue smoothly. Error: $e"
                    # end
                    break  # Exit the loop once the matching function is found
                end
            end
        end
    end
    return out
end

"""
    parse_file(f::AbstractString, args...; kwargs...)

Opens the file specified by the path `f` in read mode and passes the resulting file handle,
along with any additional positional (`args...`) and keyword arguments (`kwargs...`), to
the inner `parse_file` method for further processing.

# Arguments
- `f::AbstractString`: The path to the file to be parsed.
- `args...`: Additional positional arguments to be forwarded to the inner `parse_file` method.
- `kwargs...`: Additional keyword arguments to be forwarded to the inner `parse_file` method.
"""
function parse_file(f::AbstractString, args...; kwargs...)
    open(f, "r") do file
        parse_file(file, args...;kwargs...)
    end
end

function qe_parse_calculation(out, line, f)
    out[:calculation] = strip(split(line)[3], ['\''])
end

function qe_parse_verbosity(out, line, f)
    out[:verbosity] = strip(split(line)[3], ['\''])
end

function qe_parse_tstress(out, line, f)
    out[:tstress] = split(line)[3] == ".true."
end

function qe_parse_tprnfor(out, line, f)
    out[:tprnfor] = split(line)[3] == ".true."
end

function qe_parse_outdir(out, line, f)
    out[:outdir] = strip(split(line)[3], ['\''])
end

function qe_parse_prefix(out, line, f)
    out[:prefix] = strip(split(line)[3], ['\''])
end

function qe_parse_pseudo_dir(out, line, f)
    out[:pseudo_dir] = strip(split(line)[3], ['\''])
end

function qe_parse_ibrav(out, line, f)
    out[:ibrav] = parse(Int, split(line)[3])
end

function qe_parse_nbnd(out, line, f)
    out[:nbnd] = parse(Int, split(line)[3])
end

function qe_parse_ecutwfc(out, line, f)

    out[:ecutwfc] = parse(Float64, split(line)[3])
end

function qe_parse_ecutrho(out, line, f)
    out[:ecutrho] = parse(Float64, split(line)[3])
end

function qe_parse_nosym(out, line, f)
    out[:nosym] = split(line)[3] == ".true."
end

function qe_parse_noinv(out, line, f)
    out[:noinv] = split(line)[3] == ".true."
end

function qe_parse_nat(out, line, f)
    out[:nat] = parse(Int, split(line)[3])
end

function qe_parse_diagonalization(out, line, f)
    out[:diagonalization] = strip(split(line)[3], ['\''])
end

function qe_parse_electrons_maxstep(out, line, f)
    out[:electron_maxstep] = parse(Int, split(line)[3])
end

function qe_parse_mixing_mode(out, line, f)
    out[:mixing_mode] = strip(split(line)[3], ['\''])
end

function qe_parse_mixing_beta(out, line, f)
    out[:mixing_beta] = parse(Float64, split(line)[3])
end

function qe_parse_conv_thr(out, line, f)
    out[:conv_thr] = parse(Float64, split(line)[3])
end

function qe_parse_kpoints(out, line, f)
    line = readline(f)
    values = [parse(Int,val) for val in  split(line)]
    #case of uniform unshifted grid for now
    out[:kpts] = pytuple((values[1], values[2], values[3]))
end

function qe_parse_pseudo(out, line, f)
    out[:pseudopotentials] = Dict()
    # line = readline(f)
    while !eof(f)
        line = strip(readline(f))
        length(split(line)) != 3 && break
        species = split(line)
        out[:pseudopotentials][species[1]] = species[3]
    end
end

const QE_PW_PARSE_FUNCTIONS::Vector{Pair{NeedleType, Any}}  = [
    #Contol
    "calculation" => qe_parse_calculation,
    "verbosity" => qe_parse_verbosity,
    "tstress" => qe_parse_tstress,
    "tprnfor" => qe_parse_tprnfor,
    "outdir" => qe_parse_outdir,
    "prefix" => qe_parse_prefix,
    "pseudo_dir" => qe_parse_pseudo_dir,
    #System
    "ibrav" => qe_parse_ibrav,
    "nbnd" => qe_parse_nbnd,
    "ecutwfc" => qe_parse_ecutwfc,
    "ecutrho" => qe_parse_ecutrho,
    "nosym" => qe_parse_nosym,
    "noinv" => qe_parse_noinv,
    "nat" => qe_parse_nat,
    #Electrons
    "diagonalization" => qe_parse_diagonalization,
    "electrons_maxstep" => qe_parse_electrons_maxstep,
    "mixing_mode" => qe_parse_mixing_mode,
    "mixing_beta" => qe_parse_mixing_beta,
    "conv_thr" => qe_parse_conv_thr,
    #Atomic_species
    "ATOMIC_SPECIES" => qe_parse_pseudo,
    #K-points
    "K_POINTS" => qe_parse_kpoints,
]


#simple test

# path_to_scf = "/scratch/apolyukhin/julia_tests/elph_tst/sym_tst/si_sc_kpoints_new/displacements/scf_0/scf.in"
# unitcell, scf_parameters = parse_qe_in(path_to_scf)
# println(unitcell)
# println(scf_parameters)

# Memory-mapped wave function loading for large datasets
function load_wave_functions_mmap(path_to_in::String, ik::Int)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik"
    
    if isfile(file_path*".dat")
        return load_wave_functions_mmap_dat(file_path*".dat")
    elseif isfile(file_path*".hdf5")
        return load_wave_functions_mmap_hdf5(file_path*".hdf5")
    else
        error("No wave function file found: $file_path")
    end
end

function load_wave_functions_mmap_dat(file_path::String)
    # Memory-map the binary file for efficient reading
    mmap_file = Mmap.mmap(file_path)
    
    # Parse header information
    ik = reinterpret(Int32, mmap_file[1:4])[1]
    xkx, xky, xkz, ispin = reinterpret(Float64, mmap_file[5:20])
    ngw, igwx, npol, nbnd = reinterpret(Int32, mmap_file[21:28])
    
    # Calculate offsets
    header_size = 28
    dummy_size = 9 * 8  # 9 Float64 values
    miller_size = 3 * igwx * 4  # 3*igwx Int32 values
    
    miller_offset = header_size + dummy_size
    evc_offset = miller_offset + miller_size
    
    # Extract Miller indices
    miller_data = reinterpret(Int32, mmap_file[miller_offset+1:miller_offset+miller_size])
    miller = reshape(miller_data, (3, igwx))
    
    # Extract eigenvector coefficients
    evc_list = []
    current_offset = evc_offset
    
    for band in 1:nbnd
        evc_data = reinterpret(ComplexF64, mmap_file[current_offset+1:current_offset+igwx*16])
        push!(evc_list, copy(evc_data))
        current_offset += igwx * 16
    end
    
    Mmap.munmap(mmap_file)
    return miller, evc_list
end

function load_wave_functions_mmap_hdf5(file_path::String)
    # For HDF5 files, we still use the regular HDF5 interface
    # but can optimize the reading of large datasets
    return parse_hdf(file_path)
end

# Optimized wave function saving with compression
function save_wave_functions_optimized(path_to_in::String, wfc_list::Dict, ik::Int; compress::Bool=true)
    if compress
        # Use compression for large datasets
        save(path_to_in*"/wfc_list_$ik.jld2", wfc_list, compress=true)
    else
        save(path_to_in*"/wfc_list_$ik.jld2", wfc_list)
    end
end

# Streaming wave function processing for very large datasets
function process_wave_functions_streaming(path_to_in::String, ik::Int; batch_size::Int=5)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik"
    miller, evc_list = parse_wf(file_path)
    Nxyz = determine_fft_grid(path_to_in*"/scf.out")
    
    n_bands = length(evc_list)
    wfc_list = Dict()
    
    # Process in batches to manage memory
    for batch_start in 1:batch_size:n_bands
        batch_end = min(batch_start + batch_size - 1, n_bands)
        batch_indices = batch_start:batch_end
        
        # Process batch in parallel
        batch_results = Dict()
        @threads for index in batch_indices
            wfc = wf_from_G_optimized(miller, evc_list[index], Nxyz)
            batch_results["wfc$index"] = wfc
        end
        
        # Save batch immediately
        for (key, value) in batch_results
            wfc_list[key] = value
        end
        
        # Force garbage collection
        GC.gc()
    end
    
    save_wave_functions_optimized(path_to_in, wfc_list, ik)
    return wfc_list
end

# Optimized force collection with MPI
function collect_forces_mpi(path_to_in::String, unitcell, sc_size, Ndispalce)
    rank = mpi_rank()
    size = mpi_size()
    
    number_atoms = length(unitcell[:symbols])*sc_size[1]*sc_size[2]*sc_size[3]
    
    # Distribute displacements across MPI ranks
    local_disps = rank+1:size:Ndispalce
    local_forces = Array{Float64}(undef, length(local_disps), number_atoms, 3)
    
    for (local_idx, i_disp) in enumerate(local_disps)
        dir_name = "group_"*string(i_disp)*"/"
        force = nothing
        
        if isfile(path_to_in*dir_name*"/tmp/scf.save/data-file-schema-scf.xml")
            force = read_forces_xml(path_to_in*dir_name*"/tmp/scf.save/data-file-schema-scf.xml")
        else
            force = read_forces_xml(path_to_in*dir_name*"/tmp/scf.save/data-file-schema.xml")
        end
        
        local_forces[local_idx,:,:] = force
    end
    
    # Gather all forces to master
    if is_master()
        forces = Array{Float64}(undef, Ndispalce, number_atoms, 3)
        
        # Copy local forces
        for (local_idx, global_idx) in enumerate(local_disps)
            forces[global_idx,:,:] = local_forces[local_idx,:,:]
        end
        
        # Receive forces from other ranks
        for r in 1:size-1
            for (local_idx, global_idx) in enumerate(r+1:size:Ndispalce)
                force_data = MPI.Recv(Array{Float64,3}, r, MPI_COMM[], global_idx)
                forces[global_idx,:,:] = force_data
            end
        end
        
        return forces
    else
        # Send forces to master
        for (local_idx, global_idx) in enumerate(local_disps)
            MPI.Send(local_forces[local_idx,:,:], 0, MPI_COMM[], global_idx)
        end
        return nothing
    end
end

export parse_qe_in, parse_frozen_params
export load_wave_functions_mmap, save_wave_functions_optimized, process_wave_functions_streaming, collect_forces_mpi
include("io.jl")
