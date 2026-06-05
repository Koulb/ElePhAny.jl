using JSON3

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

#-----------------------------------------------------run_parsing::uses both parse_eb and fake2nscf and the model
function run_parsing(model::ModelQE,only_g::Bool = false,type_of_calc::String = "dft")

    #parsing epw0
    epw0_keys = ["prefix", "bands_skipped"]
    epw0_path="./epw0.in"
    epw0_params = parse_file_for_values(epw0_path, epw0_keys)
    println("Parsed parms epw0: ", epw0_params)

    # Extract max(number of bands skipped) from bands_skipped
    if haskey(epw0_params, "bands_skipped")
        numbers_reg = eachmatch(r"\d+", epw0_params["bands_skipped"])  # \d+ integer sequence
        values = [parse(Int, m.match) for m in numbers_reg]

        if !isempty(values)
            max_number_bands_skipped = last(values)
            println("bands_skipped found: max value = $max_number_bands_skipped")
        else
            max_number_bands_skipped = 0
            println("bands_skipped found but no numeric values, taking max_number_bands_skipped = 0")
        end
    else
        max_number_bands_skipped = 0
        println("bands_skipped not found, taking max_number_bands_skipped = 0")
    end

    #parsing nscf and sanity check
    nscf_path="./nscf.in"
    nscf_params = parse_file_for_values(nscf_path, nscf_keys)
    nscf_params["nbndep"] = model.scf_parameters(["nbnd"]) - max_number_bands_skipped

    if epw0_params["prefix"] != nscf_params["prefix"]
        error("unequal prefix values: epw0 = $(epw0_params["prefix"]) nscf = $(nscf_params["prefix"])")
    end
    
    #create single dictionary with no dups
    params = mergewith((x, y) -> x, epw0_params, nscf_params)

    #we launch parse_epb
    nbndep = params["nbndep"]
    epb_name = params["prefix"] * ".epb1"
    parse_epb(model ,nbndep,epb_name,only_g)

    #we launch fake2nscf 
    path_to_save = "./" * params["prefix"] * "_dft.save"
    path_to_out =  "./" * params["prefix"] * ".save"
    fake2nscf(model,path_to_save,path_to_out,type_of_calc)
end
# -----------------------------------------------------parse_epb and fake2nscf functions

function parse_epb(model::ModelQE,nbndep::Int,epb_name::String,only_g::Bool=false)
    parse_epb(
        model.path_to_calc,
        model.path_to_calc  * "displacements",
        model.scf_parameters(["nbnd"]),
        nbndep,
        prod(model.scf_parameters(["kpts"])),
        prod(model.sc_size),
        length(model.unitcell(["symbols"])),
        epb_name,
        only_g
    )   
end


function fake2nscf(model::ModelQE,path_to_save::String,path_to_out::String,type_of_calc::String = "dft")
    fake2nscf(
        model.path_to_calc,
        model.path_to_calc * "displacements/scf_0/tmp/scf.save",
        path_to_save,
        path_to_out,
        prod(model.scf_parameters(["kpts"])),
        type_of_calc
    )
end

#generic functions---------------------------------------------------------


function parse_epb(
    path_to_epw::String, 
    path_to_frozen::String, 
    nbnd::Int,  
    nbndep::Int,# - / parse_epw0.in
    mesh::Int, 
    mesh_q::Int,
    nat::Int, 
    epb_name::String,#- see the suffix  
    only_g::Bool = false# will be an option
    ) 

    # Derived parameters
    nbndep_skip = nbnd - nbndep
    nks = mesh
    nqtot = mesh_q
    nmodes = 3 * nat

    # Read the Fortran binary file
    fortranFilename = joinpath(path_to_epw, epb_name)
    epb_data = read_fortran_binary_generic(fortranFilename, nqtot, nks, nbnd, nmodes, nbndep, nat)
    nqc, xqc, et_loc, dynq, epmatq, zstar, epsi = epb_data

    # Get the reciprocal lattice vectors
    ase = pyimport("ase.io")
    path_to_scf = joinpath(path_to_epw, "scf.out")
    atoms = ase.read(path_to_scf)

    R = pyconvert(Matrix{Float64}, atoms.cell.reciprocal().T)
    a = pyconvert(Float64, atoms.cell.cellpar()[1])
    
    R .*= a
    
    a_factor = 2 * abs(R[1,1])
    
    real_vectors = R \ I
    real_vectors ./= a_factor

    #get the q points
    q_ph = copy(xqc)
    q_nscf = [determine_q_point_cart(path_to_epw, ik) for ik in 1:nqtot]
    if size(q_nscf)[1] != nqtot
        println("ERROR: Len of q_nscf $(size(q_nscf)[1]) does not match nqtot = $nqtot")
    end

    #create the matching sequence of q points
    iq_ph_list = []
    for i_ph in 1:size(q_ph)[1]
        for i_nscf in 1:size(q_nscf)[1]
            q_nscf_crystal = real_vectors * q_nscf[i_nscf]
            q_ph_crystal = real_vectors * q_ph[i_ph,:]
            check = [false, false, false]
            delta_q_all = abs.(q_nscf_crystal - q_ph_crystal)
            
            for (ind_q, delta_q) in enumerate(delta_q_all)
                if isapprox(delta_q, 0, atol=1e-5) || isapprox(delta_q, 1, atol=1e-5)
                    check[ind_q] = true
                end
            end
            
            if all(check)
                push!(iq_ph_list, i_nscf)
                break
            end
        end
    end

    # Reading the frozen phonon data for electron-phonon matrix elements
    g_frozen = zeros(ComplexF64, nbndep, nbndep, nks, nmodes, nqtot)
    for ik in 1:nks
        for (iq_ph, iq_nscf) in enumerate(iq_ph_list)
            path_to_file = path_to_frozen * "/epw/braket_list_rotated_$(ik)_$(iq_nscf)"
            data_lm = readdlm(path_to_file)
            
            for line in eachrow(data_lm)
                iat = Int(line[1])
                i_cart = Int(line[2])
                i_m = 3 * (iat - 1) + i_cart
                i = Int(line[4])
                j = Int(line[3])
                
                if (i > nbndep_skip) && (j > nbndep_skip) && (i <= nbndep + nbndep_skip) && (j <= nbndep + nbndep_skip)
                    g_frozen[i - nbndep_skip, j - nbndep_skip, ik, i_m, iq_ph] = line[5] - im * line[6]
                end
            end
        end
    end
    
    # Read dynamic matrix
    dynq_frozen = zeros(ComplexF64, nmodes, nmodes, nqtot)

    for (iq_ph, iq_nscf) in enumerate(iq_ph_list)
        dynq_frozen_data = readdlm(path_to_frozen * "/dyn_mat/dyn_mat$(iq_nscf)")
        dynq_frozen[:, :, iq_ph] = dynq_frozen_data[:, 1:2:end] .+ im .* dynq_frozen_data[:, 2:2:end]
    end

    # Update eigenvalues
    et_loc_frozen = zeros(size(et_loc))

    frozen_file = path_to_frozen * "/scf_0/tmp/scf.save/data-file-schema.xml"
    energy_dicts = read_kpoint_eigenvals(frozen_file)

    for ik in 1:nks
        e_values = energy_dicts[ik]["energies"]
        et_loc_frozen[ik, :] = 2 .* e_values  # Ha to Ry
    end


    #prepare the data for writing and write into the epb file
    epb_data_list = collect(epb_data)
    xqc_f = permutedims(xqc, (2,1))  # (3, nqtot)
    if only_g
        println("doing only_g == true")
        et_loc_fortan = permutedims(et_loc,(2,1))
        epb_data_list[5] = g_frozen  # Only update g array
        epb_data_list[2] = xqc_f
        epb_data_list[3] = et_loc_fortan
        epb_data_list[4] = dynq

    else
        println("doing only_g == false")
        epb_data_list[2] = xqc_f        #phonon points
        epb_data_list[5] = g_frozen      # electron-phonon coupling
        epb_data_list[4] = dynq_frozen   # phonons
        epb_data_list[3] = et_loc_frozen' # eigenvalues
    end

    new_epb_data = tuple(epb_data_list...)
    write_binary_fortran(fortranFilename, new_epb_data)

    println("Successfully wrote updated EPB file: $fortranFilename")

end

function fake2nscf(
    path_to_data::String,
    path_to_kcw::String,
    path_to_save::String,
    path_to_out::String,
    nks::Int,
    type_of_calc::String = "dft"
    )
    
    #util functions-----------------------------------------------------------------------------

    function read_kpoint_eigenvals_as_strings(xmlPath; type_of_calc = "dft")
        doc = readxml(xmlPath)
        ks_energies_nodes = findall("//ks_energies", doc)
        Energy_dicts = []
        
        for ks_energies in ks_energies_nodes
            # Get k_point
            k_point_elem = findfirst("k_point", ks_energies)
            k_point = nodecontent(k_point_elem)
            
            # Get energies as STRINGS (like Python)
            energy_elem = findfirst("eigenvalues", ks_energies)
            eigenvals_txt = nodecontent(energy_elem)
            
            # Extract energy strings and replace 'e-' with 'E-0' (like Python)
            pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
            matches = eachmatch(pattern, eigenvals_txt)
            energies = [replace(m.match, "e-" => "E-0") for m in matches]
            
            # For kcw, take only first half
            if type_of_calc == "kcw"
                half_length = length(energies) ÷ 2
                energies = energies[1:half_length]
            end
            
            push!(Energy_dicts, Dict("k_point" => k_point, "energies" => energies))
        end
        
        return Energy_dicts
    end

    function expand_self_closing_tags!(xml_path)
        lines = readlines(xml_path)
        
        expanded = map(lines) do line
            replace(line, r"<(\w+)([^>]*)/>" => s"<\1\2></\1>")
        end
        
        open(xml_path, "w") do f
            write(f, join(expanded, "\n"))
        end
    end
    
    #core of the function fake2nscf-----------------------------------------------------------
    # Build file paths
    path_kp = joinpath(path_to_data, path_to_kcw, "data-file-schema.xml")
    path_qe = joinpath(path_to_data, path_to_save, "data-file-schema.xml")
    path_out = joinpath(path_to_data, path_to_out, "data-file-schema.xml")

    # Save first 3 and last line for fixing at the end
    lines = readlines(path_qe)
    replace_lines = [lines[1], lines[2], lines[3], lines[end]]

    # Parse tree and get energies from kcw file AS STRINGS
    tree_qe = readxml(path_qe)
    E_kp_array = read_kpoint_eigenvals_as_strings(path_kp, type_of_calc = type_of_calc)

    # Change energies in target file
    ks_energies_nodes = findall("//ks_energies", tree_qe)

    for (index_k, ks_energies) in enumerate(ks_energies_nodes)
        k_point_node = findfirst("k_point", ks_energies)
        k_point = nodecontent(k_point_node)
        
        eigenvalues_node = findfirst("eigenvalues", ks_energies)
        energies_qe_raw = nodecontent(eigenvalues_node)
        
        # Extract energy strings from target file
        pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
        matches_qe = eachmatch(pattern, energies_qe_raw)
        energies_qe_strings = [m.match for m in matches_qe]
        
        # Get the kp energies (these are STRINGS now, not Float64)
        energies_kp = E_kp_array[index_k]["energies"]
        
        # Replace energies one by one (matching Python logic)
        if type_of_calc == "kcw"
            # Replace each energy value in the original string, one at a time
            for (index_e, energy_qe_str) in enumerate(energies_qe_strings)
                if index_e <= length(energies_kp)
                    # energies_kp is already a string, no conversion needed!
                    energy_kp_str = energies_kp[index_e]
                    # Replace first occurrence only (like Python's replace with count=1)
                    energies_qe_raw = replace(energies_qe_raw, energy_qe_str => energy_kp_str, count=1)
                end
            end
        else
            # For dft, replace all energies
            for (index_e, energy_kp_str) in enumerate(energies_kp)
                if index_e <= length(energies_qe_strings)
                    # energy_kp_str is already a string!
                    energies_qe_raw = replace(energies_qe_raw, energies_qe_strings[index_e] => energy_kp_str, count=1)
                end
            end
        end
        
        # Update the node
        setnodecontent!(eigenvalues_node, energies_qe_raw)
    end
    
    # Save results
    write(path_out, tree_qe)

    # Additional change of first 3 and last line for complete agreement
    lines_kp = readlines(path_out)
    lines_kp[1] = replace_lines[1]
    lines_kp[2] = replace_lines[2]
    lines_kp[3] = replace_lines[3]
    lines_kp[end] = replace_lines[4]

    open(path_out, "w") do f
        write(f, join(lines_kp, "\n"))
    end
    expand_self_closing_tags!(path_out
    )
    
    # Copy wavefunction files
    if type_of_calc == "kcw"
        for i in 1:nks
            src_file = joinpath(path_to_data, path_to_kcw, "wfcup$(i).dat")
            dest_file = joinpath(path_to_data, path_to_out, "wfc$(i).dat")
            cp(src_file, dest_file, force=true)
        end
        
        src_charge = joinpath(path_to_data, path_to_save, "charge-density.dat")
        dest_charge = joinpath(path_to_data, path_to_out, "charge-density.dat")
        cp(src_charge, dest_charge, force=true)
    else
        for i in 1:nks
            src_file = joinpath(path_to_data, path_to_kcw, "wfc$(i).dat")
            dest_file = joinpath(path_to_data, path_to_out, "wfc$(i).dat")
            cp(src_file, dest_file, force=true)
        end
        
        src_charge = joinpath(path_to_data, path_to_kcw, "charge-density.dat")
        dest_charge = joinpath(path_to_data, path_to_out, "charge-density.dat")
        cp(src_charge, dest_charge, force=true)
    end

    
end

function read_fortran_binary_generic(filename::String, nqtot::Int, nks::Int, nbnd::Int, nmodes::Int, nbndep::Int, nat::Int)
    open(filename, "r") do io
        # reading an Int32 first which is the size of the record; this is the record marker
        read(io, Int32)
        
        function read_fortran_array(io, T, doReshape, dims...)            
            total_elements = prod(dims)
            arr = Vector{T}(undef, total_elements)
            flat_data = read!(io, arr)
            if doReshape
                if length(dims) == 2
                    return reshape(flat_data, reverse(dims)...)'
                elseif length(dims) > 2 # 3 dim or more
                    return permutedims(reshape(flat_data, reverse(dims)), (2,1,3:length(dims)...) ) 
                end
            else
                return flat_data
            end
        end

        function read_fortran_complex_ndarray(io, dims::NTuple{N,Int};
                                                real_type::DataType=Float64,
                                                complex_type::DataType=ComplexF64) where N
            total_elements = prod(dims)
        
            # Read raw data and reinterpret as complex
            raw_data = read_fortran_array(io, real_type, false, 2 * total_elements)
            reshaped_data = reshape(raw_data, (2, total_elements))
            complex_flat = dropdims(reinterpret(complex_type, reshaped_data), dims=1)
            
            # Reshape to the target dimensions and convert Fortran→Julia ordering
            # return permutedims(reshape(complex_flat, reverse(dims)), (2,1,3:length(dims)...))
            return permutedims(reshape(complex_flat, reverse(dims)),reverse(1:N))
        end

        # Read each field
        data_tuple = (
            read(io, Int32),                                                     # nqc 
            read_fortran_array(io, Float64, true, nqtot, 3),                    # xqc
            read_fortran_array(io, Float64, true, nks, nbnd),                   # et_loc
            read_fortran_complex_ndarray(io, (nqtot, nmodes, nmodes)),          # dynq
            read_fortran_complex_ndarray(io, (nqtot, nmodes, nks, nbndep, nbndep)), # epmatq
            read_fortran_array(io, Float64, true, 3, 3, nat),                   # zstar
            read_fortran_array(io, Float64, true, 3, 3)                         # epsi
        )
        
        return data_tuple
    end
end

function sizeofArr(arr)
    return prod(size(arr)) * sizeof(eltype(arr))
end

function write_binary_fortran(filename, data_tuple)
    open(filename, "w") do io
        rec_size = sum([sizeofArr(arr) for arr in data_tuple])
        println("Saving Fortran file $filename with record size $rec_size bytes")
        
        # Write start record marker (Int32)
        write(io, Int32(rec_size))
        
        # Write data
        for arr in data_tuple
            write(io, arr)
        end
        
        # Write end record marker (Int32)
        write(io, Int32(rec_size))
    end
end

function determine_q_point_cart(path_to_in::String,ik::Int)
    file = open("$(path_to_in)/nscf.out","r")
    lines = readlines(file)
    result = 0.0

    count = 1
    for (index, line) in enumerate(lines)
        if occursin("        k(" , line)
            if ik == count
                result_str = split(line)[5:7]#
                result_str[3] = result_str[3][1:end-2]
                result = parse.(Float64, result_str)
                break
            else
                count += 1
            end
        end
    end

    return result
end

function read_kpoint_eigenvals(xmlPath; type_of_calc = "dft")

    """
        Returns all K-Point and Eigenvalues from an XML file
        If type_of_calc = "kcw", it only returns half of the available eigenvalues
        Args:
            xmlPath: Path to the XML file
            type_of_calc: determines the eigenvalues selected; "dft" returns all
        
        Returns:
            Energy_dicts: Array of dict {k_points,energies}: empty if no nodes found
    """

    # Find all ks_energies nodes
    doc = readxml(xmlPath)
    ks_energies_nodes = findall("//ks_energies", doc)
    Energy_dicts = []

    for ks_energies in ks_energies_nodes
        # Get k_point from first child element
        k_point_elem = findfirst("k_point", ks_energies)
        k_point = nodecontent(k_point_elem)
        
        # Get energies from third child element (assuming specific structure)
        energy_elem =  findfirst("eigenvalues", ks_energies)
        eigenvals_txt = nodecontent(energy_elem)
        eigenvals = parse_energy_text(eigenvals_txt)
        
        if type_of_calc == "kcw"
            eigenvals = eigenvals[1:length(eigenvals)÷2]
        end
        
        push!(Energy_dicts, Dict("k_point" => k_point, "energies" => eigenvals))
    end

    return Energy_dicts
end


function parse_energy_text(energy_text)
    # Normalize scientific notation
    normalized = replace(energy_text, "e-" => "E-0", "e+" => "E+0")

    # Extract all float patterns
    pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[E][-+]?[0-9]+)?"
    matches = eachmatch(pattern, normalized)

    # Parse to Float64
    return parse.(Float64, [m.match for m in matches])
end

function parse_line(line) # avoid issues where there is an equal sign on the RHS
    # Use regex to split on the first '=' not inside quotes
    m = match(r"^\s*([^=]+?)\s*=\s*(.*)$", line)
    if m === nothing
        throw(ArgumentError("Invalid line format: $line"))
    end
    lhs = strip(m.captures[1])  # Left-hand side (e.g., "y")
    rhs = strip(m.captures[2])  # Right-hand side (e.g., "'hello = world'")

    # Remove surrounding quotes if present
    if startswith(rhs, '\'') && endswith(rhs, '\'')
        rhs = rhs[2:end-1]  # Remove the quotes
    elseif startswith(rhs, '"') && endswith(rhs, '"')
        rhs = rhs[2:end-1]  # Remove the quotes
    end

    return (lhs, rhs)
end

function parse_file_for_values(file_path::String, keys::Vector{String})::Dict{String, Any}
    # Initialize the dictionary to store key-value pairs
    key_value_pairs = Dict{String, Any}()
    key = "" 
    value = ""
    # Read the file line by line
    open(file_path, "r") do file
        for line in eachline(file)
            try 
                key, value = parse_line(line) 
            catch
                continue
            end
            # Check if the key is in the list of keys we are interested in
            if key in keys
                # Try to convert the value to a number if possible
                try
                    value = parse(Int, value)
                catch
                    # If conversion fails, keep the value as a string
                end
                key_value_pairs[key] = value
            end
        end
    end

    return key_value_pairs
end