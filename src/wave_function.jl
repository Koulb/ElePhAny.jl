using FortranFiles, LinearAlgebra, Base.Threads, ProgressMeter, JLD2, FFTW, HDF5, Statistics

"""
    parse_wf(path::String)

Parses a wave function file specified by `path`.
The function checks for the existence of either a `.dat` or `.hdf5` file with the given base path.
Returns a tuple `(miller, evc_list)` containing the Miller indices and the eigenvector coefficients.

# Arguments
- `path::String`: The base path (without extension) to the wave function file.

# Returns
- `miller`: Miller indices 3xNG.
- `evc_list`: List of eigenvector coefficients parsed from the file NbandsxNg.

# Throws
- An error if neither a `.dat` nor a `.hdf5` file is found at the specified path.
"""
function parse_wf(path::String)
    evc_list = []
    miller = []

    #check if path*".dat" or path*".hdf5" exists
    if isfile(path*".dat")
        miller, evc_list = parse_fortran_bin(path*".dat")
    elseif isfile(path*".hdf5")
        miller, evc_list = parse_hdf(path*".hdf5")
    else
        error("File not found: $path"*".dat or $path"*".hdf5")
    end
    return miller, evc_list
end

"""
    parse_hdf(path::String) -> (miller, evc_list)

Parse an HDF5 file at the given `path` to extract Miller indices and eigenvector coefficients.

# Arguments
- `path::String`: The file path to the HDF5 file.

# Returns
- `miller`: Array containing Miller indices.
- `evc_list`: Vector of complex eigenvector coefficients, where each element corresponds to a band.
"""
function parse_hdf(path::String)
    evc_list = []
    miller = []

    miller, evc_list = h5open(path, "r") do f
        fkeys = collect(keys(f))
        miller_key = fkeys[1]
        evc_key = fkeys[2]
        data = read(f)
        miller = data[miller_key]

        evc_list_raw = data[evc_key]
        nbands = size(data[evc_key], 2)
        evc_list = []
        for ind in 1:nbands
            evc_real = evc_list_raw[1:2:end, ind]
            evc_imag = evc_list_raw[2:2:end, ind]
            push!(evc_list, evc_real .+ im * evc_imag)
        end

        (miller,evc_list)
    end

    return miller, evc_list
end

"""
    parse_fortran_bin(path::String) -> (miller, evc_list)

Parse an Fortran biniary file at the given `path` to extract Miller indices and eigenvector coefficients.

# Arguments
- `path::String`: The file path to the wfc#.dat file.

# Returns
- `miller`: Array containing Miller indices.
- `evc_list`: Vector of complex eigenvector coefficients, where each element corresponds to a band.
"""
function parse_fortran_bin(file_path::String)
    f = FortranFile(file_path)
    ik, xkx, xky, xkz, ispin = read(f, (Int32,5))
    ngw, igwx, npol, nbnd = read(f, (Int32,4))
    dummy_vector = read(f, (Float64,9))
    miller = reshape(read(f, (Int32,3*igwx)),(3, igwx))

    evc_list = []
    for _ in 1:nbnd
        evc = read(f, (ComplexF64,igwx))
        push!(evc_list,evc)
    end
    return miller, evc_list
end

"""
    wf_from_G(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz) -> Array{ComplexF64, 3}

Constructs a real-space wave function from its plane-wave expansion coefficients in reciprocal space.

# Arguments
- `miller::Matrix{Int32}`: A 3×N matrix where each column represents a Miller index (reciprocal lattice vector) for a plane wave component.
- `evc::Vector{ComplexF64}`: A vector of complex coefficients corresponding to each Miller index, representing the amplitude and phase of each plane wave.
- `Nxyz`: The size of the cubic grid in each spatial dimension (assumes a cubic box of size Nxyz × Nxyz × Nxyz).

# Returns
- `wave_function::Array{ComplexF64, 3}`: The reconstructed wave function in real space, represented as a 3D array of complex values.

# Details
- The function maps Miller indices to the appropriate indices in a reciprocal space grid, fills in the provided coefficients, and then performs an inverse FFT (using `bfft` after `ifftshift`) to obtain the real-space wave function.
- The Miller indices are shifted and wrapped to fit within the grid dimensions.
"""
function wf_from_G(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz)
    reciprocal_space_grid = zeros(ComplexF64, Nxyz[1], Nxyz[2], Nxyz[3])
    # Determine the shift needed to map Miller indices to grid indices
    shift = div.(Nxyz, 2)
    # shift = div(Nxyz, 2) .+ 1

    # Iterate through Miller indices and fill in the known coefficients
    for idx in 1:size(miller, 2)
        g_vector = Int.(miller[:, idx])
        # Map the Miller indices to reciprocal space grid indices
        i, j, k = ((g_vector .+ shift) .% Nxyz) .+ 1
        coefficient = evc[idx]
        reciprocal_space_grid[i, j, k] = coefficient
    end

    # Perform the inverse FFT to obtain the real-space wave function
    wave_function = bfft(ifftshift(reciprocal_space_grid))
    return wave_function
end

"""
    wf_from_G_slow(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz) -> Array{ComplexF64,3}

Constructs a real-space wave function on a 3D grid from its plane-wave expansion coefficients withot FFT.

# Arguments
- `miller::Matrix{Int32}`: A matrix where each column represents a reciprocal lattice vector (Miller indices) for the plane-wave basis.
- `evc::Vector{ComplexF64}`: The expansion coefficients (eigenvector components) corresponding to each plane-wave basis vector.
- `Nxyz`: The number of grid points along each spatial dimension (assumes a cubic grid).

# Returns
- `wave_function::Array{ComplexF64,3}`: The computed wave function values on a 3D grid of size `(Nxyz, Nxyz, Nxyz)`.
"""
function wf_from_G_slow(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz)
    x = range(0, 1-1/Nxyz[1], Nxyz[1])
    y = range(0, 1-1/Nxyz[2], Nxyz[2])
    z = range(0, 1-1/Nxyz[3], Nxyz[3])

    wave_function = zeros(ComplexF64,(Nxyz[1], Nxyz[2], Nxyz[3]))

    @threads for i in eachindex(x)
        for j in eachindex(y)
            for k in eachindex(z)
                x_i = x[i]
                y_j = y[j]
                z_k = z[k]

                r_ijk = [x_i, y_j, z_k]
                temp  = transpose(r_ijk) * miller
                exponent =  exp.(-2 * π * 1im * temp)
                # wave_function[i, j, k] = sum(transpose(exponent)'evc)
                wave_function[i, j, k] = dot(exponent, evc)
            end
        end
    end
    return wave_function
end

"""
    wf_from_G_list(miller::Matrix{Int32}, evc_list::AbstractArray{Any}, Nxyz) -> Array{ComplexF64, 4}

Constructs the real-space wave function from a list of plane-wave coefficients and their corresponding Miller indices.

# Arguments
- `miller::Matrix{Int32}`: A 3×N matrix where each column contains the Miller indices (G-vectors) for a plane wave.
- `evc_list::AbstractArray{Any}`: A list (or array) of Nbands wave-functions' coefficients corresponding to each G-vector.
- `Nxyz`: The size of the cubic grid in each spatial direction.

# Returns
- `wave_function::Array{ComplexF64, 4}`: A 4D array of complex values representing the wave function in real space, with dimensions `(N_evc, Nxyz, Nxyz, Nxyz)`, where `N_evc` is the number of eigenvector coefficients.

# Description
This function maps the plane-wave coefficients onto a reciprocal space grid according to their Miller indices, applies an inverse FFT shift, and then performs an inverse FFT to obtain the real-space wave function. The function is parallelized over the Miller index list for efficiency.

# Notes
- The Miller indices are shifted to map correctly onto the FFT grid.
- The function assumes a cubic grid and that the Miller indices are provided in the correct format.
"""
function wf_from_G_list(miller::Matrix{Int32}, evc_list::AbstractArray{Any}, Nxyz)
    evc_matrix = permutedims(hcat(evc_list...))
    N_evc = size(evc_matrix)[1]
    reciprocal_space_grid = zeros(ComplexF64, N_evc, Nxyz[1], Nxyz[2], Nxyz[3])
    wave_function         = zeros(ComplexF64, N_evc, Nxyz[1], Nxyz[2], Nxyz[3])

    # Determine the shift needed to map Miller indices to grid indices
    shift = div.(Nxyz, 2)

    @threads for idx in 1:size(miller, 2)
        i = (Int(miller[1, idx]) + shift[1]) % Nxyz[1] + 1
        j = (Int(miller[2, idx]) + shift[2]) % Nxyz[2] + 1
        k = (Int(miller[3, idx]) + shift[3]) % Nxyz[3] + 1
        reciprocal_space_grid[:, i, j, k] .= evc_matrix[:,idx]
    end

    wave_function_raw = ifftshift(reciprocal_space_grid, (2, 3, 4))
    reciprocal_space_grid = nothing;

    wave_function = ifft(wave_function_raw, (2, 3, 4))
    wave_function_raw = nothing

    return wave_function
end

"""
    wf_to_G(miller::Matrix{Int32}, wfc, Nxyz) -> Vector{ComplexF64}

Transforms a wave function `wfc` from real space to G-space using the provided Miller indices.

# Arguments
- `miller::Matrix{Int32}`: A 3×N matrix where each column represents a Miller index (G-vector) in reciprocal space.
- `wfc`: The wave function Nxyz×Nxyz×Nxyz in real space, assumed to be a 3D array compatible with FFT operations.
- `Nxyz`: The size of the FFT grid along each dimension (assumed cubic).

# Returns
- `evc_sc::Vector{ComplexF64}`: The wave function coefficients in G-space, normalized.

# Notes
- The function applies an FFT and fftshift to the input wave function, then extracts the coefficients corresponding to the provided Miller indices.
- The output is normalized such that its norm is 1.
"""
function wf_to_G(miller::Matrix{Int32}, wfc, Nxyz)
    Nevc = size(miller, 2)

    evc_sc = zeros(ComplexF64, size(miller, 2))
    wfc_g = fftshift(fft(wfc))
    shift = div.(Nxyz, 2)
    for idx in 1:Nevc
        g_vector = Int.(miller[:, idx])
        i, j, k = ((g_vector .+ shift) .% Nxyz) .+ 1
        evc_sc[idx] = wfc_g[i, j, k]
    end

    #TODO Is it possible to not calculate this norm? Could reduce the computational cost
    norm = sqrt(1/calculate_braket(evc_sc,evc_sc))
    evc_sc = evc_sc .* norm

    return evc_sc
end


"""
    wf_to_G_list(miller::Matrix{Int32}, wfc::AbstractArray{ComplexF64, 4}, Nxyz) -> Matrix{ComplexF64}

Transforms a list of real-space wave functions `wfc` into a list of G-space wave functions.

# Arguments
- `miller::Matrix{Int32}`: A 3×Ng matrix where each column represents the Miller indices (G-vector components) for which the plane-wave coefficients are to be extracted.
- `wfc::AbstractArray{ComplexF64, 4}`: The real-space wave function array with dimensions (Nbands, Nx, Ny, Nz), where Nevc is the number of eigenvectors (bands), and Nx, Ny, Nz are the grid sizes in each spatial direction.
- `Nxyz`: The size of the FFT grid in each spatial direction (assumed cubic).

# Returns
- `evc_sc::Matrix{ComplexF64}`: A matrix of size (Nbands, NG), where each column contains the plane-wave coefficients for the corresponding G-vector from `miller`.

# Notes
- The function assumes that the FFT grid is cubic (Nx = Ny = Nz = Nxyz).
"""
function wf_to_G_list(miller::Matrix{Int32}, wfc::AbstractArray{ComplexF64, 4}, Nxyz)
    Ng = size(miller, 2)
    Nevc = size(wfc, 1)

    evc_sc = zeros(ComplexF64, (Nevc, Ng))
    wfc_g_raw = fft(wfc, (2, 3, 4))
    wfc_g = fftshift(wfc_g_raw, (2, 3, 4))
    wfc_g_raw = nothing

    shift = div.(Nxyz, 2)
    for idx in 1:Ng
        i = (Int(miller[1, idx]) + shift[1]) % Nxyz[1] + 1
        j = (Int(miller[2, idx]) + shift[2]) % Nxyz[2] + 1
        k = (Int(miller[3, idx]) + shift[3]) % Nxyz[3] + 1
        evc_sc[:,idx] = wfc_g[:,i, j, k]
    end

    wfc_g = nothing

    #TODO Is it possible to not calculate this norm? Could reduce the computational cost
    # norm = sqrt(1/calculate_braket(evc_sc,evc_sc))
    # evc_sc = evc_sc .* norm

    return evc_sc
end

"""
    wf_pc_to_sc(wfc, sc_size)

Expands a wave function `wfc` defined in a primitive cell to a supercell of size `sc_size × sc_size × sc_size`.

# Arguments
- `wfc`: The wave function array defined in the primitive cell.
- `sc_size`: The size of the supercell along each spatial dimension.

# Returns
- `wfc_sc`: The wave function array repeated to fill the supercell.
"""
function wf_pc_to_sc(wfc, sc_size)
    wfc_sc = repeat(wfc, outer=(sc_size[1], sc_size[2], sc_size[3]))
    return wfc_sc
end

"""
    determine_fft_grid(path_to_file::String; use_xml::Bool = false) -> Int

Determines the FFT grid size (`Nxyz`) from a given file.

# Arguments
- `path_to_file::String`: Path to the file containing FFT grid information. This can be either an XML file or a plain text output file.
- `use_xml::Bool = false`: If `true`, parses the file as an XML file; otherwise, parses it as a plain text file.

# Returns
- `Nxyz`: The FFT grid size along each dimension.
"""
function determine_fft_grid(path_to_file::String; use_xml::Bool = false)
    Nxyz= [0, 0, 0]
    if use_xml
        # Parse the XML file
        doc = EzXML.readxml(path_to_file)

        # Navigate to the `fft_grid` node
        fft_grid_node = findfirst("/qes:espresso/output/basis_set/fft_grid", root(doc))

        if fft_grid_node === nothing
            error("fft_grid section not found in the XML file.")
        end

        # # Extract the attributes
        nr1 = parse(Int, fft_grid_node["nr1"])
        nr2 = parse(Int, fft_grid_node["nr2"])
        nr3 = parse(Int, fft_grid_node["nr3"])

        Nxyz = [nr1, nr2, nr3]
    else
        scf_file = open(path_to_file, "r")
        fft_line = ""
        for line in eachline(scf_file)
            if contains(line, "FFT dimensions:")
                fft_line = line
                break
            end
        end
        close(scf_file)

        nr1 = parse(Int64, split(fft_line)[8][1:end-1])
        nr2 = parse(Int64, split(fft_line)[9][1:end-1])
        nr3 = parse(Int64, split(fft_line)[10][1:end-1])

        Nxyz = [nr1, nr2, nr3]
    end

    return Nxyz
end

"""
    determine_phase(q_point, Nxyz)

Compute the phase factor `exp(2im * π * dot(r, q_point))` on a 3D grid of size `Nxyz × Nxyz × Nxyz`.

# Arguments
- `q_point::AbstractVector`: A 3-element vector representing the q-point in reciprocal space.
- `Nxyz`: The number of grid points along each spatial dimension.

# Returns
- `exp_factor::Array{Complex{Float64},3}`: A 3D array of complex phase factors evaluated at each grid point.
"""
function determine_phase(q_point, Nxyz)
    x = range(0, 1-1/Nxyz[1], Nxyz[1])
    y = range(0, 1-1/Nxyz[2], Nxyz[2])
    z = range(0, 1-1/Nxyz[3], Nxyz[3])

    exp_factor = zeros(Complex{Float64}, Nxyz[1], Nxyz[2], Nxyz[3])
    @threads for i in eachindex(x)
        for j in eachindex(y)
            for k in eachindex(z)
                x_i = x[i]
                y_j = y[j]
                z_k = z[k]

                r_ijk = [x_i, y_j, z_k]
                temp = dot(transpose(r_ijk), q_point)
                exp_factor[i, j, k] = exp.(2im * π * temp)
            end
        end
    end

    return exp_factor
end

"""
    prepare_unfold_to_sc(path_to_in::String, sc_size::Int, ik::Int)

Prepares and saves the unfolded wave functions for a supercell calculation.

# Arguments
- `path_to_in::String`: Path to the input directory containing required files.
- `sc_size::Int`: Supercell size multiplier.
- `ik::Int`: Index of the k-point to process.

# Description
This function performs the following steps:
1. Determines the q-point vector for the specified k-point.
2. Calculates the phase factor for the q-point and FFT grid.
3. Loads the list of wave functions from a JLD2 file.
4. For each wave function:
    - Converts it from primitive cell to supercell representation.
    - Applies the calculated phase factor.
    - Stores the processed wave function in a new dictionary.
5. Saves the processed wave functions to a new JLD2 file with a modified filename.

# Output
Saves the processed wave functions with phase factors applied to a file named `wfc_list_phase_ik.jld2` in the specified input directory.
"""
function prepare_unfold_to_sc(path_to_in::String, sc_size::Vector{Int}, ik::Int)
    Nxyz = determine_fft_grid(path_to_in*"/scf.out") .* sc_size
    q_vector = determine_q_point(path_to_in, ik; sc_size = sc_size)
    exp_factor = determine_phase(q_vector, Nxyz)

    wfc_list_old = load(path_to_in*"wfc_list_$ik.jld2")
    N_evc = length(wfc_list_old)

    wfc_list = Dict()
    for index in 1:N_evc
        wfc = wfc_list_old["wfc$index"]
        wfc = wf_pc_to_sc(wfc, sc_size)
        wfc = wfc .* exp_factor
        wfc_list["wfc$index"]  = wfc
    end

    @info "Data saved in "*path_to_in*"wfc_list_phase_$ik.jld2"
    save(path_to_in*"wfc_list_phase_$ik.jld2",wfc_list)
end

"""
    wf_phase!(path_to_in::String, wfc::AbstractArray{ComplexF64, 4}, sc_size::Int, ik::Int)

Applies a phase factor to an array of complex wave functions  `wfc` in-place, based on the specified supercell size and k-point index.

# Arguments
- `path_to_in::String`: Path to the input directory containing SCF calculation outputs.
- `wfc::AbstractArray{ComplexF64, 4}`: 4D array Nbands × Nxyz × Nxyz × Nxyz representing the wave functions to be modified.
- `sc_size::Int`: Supercell size multiplier.
- `ik::Int`: Index of the k-point for which the phase factor is determined.

"""
function wf_phase!(path_to_in::String, wfc::AbstractArray{ComplexF64, 4}, sc_size::Vector{Int}, ik::Int)
    Nxyz = determine_fft_grid(path_to_in*"/scf_0/scf.out") .* sc_size
    q_vector = determine_q_point(path_to_in*"/scf_0/", ik; sc_size = [1,1,1], use_sc = true)
    exp_factor = determine_phase(q_vector, Nxyz)

    N_evc = size(wfc)[1]
    for iband in 1:N_evc
        wfc[iband,:,:,:] = wfc[iband,:,:,:] .* exp_factor
    end

    return nothing
end

"""
    prepare_wave_functions_to_R(path_to_in::String; ik::Int=1)

Prepares and saves wave functions in real space for a given k-point index.

# Arguments
- `path_to_in::String`: Path to the input directory containing calculation results.
- `ik::Int=1`: (Optional) Index of the k-point to process. Defaults to 1.

# Output
Saves a dictionary of wave functions to a file named `wfc_list_<ik>.jld2` in the specified input directory.
"""
function prepare_wave_functions_to_R(path_to_in::String; ik::Int=1)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik"
    miller, evc_list = parse_wf(file_path)

    Nxyz = determine_fft_grid(path_to_in*"/scf.out")

    wfc_list = Dict()
    for (index, evc) in enumerate(evc_list)
        wfc = wf_from_G(miller, evc, Nxyz)
        wfc_list["wfc$index"] = wfc
    end

    @info "Data saved in "*path_to_in*"wfc_list_$ik.jld2"
    save(path_to_in*"/wfc_list_$ik.jld2",wfc_list)
    wfc_list = 0 # attempt to free memory
    GC.gc()
end

"""
    prepare_wave_functions_to_G(path_to_in::String; ik::Int=1)

Loads wave function in real space from a specified directory, transforms them to G-space and saves the resulting data.

# Arguments
- `path_to_in::String`: Path to the input directory containing the wave function files.
- `ik::Int=1`: (Optional) Index of the k-point to process.

# Description
This function:
1. Loads the wave function list for the specified k-point from a JLD2 file.
2. Determines the grid size and Miller indices by parsing a reference wave function file.
3. Transforms each wave function in the list to G-space using `wf_to_G`.
4. Saves the resulting dictionary of G-space wave functions to a new JLD2 file.
"""
function prepare_wave_functions_to_G(path_to_in::String; ik::Int=1)
    wfc_list = load(path_to_in*"/scf_0/wfc_list_phase_$ik.jld2")
    Nxyz = size(wfc_list["wfc1"])
    miller_sc, _ = parse_wf(path_to_in*"/group_1/tmp/scf.save/wfc1")

    g_list = Dict()
    for (key, wfc) in wfc_list
        g_list[key] = wf_to_G(miller_sc, wfc, Nxyz)
    end

    @info "Data saved in "*path_to_in*"/scf_0/g_list_sc_$ik.jld2"
    save(path_to_in*"/scf_0/g_list_sc_$ik.jld2", g_list)
end

#Block with new implementation of Folding/Unfolding
function is_within_cutoff(hi, ki, li, kpt, cutoff_radius)#Only for FCC for now
    h = hi + kpt[1]
    k = ki + kpt[2]
    l = li + kpt[3]

    G_squared = 3 * (h^2 + k^2 + l^2) - 2 * (h * k + h * l + k * l)
    return G_squared <= cutoff_radius^2
end

#TODO fix it in the case of inosotropic systems
function create_miller_index(a, Ecut, mesh_scale, kpt)
    # Constants
    a_new =  mean(mesh_scale.*a)  # Lattice constant in Ångstroms #mean to have isotropic system for now
    ecutoff = Ecut  # Kinetic energy cutoff in Ha

    # Reciprocal lattice vector magnitude in Å⁻¹
    g_mag = (2 * π) ./ a_new
    miller_new_raw = []

    # Calculate cutoff radius in reciprocal lattice units
    cutoff_radius = sqrt(2 * ecutoff) / 0.529 / g_mag
    kpt_new = kpt .* mesh_scale

    # Collect points inside the cutoff
    max_index = Int(ceil(cutoff_radius))
    for h in -max_index:max_index
        for k in -max_index:max_index
            for l in -max_index:max_index
                if is_within_cutoff(h, k, l, kpt_new, cutoff_radius)
                    push!(miller_new_raw, [h, k, l])
                end
            end
        end
    end

    miller_new::Matrix{Int32} = hcat(miller_new_raw...)
    return miller_new
end

function create_unified_Grid(path_to_dat, a, ecutoff, mesh_scale )

    ## Create a grid comensurate with SCII
    ## TODO Come up with a better way to determine the cutoff
    ecutoff = ecutoff + 5#65.0/2#(ecutoff + 5) / 2 #??

    miller_list_pc_raw  = [0;0;0]

    for ik in 1:mesh_scale[1] * mesh_scale[2] * mesh_scale[3]
        kpt =  ElectronPhonon.determine_q_point(path_to_dat*"scf_0", ik)
        miller_pc_ik =  create_miller_index(a, ecutoff, mesh_scale, kpt)
        miller_list_pc_raw = hcat(miller_list_pc_raw, miller_pc_ik)
    end

    #conver matrix 3xN to vector of points
    miller_list_pc_raw_points = [miller_list_pc_raw[:,i] for i in 1:size(miller_list_pc_raw, 2)]
    miller_list_pc_raw_unique = hcat(unique(miller_list_pc_raw_points)...)

    # Create a dictionary to map each point to its column index
    miller_map = Dict{Tuple{Int, Int, Int}, Int}()

    # Get the number of columns in the matrix
    num_columns = size(miller_list_pc_raw_unique, 2)

    # Fill the dictionary with points as keys and column indices as values
    for i in 1:num_columns
        point = tuple(miller_list_pc_raw_unique[:, i]...)  # Convert the column to a tuple (x, y, z)
        miller_map[point] = i
    end

    # save miller_map to a file
    save(path_to_dat*"scf_0/miller_list_sc.jld2", "miller_list", miller_list_pc_raw_unique)

    return miller_map
end

function get_unfolded_wf(miller_final_map, miller_pc_ik, wfc_pc_ik, K_init, mesh_scale)
    wfc_sc_ik_shifted = [zeros(ComplexF64, length(miller_final_map)) for _ in 1:size(wfc_pc_ik)[1]]
    K = [Int(round(K_init[1])), Int(round(K_init[2])), Int(round(K_init[3]))]
    # println("K = $K , mesh_scale = $mesh_scale , Grand = $(miller_pc_ik[:, 10]) ")
    for iG in 1:size(miller_pc_ik)[2]#[1:10]

        G = miller_pc_ik[:, iG]
        G_shifted = mesh_scale .* (G) + K
        iG_shifted = get(miller_final_map, (G_shifted[1], G_shifted[2], G_shifted[3]), nothing)

        if iG_shifted !== nothing
            for i in 1:size(wfc_pc_ik)[1]
                wfc_sc_ik_shifted[i][iG_shifted] = wfc_pc_ik[i][iG]
            end
            # wfc_sc_ik_shifted[:][iG_shifted] = wfc_pc_ik[:][iG]
        else
            @warn "Point  $G_shifted  not found in miller_list_final"
        end

    end

    return wfc_sc_ik_shifted
end

function prepare_wave_functions_undisp(path_to_in::String, miller_final_map, ik::Int, mesh_scale::Vector{Int})
    miller_pc_ik, wfc_pc_ik =  ElectronPhonon.parse_wf(path_to_in*"scf_0/tmp/scf.save/wfc$(ik)")
    K = ElectronPhonon.determine_q_point(path_to_in*"scf_0", ik; sc_size = mesh_scale)
    wfc_pc_ik1_unf = get_unfolded_wf(miller_final_map,miller_pc_ik, wfc_pc_ik, K, mesh_scale)

    g_list = Dict()
    for (index, evc) in enumerate(wfc_pc_ik1_unf)
        g_list["wfc$index"] = evc
    end

    @info "Data saved in "*path_to_in*"/scf_0/g_list_sc_$ik.jld2"
    save(path_to_in*"/scf_0/g_list_sc_$ik.jld2", g_list)
end

function prepare_wave_functions_undisp(path_to_in::String, miller_final_map, sc_size::Vector{Int}; k_mesh::Vector{Int} = [1,1,1])
    for ik in 1:prod(sc_size)*prod(k_mesh)
        prepare_wave_functions_undisp(path_to_in,miller_final_map,ik,sc_size.*k_mesh)
        @info "ik = $ik/$(prod(sc_size)*prod(k_mesh)) is ready"
    end
end

function prepare_wave_functions_disp(path_to_in::String, miller_final_map, ik::Int, Ndisplace::Int, mesh_scale::Vector{Int})
    @threads for ind in 1:Ndisplace
        miller_sc_ik, wfc_sc_ik =  ElectronPhonon.parse_wf(path_to_in*"group_$ind/tmp/scf.save/wfc$(ik)")
        K = ElectronPhonon.determine_q_point(path_to_in*"scf_0", ik; sc_size = mesh_scale, use_sc = true)
        wfc_sc_ik1_unf = get_unfolded_wf(miller_final_map, miller_sc_ik, wfc_sc_ik, K, mesh_scale)

        g_list = Dict()
        for (index, evc) in enumerate(wfc_sc_ik1_unf)
            g_list["wfc$index"] = evc
        end

        @info "Data saved in "*path_to_in*"/group_$ind/g_list_sc_$ik.jld2"
        save(path_to_in*"/group_$ind/g_list_sc_$ik.jld2", g_list)
    end
end

function prepare_wave_functions_disp(path_to_in::String, miller_final_map, Ndisplace::Int, k_mesh::Vector{Int})
    for ik in 1:prod(k_mesh)
        prepare_wave_functions_disp(path_to_in, miller_final_map, ik, Ndisplace, k_mesh)
        @info "ik = $ik/$(prod(k_mesh)) is ready"
    end
end
#End of block


"""
    prepare_wave_functions_undisp(path_to_in::String, ik::Int, sc_size::Int)

Prepares wave functions for further calculations by performing the following steps:

1. Transforms wave functions to real (R) space for the specified k-point.
2. Unfolds the wave functions to the supercell of given size.
3. Transforms the wave functions back to reciprocal (G) space.

# Arguments
- `path_to_in::String`: Path to the input directory containing wave function data.
- `ik::Int`: Index of the k-point to process.
- `sc_size::Int`: Size of the supercell for unfolding.
"""
function prepare_wave_functions_undisp(path_to_in::String, ik::Int, sc_size::Vector{Int})
    file_path=path_to_in*"/scf_0/"
    @info "Tranforming wave functions to R space:"
    prepare_wave_functions_to_R(file_path;ik=ik)
    @info "Unfolding wave functions to supercell:"
    prepare_unfold_to_sc(file_path,sc_size,ik)
    @info "Tranforming wave functions to G space:"
    prepare_wave_functions_to_G(path_to_in;ik=ik)
end

function prepare_wave_functions_undisp(path_to_in::String, sc_size::Vector{Int}; k_mesh::Vector{Int} = [1,1,1])
    for ik in 1:prod(sc_size)*prod(k_mesh)
        prepare_wave_functions_undisp(path_to_in,ik,sc_size)
        @info "ik = $ik/$(prod(sc_size)*prod(k_mesh)) is ready"
    end
end

"""
    prepare_wave_functions_disp(path_to_in::String, ik::Int, Ndisplace::Int, sc_size::Int, k_mesh::Int)

Prepares and processes wave functions for a set of displaced structures in a supercell calculation.

# Arguments
- `path_to_in::String`: Path to the input directory containing data for each displacement group.
- `ik::Int`: Index of the k-point for which the wave functions are prepared.
- `Ndisplace::Int`: Number of displacement groups to process.
- `sc_size::Int`: Size of the supercell (number of unit cells along one dimension).
- `k_mesh::Int`: Number of k-points in the mesh (currently unused in the function).

# Description
For each displacement group, this function:
1. Parses the wave function coefficients from the corresponding data files.
2. Processes the wave functions in chunks to manage memory usage:
    - Converts G-space coefficients to real-space wave functions.
    - Applies phase factor.
    - Converts the corrected wave functions back to G-space.
3. Saves the phase-corrected wave function coefficients to a JLD2 file for each displacement group.
"""
function prepare_wave_functions_disp(path_to_in::String, ik::Int, Ndisplace::Int, sc_size::Vector{Int}, k_mesh::Vector{Int})

    @threads for ind in 1:Ndisplace
        path_to_data = path_to_in*"group_$ind/"
        miller, evc_list_sc = parse_wf(path_to_data*"tmp/scf.save/wfc$ik")
        N_evc = size(evc_list_sc)[1]
        N_g   = length(evc_list_sc[1])
        N = determine_fft_grid(path_to_data*"tmp/scf.save/data-file-schema.xml"; use_xml = true)
        println("N = $N")

        wave_function_result = Array{ComplexF64, 4}(undef, N_evc, N, N, N)
        evc_list_phase = Array{ComplexF64, 2}(undef, N_evc, N_g)

        Nchunk = prod(sc_size)
        chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]
        evc_chunks = chunk(evc_list_sc[1:N_evc], Nchunk)

        for (index, evc_list_chunk) in enumerate(evc_chunks)
            wave_function_result[(index-1)*Nchunk+1:index*Nchunk,:,:,:] = wf_from_G_list(miller, evc_list_chunk, N)
            GC.gc()

            wf_phase!(path_to_in, wave_function_result[(index-1)*Nchunk+1:index*Nchunk,:,:,:], sc_size, ik)

            evc_list_phase[(index-1)*Nchunk+1:index*Nchunk,:] = wf_to_G_list(miller, wave_function_result[(index-1)*Nchunk+1:index*Nchunk,:,:,:], N)
            GC.gc()
        end

        #save resulting evc list
        save(path_to_data*"/evc_list_sc_$ik.jld2", "evc_list_phase", evc_list_phase)
        @info "idisp = $ind/$(Ndisplace) is ready"
    end
end

"""
    prepare_wave_functions_disp(path_to_in::String, Ndisplace::Int, sc_size::Int, k_mesh::Int)
A wrapper function that calls another method of `prepare_wave_functions_disp` with an additional k-point index argument.

# Arguments
- `path_to_in::String`: Path to the input directory or file.
- `Ndisplace::Int`: Number of displacements to consider.
- `sc_size::Int`: Size of the supercell.
- `k_mesh::Int`: Number of k-points along each reciprocal lattice direction.
"""
function prepare_wave_functions_disp(path_to_in::String, Ndisplace::Int, sc_size::Vector{Int}, k_mesh::Vector{Int})
    for ik in 1:prod(k_mesh)
        prepare_wave_functions_disp(path_to_in,ik, Ndisplace, sc_size, k_mesh)
        @info "ik = $ik/$(k_mesh^3) is ready"
    end
end

function prepare_u_matrixes(path_to_in::String, natoms::Int, sc_size::Vector{Int}, k_mesh::Vector{Int}; symmetries::Symmetries = Symmetries([], [], []), save_matrixes::Bool=true)
    U_list = []
    V_list = []

    Ndisplace_nosym = 6 * natoms

    N_fft = 1
    if any(sc_size .!= 1)
        N_fft = determine_fft_grid(path_to_in*"group_1/tmp/scf.save/data-file-schema.xml"; use_xml = true).* k_mesh
    else
        N_fft = determine_fft_grid(path_to_in*"group_1/tmp/scf.save/data-file-schema.xml"; use_xml = true)
    end

    ψₚ0_real_list = []
    miller_list = []
    use_symm = isempty(symmetries.trans_list) && isempty(symmetries.rot_list)

    for (ind, _) in enumerate(unique(symmetries.ineq_atoms_list))
        ψₚ0_real_ip = []
        miller_ip = []
        for ip in 1:prod(k_mesh)
            if use_symm && any(k_mesh .!= 1)
                miller1 = load(path_to_in*"/scf_0/miller_list_sc.jld2")["miller_list"]
                ψₚ0_list_raw = load(path_to_in*"/group_$ind/g_list_sc_$ip.jld2")
                ψₚ0_list = [ψₚ0_list_raw["wfc$iband"] for iband in 1:length(ψₚ0_list_raw)]

                ψₚ0_real_phase = [wf_from_G(miller1, evc, N_fft) for evc in ψₚ0_list]
                K = determine_q_point(path_to_in*"/scf_0", ip; sc_size = k_mesh, use_sc = true)
                ψₚ0_real = [wf .* conj(determine_phase(K, N_fft)) for wf in ψₚ0_real_phase]
                
                # miller1, ψₚ0 = parse_wf(path_to_in*"/group_$ind/tmp/scf.save/wfc$ip")
                # ψₚ0_real = [wf_from_G(miller1, evc, N_fft) for evc in ψₚ0]
            else
                miller1, ψₚ0 = parse_wf(path_to_in*"/group_$ind/tmp/scf.save/wfc$ip")
                ψₚ0_real = [wf_from_G(miller1, evc, N_fft) for evc in ψₚ0]
            end
            push!(ψₚ0_real_ip, ψₚ0_real)
            push!(miller_ip, miller1)
        end
        push!(ψₚ0_real_list, ψₚ0_real_ip)
        push!(miller_list, miller_ip)
    end

    nbnds = length(parse_wf(path_to_in*"/scf_0/tmp/scf.save/wfc1")[2])
    
    kpoints = []
    if any(sc_size .!= 1)
        kpoints = [determine_q_point(path_to_in*"/scf_0",ik; sc_size=k_mesh, use_sc = true) for ik in 1:prod(k_mesh)]
    else
        kpoints = [determine_q_point(path_to_in*"/scf_0",ik) for ik in 1:prod(k_mesh)]
    end
    println("nbnds = $nbnds")

    println("Preparing u matrixes:")
    for ind in 1:Ndisplace_nosym
        ψₚ = []
        local tras, rot

        #check if symmetries are empty
        if use_symm
            tras = [0.0,0.0,0.0]
            rot  = [[1.0,0.0,0.0] [0.0,1.0,0.0] [0.0,0.0,1.0]]
            ind_k_list = [1:prod(k_mesh)]
            append!(symmetries.ineq_atoms_list, ind)
        else
            tras  = symmetries.trans_list[ind] #./sc_size

            if any(sc_size .!= 1) #TODO understand corner case with sc size  and k_mesh != 1
                tras = tras ./ k_mesh
            end

            rot   = symmetries.rot_list[ind]
            ind_k_list = symmetries.ind_k_list[ind]
        end

        Uₚₖᵢⱼ = zeros(ComplexF64, prod(k_mesh), prod(k_mesh)*prod(sc_size), nbnds*prod(sc_size), nbnds)

        for ip in 1:prod(k_mesh)
            if all(isapprox.(tras,[0.0,0.0,0.0], atol = 1e-15)) &&
            all(isapprox.(rot, [[1.0,0.0,0.0] [0.0,1.0,0.0] [0.0,0.0,1.0]], atol = 1e-15))
                if any(sc_size .!= 1) && any(k_mesh .!= 1)
                    ψₚ_list = load(path_to_in*"/group_$(symmetries.ineq_atoms_list[ind])/g_list_sc_$ip.jld2")
                    ψₚ = [ψₚ_list["wfc$iband"] for iband in 1:length(ψₚ_list)]
                else
                    _, ψₚ = parse_wf(path_to_in*"/group_$(symmetries.ineq_atoms_list[ind])/tmp/scf.save/wfc$ip")
                end
            else
                ψₚ0_real = ψₚ0_real_list[symmetries.ineq_atoms_list[ind]][ind_k_list[ip]]
                miller1 = miller_list[symmetries.ineq_atoms_list[ind]][ip]
                map1 = rotate_grid(N_fft[1], N_fft[2], N_fft[3], rot, tras)
                ψₚ_real = [rotate_deriv(N_fft[1], N_fft[2], N_fft[3], map1, wfc) for wfc in ψₚ0_real]

                # in case of symmetries with kpoints need to multiply by a phase factor 
                kpoint_rotated = transpose(inv(rot)) * kpoints[ind_k_list[ip]]
                phase_in = determine_phase(kpoint_rotated, N_fft)
                phase_out = 1.0 

                if all(sc_size .== 1)
                    phase_out = determine_phase(kpoints[ip], N_fft) 
                end

                ψₚ_real = [wf .* phase_in .* conj(phase_out) for wf in ψₚ_real]
                ψₚ = [wf_to_G(miller1, evc, N_fft) for evc in ψₚ_real]

                # DEBUG save the transformed wave functions
                # ψₚ_list =Dict()
                # for (iband, wfc) in enumerate(ψₚ)
                #     ψₚ_list["wfc$iband"] = wfc
                # end
                # save("/home/poliukhin/Development/ElectronPhonon/example/tst/si_k_sc_symm/displacements/tmp_check/"*"/group_$(ind)/g_list_sc_$ip.jld2", ψₚ_list)

            end

            for ik in 1:prod(sc_size)*prod(k_mesh)
                if any(sc_size .!= 1)
                    ψkᵤ_list = load(path_to_in*"/scf_0/g_list_sc_$ik.jld2")
                    ψkᵤ = [ψkᵤ_list["wfc$iband"] for iband in 1:length(ψkᵤ_list)]
                else
                    _, ψkᵤ = parse_wf(path_to_in*"scf_0/tmp/scf.save/wfc$ik")
                end

                if all(sc_size .== 1) && ik != ip #orthogonality in unitcell at different k-points
                    Uₚₖᵢⱼ[ip, ik, :, :] .= 0.0
                else
                    Uₚₖᵢⱼ[ip, ik, :, :] = calculate_braket(ψₚ, ψkᵤ)
                end
                @info ("idisp = $(ind), ik = $ik")
            end

            @info ("ik_sc = $ip is ready")
        end

        if isodd(ind)
            push!(U_list, Uₚₖᵢⱼ)
        else
            push!(V_list, Uₚₖᵢⱼ)
        end
        @info ("group_$ind is ready")
    end

    # Save U_list to a hdf5-like file
    if save_matrixes == true
        save(path_to_in * "scf_0/U_list.jld2", "U_list", U_list)
        save(path_to_in * "scf_0/V_list.jld2", "V_list", V_list)
    end

    return U_list, V_list
end

"""
    calculate_braket_real(bra::Array{Complex{Float64}, 3}, ket::Array{Complex{Float64}, 3}) -> Complex{Float64}

Calculates the normalized inner product ⟨bra|ket⟩ between two 3D complex-valued arrays `bra` and `ket`.

# Arguments
- `bra::Array{Complex{Float64}, 3}`: The "bra" vector, represented as a 3D array of complex numbers.
- `ket::Array{Complex{Float64}, 3}`: The "ket" vector, represented as a 3D array of complex numbers.

# Returns
- `Complex{Float64}`: The normalized inner product ⟨bra|ket⟩, computed as the sum over all elements of `conj(bra[i]) * ket[i]`, divided by the total number of elements.
"""
function calculate_braket_real(bra::Array{Complex{Float64}, 3}, ket::Array{Complex{Float64}, 3})
    Nxyz = size(ket)
    result = zero(Complex{Float64})

    @inbounds @simd for i in 1:prod(Nxyz)
        result += conj(bra[i]) * ket[i]
    end

    result /= prod(Nxyz)
    return result
end

"""
    calculate_braket_real(bras::Dict{String, Array{Complex{Float64}, 3}}, kets::Dict{String, Array{Complex{Float64}, 3}}) -> Array{Complex{Float64}, 2}

Compute the matrix of real-valued inner product ⟨bra|ket⟩ values between sets of "bra" and "ket" wave functions.

# Arguments
- `bras`: A dictionary mapping string identifiers to 3-dimensional arrays of complex numbers, representing the "bra" wave functions.
- `kets`: A dictionary mapping string identifiers to 3-dimensional arrays of complex numbers, representing the "ket" wave functions.

# Returns
- A 2D array of complex numbers where the element at position `(i, j)` contains the result of `calculate_braket_real` applied to the `i`-th bra and `j`-th ket.
"""
function calculate_braket_real(bras::Dict{String, Array{Complex{Float64}, 3}}, kets::Dict{String, Array{Complex{Float64}, 3}})
    result = zeros(Complex{Float64}, length(bras), length(kets))

    for (i, bra) in enumerate(values(bras))
        for (j, ket) in enumerate(values(kets))
            result[i,j] = calculate_braket_real(bra, ket)
        end
    end

    return result
end

"""
    calculate_braket(bra::Array{Complex{Float64}}, ket::Array{Complex{Float64}}) -> Complex{Float64}

Computes the inner product ⟨bra|ket⟩ between two complex-valued vectors.

# Arguments
- `bra::Array{Complex{Float64}}`: The "bra" vector (conjugate transpose).
- `ket::Array{Complex{Float64}}`: The "ket" vector.

# Returns
- `Complex{Float64}`: The resulting inner product as a complex number.
"""
function calculate_braket(bra::Array{Complex{Float64}}, ket::Array{Complex{Float64}})
    Nevc = length(bra)
    result = zero(Complex{Float64})

    @inbounds @simd for i in 1:Nevc
        result += conj(bra[i]) * ket[i]
    end

    return result
end

"""
    calculate_braket(bras, kets)

Compute the matrix of inner products ⟨bra|ket⟩ between two collections of quantum states.

# Arguments
- `bras`: A collection (e.g., array or vector) of bra vectors (quantum states).
- `kets`: A collection (e.g., array or vector) of ket vectors (quantum states).

# Returns
- A matrix of complex numbers where the element at position (i, j) is the inner product between `bras[i]` and `kets[j]`.

"""
function calculate_braket(bras, kets)
    result = zeros(Complex{Float64}, length(bras), length(kets))

    @threads for i in eachindex(bras)
        for j in eachindex(kets)
            result[i,j] = calculate_braket(bras[i],kets[j])
        end
    end

    return result
end


#TEST
#path_to_in = "/home/apolyukhin/Development/julia_tests/qe_inputs/displacements/"
# path_to_in = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/group_1/tmp_001/silicon.save"
# N = 72
# ik = 1
# prepare_wave_functions_opt(path_to_in, ik,N)
