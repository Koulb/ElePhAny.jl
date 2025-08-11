"""
    check_symmetries(path_to_calc, unitcell, sc_size, abs_disp)

Checks the symmetries of atomic displacements in a crystal structure using Phonopy.

# Arguments
- `path_to_calc::String`: Path to the calculation directory where displacement files will be saved.
- `unitcell::Dict`: Dictionary containing unit cell information with keys:
    - `:symbols`: Atomic symbols.
    - `:cell`: Lattice vectors.
    - `:scaled_positions`: Atomic positions in scaled coordinates.
    - `:masses`: Atomic masses.
- `sc_size::Int`: Size of the supercell (applied equally along all axes).
- `abs_disp::Float64`: Magnitude of the atomic displacement to generate.

# Returns
- `Symmetries`: An object containing:
    - List of inequivalent atom indices.
    - List of translation vectors for each symmetry operation.
    - List of rotation matrices for each symmetry operation.
"""
function check_symmetries(path_to_calc, unitcell, sc_size, k_mesh, abs_disp)
    unitcell_phonopy = phonopy.structure.atoms.PhonopyAtoms(;symbols=unitcell[:symbols],
    cell=pylist(pyconvert(Array,unitcell[:cell])./bohr_to_ang),#Should be in Bohr, hence conversion
    scaled_positions=unitcell[:scaled_positions],
    masses=unitcell[:masses])
    supercell_matrix=pylist([[sc_size[1], 0, 0], [0, sc_size[2], 0], [0, 0, sc_size[3]]])

    phonon_symm = phonopy.Phonopy(unitcell_phonopy,supercell_matrix=supercell_matrix)
    phonon_nosymm = phonopy.Phonopy(unitcell_phonopy, is_symmetry=false,supercell_matrix=supercell_matrix)
    symm_ops = phonon_symm.symmetry.get_symmetry_operations()
    scaled_pos = phonon_symm.supercell.scaled_positions

    phonon_symm.generate_displacements(distance=abs_disp)
    phonon_nosymm.generate_displacements(distance=abs_disp)
    phonon_nosymm.save(path_to_calc*"displacements/phonopy_params_nosym.yaml")

    scaled_pos = pyconvert(Vector{Vector{Float64}},scaled_pos)
    Uᶜʳʸˢᵗ = pyconvert(Matrix{Float64},phonon_symm.supercell.cell)

    dataˢʸᵐ = pyconvert(Vector{Vector{Float64}},phonon_symm.get_displacements())
    dRˢʸᵐ = [round.(transpose(Uᶜʳʸˢᵗ^-1) * vec[2:4], digits=16) for vec in dataˢʸᵐ]
    Rˢʸᵐ  = [scaled_pos[convert(Int64, vec[1])+1] for vec in dataˢʸᵐ]
    Ndisplace_symm = length(Rˢʸᵐ)

    dataⁿᵒˢʸᵐ = pyconvert(Vector{Vector{Float64}},phonon_nosymm.get_displacements())
    dRⁿᵒˢʸᵐ = [round.(transpose(Uᶜʳʸˢᵗ^-1) * vec[2:4], digits=16) for vec in dataⁿᵒˢʸᵐ]
    Rⁿᵒˢʸᵐ  = [scaled_pos[convert(Int64, vec[1])+1] for vec in dataⁿᵒˢʸᵐ]

    use_sc = false
    if any(sc_size .!= 1)
        use_sc = true
    end

    kpoints = [determine_q_point(path_to_calc*"displacements/scf_0",ik; use_sc = use_sc) for ik in 1:prod(k_mesh)]

    trans_list = []
    rot_list   = []
    ineq_atoms_list = []
    ind_k_list = []
    index = 1

    inosym = 1
    isym   = 1
    while inosym <= length(Rⁿᵒˢʸᵐ)
        R2 = Rⁿᵒˢʸᵐ[inosym] + dRⁿᵒˢʸᵐ[inosym]
        check  = true
        isym = 1
        while check == true && isym <= length(Rˢʸᵐ)
            R1 = Rˢʸᵐ[isym] + dRˢʸᵐ[isym]
            ind_sym = 1
            for (tras_py, rot_py) in zip(symm_ops["translations"], symm_ops["rotations"])
                trans = pyconvert(Vector{Float64}, tras_py)
                rot = pyconvert(Matrix{Float64}, rot_py)
                rotR1 = fold_component.(rot * R1 .+ trans)
                ind_sym += 1

                if all(abs.(R2 - rotR1) .< 1e-8)
                    @info "Found symmetry $index out of $(length(Rⁿᵒˢʸᵐ))"
                    index += 1
                    @info "translation: $trans"
                    @info "rotation   : $rot"
                    push!(trans_list, trans)
                    push!(rot_list, rot)
                    push!(ineq_atoms_list, isym)

                    #saving k points ind list
                    kpoints_rotated = [transpose(inv(rot)) * k_point for k_point in kpoints]  
                    ind_k_point = find_matching_qpoints(kpoints, kpoints_rotated)
                    push!(ind_k_list, ind_k_point)

                    check = false
                    break
                end
            end
            isym += 1
        end
        inosym += 1
    end
    return Symmetries(ineq_atoms_list, trans_list, rot_list, ind_k_list), Ndisplace_symm
end

"""
    check_symmetries!(model::AbstractModel)

Checks the symmetries of the given `model` and updates its symmetry-related fields in-place.

# Arguments
- `model::AbstractModel`: The model object containing calculation path, unit cell, supercell size, and displacement information.
"""
function check_symmetries!(model::AbstractModel)
    symmetries, Ndisplace_symm = check_symmetries(model.path_to_calc, model.unitcell, model.sc_size, model.k_mesh, model.abs_disp)
    model.Ndispalce = Ndisplace_symm #length(unique(symmetries.ineq_atoms_list))

    if model.Ndispalce != length(unique(symmetries.ineq_atoms_list))
        # model.use_symm = false
        @error "Not all the symmmetries for EP were found, only phonons could be calculated"
    else
        # model.use_symm = true
        model.symmetries = symmetries
    end

end

function find_matching_qpoints(q_ph, q_nscf)
    iq_ph_list = Int[]
    for i_ph in eachindex(q_ph)
        for i_nscf in eachindex(q_nscf)
            q_nscf_crystal = q_nscf[i_nscf]
            q_ph_crystal = q_ph[i_ph]
            delta_q_all = abs.(q_nscf_crystal .- q_ph_crystal)
            check = falses(3)
            for (ind_q, delta_q) in enumerate(delta_q_all)
                if isapprox(delta_q, 0; atol=1e-5) || isapprox(delta_q, 1; atol=1e-5)#|| isapprox(delta_q, 2; atol=1e-5)
                    check[ind_q] = true
                end
            end
            if all(check)
                push!(iq_ph_list, i_nscf)
                break
            end
        end
    end

    if length(iq_ph_list) != length(q_ph)
        println("No all q-points found")
    end

    return iq_ph_list
end

"""
    fold_component(x, eps=5e-3)

Folds the input value `x` into the interval `[0, 1)` within a tolerance `eps`.

If `x` is greater than or equal to `1 - eps`, repeatedly subtracts 1 until `x` falls within the interval.
If `x` is less than `0 - eps`, repeatedly adds 1 until `x` falls within the interval.

# Arguments
- `x`: The value to be folded.
- `eps`: (optional) Tolerance for the interval boundaries. Default is `5e-3`.
"""
function fold_component(x, eps=5e-3)
    if x >= 1 - eps
        while x >= 1 - eps
            x = x - 1
        end
    elseif x < 0 - eps
        while x < 0 - eps
            x = x + 1
        end
    end
    return x
end

# Optimized vectorized version of fold_component
function fold_component_vectorized(x::AbstractArray, eps=5e-3)
    result = similar(x)
    @inbounds @simd for i in eachindex(x)
        result[i] = fold_component(x[i], eps)
    end
    return result
end

# Cache for grid mappings to avoid repeated calculations
const GRID_MAPPING_CACHE = Dict{Tuple{Int,Int,Int,Vector{Float64},Matrix{Float64}}, Vector{Int}}()

"""
    rotate_grid_optimized(N1, N2, N3, rot, tras)

Optimized version of rotate_grid with caching and vectorization.

# Arguments
- `N1::Int`: Number of grid points along the first axis.
- `N2::Int`: Number of grid points along the second axis.
- `N3::Int`: Number of grid points along the third axis.
- `rot::AbstractMatrix`: 3x3 rotation matrix to apply to each grid point.
- `tras::AbstractVector`: 3-element translation vector to apply after rotation.

# Returns
- `mapp::Vector{Int}`: A vector containing the mapped linear indices.
"""
function rotate_grid_optimized(N1, N2, N3, rot, tras)
    cache_key = (N1, N2, N3, tras, rot)
    
    if haskey(GRID_MAPPING_CACHE, cache_key)
        return GRID_MAPPING_CACHE[cache_key]
    end
    
    # Pre-allocate result vector
    total_points = N1 * N2 * N3
    mapp = Vector{Int}(undef, total_points)
    
    # Vectorized grid generation
    i_coords = repeat(0:N1-1, outer=N2*N3)
    j_coords = repeat(repeat(0:N2-1, outer=N3), outer=N1)
    k_coords = repeat(0:N3-1, outer=N1*N2)
    
    # Normalize coordinates
    u_coords = hcat(i_coords ./ N1, j_coords ./ N2, k_coords ./ N3)
    
    # Apply rotation and translation
    ru_coords = u_coords * transpose(rot) .+ transpose(tras)
    
    # Fold coordinates
    ru_coords = fold_component_vectorized(ru_coords)
    
    # Convert back to grid indices
    i1_coords = round.(Int, ru_coords[:, 1] .* N1)
    i2_coords = round.(Int, ru_coords[:, 2] .* N2)
    i3_coords = round.(Int, ru_coords[:, 3] .* N3)
    
    # Bounds checking
    @inbounds for idx in 1:total_points
        i1, i2, i3 = i1_coords[idx], i2_coords[idx], i3_coords[idx]
        
        # Ensure indices are within bounds
        i1 = max(0, min(i1, N1-1))
        i2 = max(0, min(i2, N2-1))
        i3 = max(0, min(i3, N3-1))
        
        mapp[idx] = i1 + i2 * N1 + i3 * N1 * N2
    end
    
    # Cache the result
    GRID_MAPPING_CACHE[cache_key] = mapp
    return mapp
end

"""
    rotate_grid(N1, N2, N3, rot, tras)

Maps a 3D grid of points onto itself under a given rotation and translation, returning a mapping of indices.

# Arguments
- `N1::Int`: Number of grid points along the first axis.
- `N2::Int`: Number of grid points along the second axis.
- `N3::Int`: Number of grid points along the third axis.
- `rot::AbstractMatrix`: 3x3 rotation matrix to apply to each grid point.
- `tras::AbstractVector`: 3-element translation vector to apply after rotation (in fractional coordinates).

# Returns
- `mapp::Vector{Int}`: A vector containing the mapped linear indices for each grid point after applying the rotation and translation.
"""
function rotate_grid(N1, N2, N3, rot, tras)
    # Use optimized version for better performance
    return rotate_grid_optimized(N1, N2, N3, rot, tras)
end

"""
    rotate_deriv_optimized(N1, N2, N3, mapp, ff)

Optimized version of rotate_deriv with better memory access patterns and vectorization.

# Arguments
- `N1::Int`: Size of the first dimension.
- `N2::Int`: Size of the second dimension.
- `N3::Int`: Size of the third dimension.
- `mapp::Vector{Int}`: Mapping array that specifies the new indices for rotation.
- `ff::Array{ComplexF64,3}`: The original 3D array to be rotated.

# Returns
- `ff_rot::Array{ComplexF64,3}`: The rotated 3D array.
"""
function rotate_deriv_optimized(N1, N2, N3, mapp, ff)
    ff_rot = zeros(ComplexF64, N1, N2, N3)
    
    # Pre-compute index mappings for better cache locality
    total_points = N1 * N2 * N3
    
    @inbounds for idx in 1:total_points
        # Source indices (i, j, k)
        k = div(idx - 1, N1 * N2)
        j = div((idx - 1) % (N1 * N2), N1)
        i = (idx - 1) % N1
        
        # Target index from mapping
        target_idx = mapp[idx]
        target_k = div(target_idx, N1 * N2)
        target_j = div(target_idx % (N1 * N2), N1)
        target_i = target_idx % N1
        
        # Copy with bounds checking
        if 0 <= target_i < N1 && 0 <= target_j < N2 && 0 <= target_k < N3
            ff_rot[target_i+1, target_j+1, target_k+1] = ff[i+1, j+1, k+1]
        end
    end
    
    return ff_rot
end

"""
    rotate_deriv(N1, N2, N3, mapp, ff)

Rotate a 3D array `ff` according to a mapping array `mapp` and the specified dimensions `N1`, `N2`, `N3`.

# Arguments
- `N1::Int`: Size of the first dimension.
- `N2::Int`: Size of the second dimension.
- `N3::Int`: Size of the third dimension.
- `mapp::Vector{Int}`: Mapping array that specifies the new indices for rotation. Each entry maps a linear index in the original array to a new position.
- `ff::Array{ComplexF64,3}`: The original 3D array to be rotated.

# Returns
- `ff_rot::Array{ComplexF64,3}`: The rotated 3D array.
"""
function rotate_deriv(N1, N2, N3, mapp, ff)
    # Use optimized version for better performance
    return rotate_deriv_optimized(N1, N2, N3, mapp, ff)
end

# MPI-optimized symmetry operations
function check_symmetries_mpi(path_to_calc, unitcell, sc_size, k_mesh, abs_disp)
    """MPI-parallelized symmetry checking"""
    
    if is_master()
        symmetries, Ndisplace_symm = check_symmetries(path_to_calc, unitcell, sc_size, k_mesh, abs_disp)
        
        # Broadcast results to all ranks
        mpi_bcast!(symmetries.ineq_atoms_list)
        mpi_bcast!(symmetries.trans_list)
        mpi_bcast!(symmetries.rot_list)
        mpi_bcast!(symmetries.ind_k_list)
        mpi_bcast!([Ndisplace_symm])
        
        return symmetries, Ndisplace_symm
    else
        # Receive results from master
        symmetries = Symmetries([], [], [], [])
        mpi_bcast!(symmetries.ineq_atoms_list)
        mpi_bcast!(symmetries.trans_list)
        mpi_bcast!(symmetries.rot_list)
        mpi_bcast!(symmetries.ind_k_list)
        Ndisplace_symm = mpi_bcast!([0])[1]
        
        return symmetries, Ndisplace_symm
    end
end

# Clear cache function for memory management
function clear_symmetry_cache()
    """Clear the grid mapping cache to free memory"""
    empty!(GRID_MAPPING_CACHE)
    GC.gc()
end

# Batch processing for multiple symmetry operations
function apply_symmetries_batch(ff_list::Vector{Array{ComplexF64,3}}, N1, N2, N3, rot, tras)
    """Apply the same symmetry operation to multiple arrays efficiently"""
    
    # Get mapping once
    mapp = rotate_grid_optimized(N1, N2, N3, rot, tras)
    
    # Apply to all arrays
    result = Vector{Array{ComplexF64,3}}(undef, length(ff_list))
    
    @threads for i in 1:length(ff_list)
        result[i] = rotate_deriv_optimized(N1, N2, N3, mapp, ff_list[i])
    end
    
    return result
end

# Optimized phase factor calculation
function determine_phase_optimized(kpoint, N_fft)
    """Optimized phase factor calculation with vectorization"""
    
    # Pre-compute grid coordinates
    N1, N2, N3 = N_fft
    total_points = N1 * N2 * N3
    
    # Vectorized grid generation
    i_coords = repeat(0:N1-1, outer=N2*N3)
    j_coords = repeat(repeat(0:N2-1, outer=N3), outer=N1)
    k_coords = repeat(0:N3-1, outer=N1*N2)
    
    # Calculate phase factors vectorized
    phase_factors = exp.(2π * im * (i_coords .* kpoint[1] ./ N1 .+ 
                                   j_coords .* kpoint[2] ./ N2 .+ 
                                   k_coords .* kpoint[3] ./ N3))
    
    # Reshape to 3D array
    return reshape(phase_factors, N1, N2, N3)
end
