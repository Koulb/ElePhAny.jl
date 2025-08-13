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
    symm_ops = phonon_symm.symmetry.get_symmetry_operations()
    scaled_pos = phonon_symm.supercell.scaled_positions

    phonon_symm.generate_displacements(distance=abs_disp)

    scaled_pos = pyconvert(Vector{Vector{Float64}},scaled_pos)
    Uᶜʳʸˢᵗ = pyconvert(Matrix{Float64},phonon_symm.supercell.cell)

    dataˢʸᵐ = pyconvert(Vector{Vector{Float64}},phonon_symm.get_displacements())
    dRˢʸᵐ = [round.(transpose(Uᶜʳʸˢᵗ^-1) * vec[2:4], digits=16) for vec in dataˢʸᵐ]
    Rˢʸᵐ  = [scaled_pos[convert(Int64, vec[1])+1] for vec in dataˢʸᵐ]
    Ndisplace_symm = length(Rˢʸᵐ)

    use_sc = false
    if any(sc_size .!= 1)
        use_sc = true
    end

    kpoints = [determine_q_point(path_to_calc*"displacements/scf_0",ik; use_sc = use_sc) for ik in 1:prod(k_mesh)]
    Ndisplace_nosym = length(scaled_pos) * 6
    R_nosym = repeat(scaled_pos, inner=6)

    trans_list = Vector{Vector{Float64}}(undef, Ndisplace_nosym)
    rot_list   = Vector{Matrix{Float64}}(undef, Ndisplace_nosym)
    ineq_atoms_list = Vector{Int}(undef, Ndisplace_nosym)
    ind_k_list = Vector{Vector{Int}}(undef, Ndisplace_nosym)

    dR_nosym = Vector{Vector{Float64}}()
    index = 1

    for inosym in 1:length(scaled_pos)
        dR_candidates = Vector{Vector{Float64}}()
        check = true
        isym = 1
        R2 = scaled_pos[inosym] 

        while check && isym <= length(Rˢʸᵐ)
            R1 = Rˢʸᵐ[isym]
            found = false

            for (tras_py, rot_py) in zip(symm_ops["translations"], symm_ops["rotations"])
                trans = pyconvert(Vector{Float64}, tras_py)
                rot   = pyconvert(Matrix{Float64}, rot_py)

                # only consider operations that transfrom R1 to R2
                if all(abs.(R2 .- ElectronPhonon.fold_component.(rot*R1 .+ trans)) .< 1e-8)
                    dR_tmp = ElectronPhonon.fold_component.(rot * dRˢʸᵐ[isym])

                    # only add it if it increases the rank of the candidate set
                    M_tmp = hcat(dR_candidates..., dR_tmp)
                    if rank(M_tmp; atol = 1e-8) == length(dR_candidates) + 1
                        ind_plus = 2*index-1

                        @info "Found symmetry $(ind_plus) out of $(6*length(scaled_pos))"
                        @info "translation: $trans"
                        @info "rotation   : $rot"
                        push!(dR_candidates, dR_tmp)
                        rot_list[ind_plus] = rot
                        trans_list[ind_plus] = trans
                        ineq_atoms_list[ind_plus] = isym

                        #saving k points ind list
                        kpoints_rotated = [transpose(inv(rot)) * k_point for k_point in kpoints]  
                        ind_k_point = ElectronPhonon.find_matching_qpoints(kpoints, kpoints_rotated)
                        ind_k_list[ind_plus] = ind_k_point # minus displacement
                        index += 1
                    end

                    # if we've now got 3, mark for exit
                    if length(dR_candidates) == 3
                        check = false
                        found = true
                        break  
                    end
                end
            end

            if found
                break
            end
        
            isym += 1
        end
        append!(dR_nosym, [dR_candidates[1],-dR_candidates[1], dR_candidates[2], -dR_candidates[2], dR_candidates[3], -dR_candidates[3]])
    end

    inosym = 1
    isym   = 1
    while inosym <= length(R_nosym)
        #skip positive displacements
        if isodd(inosym)
            inosym += 1
        continue
        end

        R2 = R_nosym[inosym] + dR_nosym[inosym]
        check  = true
        isym = 1
        while check == true && isym <= length(Rˢʸᵐ)
            R1 = Rˢʸᵐ[isym] + dRˢʸᵐ[isym]
            for (tras_py, rot_py) in zip(symm_ops["translations"], symm_ops["rotations"])
                trans = pyconvert(Vector{Float64}, tras_py)
                rot = pyconvert(Matrix{Float64}, rot_py)
                rotR1 = ElectronPhonon.fold_component.(rot * R1 .+ trans)

                if all(abs.(R2 - rotR1) .< 1e-8)
                    @info "Found symmetry $inosym out of $(length(R_nosym))"
                    @info "translation: $trans"
                    @info "rotation   : $rot"
                    trans_list[inosym] = trans
                    rot_list[inosym] = rot
                    ineq_atoms_list[inosym] = isym

                    #saving k points ind list
                    kpoints_rotated = [transpose(inv(rot)) * k_point for k_point in kpoints]  
                    ind_k_point = ElectronPhonon.find_matching_qpoints(kpoints, kpoints_rotated)
                    ind_k_list[inosym] = ind_k_point

                    check = false
                    break
                end
            end
            isym += 1
        end
        inosym += 1
    end

    return Symmetries(ineq_atoms_list, trans_list, rot_list, dR_nosym, ind_k_list), Ndisplace_symm #
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
        @error "Not all the symmmetries for EP were found, only phonons could be calculated"
    else
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
- `eps`: (optional) Tolerance for the interval boundaries. Default is `1e-3`.
"""
function fold_component(x, eps=1e-3)
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

                eps = 1e-5
                if i1 >= N1 - eps || i2 >= N2 - eps || i3 >= N3 - eps
                    println(i1, i2, i3, N1, N2, N3)
                    @error "Symmetries usage gave error in folding"
                end

                ind = i1 + (i2) * N1 + (i3) * N1 * N2
                push!(mapp, ind)
            end
        end
    end
    return mapp
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
               if (i1 + (i2) * N1 + (i3) * N1 * N2) != mapp[ind]
                    println("different")
                    println(i1, i2, i3, ind, mapp[ind])
                    @error "Symmetries usage gave error in rotation"
                end
                ind += 1

                ff_rot[i1+1, i2+1, i3+1] = ff[i+1, j+1, k+1]
            end
        end
    end
    return ff_rot
end
