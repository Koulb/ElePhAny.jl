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
function check_symmetries(path_to_calc, unitcell, sc_size, abs_disp)
    unitcell_phonopy = phonopy.structure.atoms.PhonopyAtoms(;symbols=unitcell[:symbols],
    cell=pylist(pyconvert(Array,unitcell[:cell])./bohr_to_ang),#Should be in Bohr, hence conversion
    scaled_positions=unitcell[:scaled_positions],
    masses=unitcell[:masses])
    supercell_matrix=pylist([[sc_size, 0, 0], [0, sc_size, 0], [0, 0, sc_size]])

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

    dataⁿᵒˢʸᵐ = pyconvert(Vector{Vector{Float64}},phonon_nosymm.get_displacements())
    dRⁿᵒˢʸᵐ = [round.(transpose(Uᶜʳʸˢᵗ^-1) * vec[2:4], digits=16) for vec in dataⁿᵒˢʸᵐ]
    Rⁿᵒˢʸᵐ  = [scaled_pos[convert(Int64, vec[1])+1] for vec in dataⁿᵒˢʸᵐ]

    trans_list = []
    rot_list   = []
    ineq_atoms_list = []
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
                    check = false
                    break
                end
            end
            isym += 1
        end
        inosym += 1
    end

    return Symmetries(ineq_atoms_list, trans_list, rot_list)
end

"""
    check_symmetries!(model::AbstractModel)

Checks the symmetries of the given `model` and updates its symmetry-related fields in-place.

# Arguments
- `model::AbstractModel`: The model object containing calculation path, unit cell, supercell size, and displacement information.
"""
function check_symmetries!(model::AbstractModel)
    symmetries = check_symmetries(model.path_to_calc, model.unitcell, model.sc_size, model.abs_disp)
    natoms = length(pyconvert(Vector{Vector{Float64}}, model.unitcell[:scaled_positions]))

    if length(symmetries.trans_list) == 6 * natoms
        model.use_symm = true
        model.symmetries = symmetries
        model.Ndispalce = length(unique(symmetries.ineq_atoms_list))
    else
        model.use_symm = false
        model.Ndispalce = 6 * natoms
    end
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
