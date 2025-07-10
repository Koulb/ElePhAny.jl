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

    kpoints = [determine_q_point(path_to_calc*"displacements/scf_0",ik) for ik in 1:prod(k_mesh)]

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

function fold_component(x, eps=5e-3)
    """
    This routine folds number with given accuracy, so it would be inside the section from 0 to 1 .

        Returns:
            :x: folded number

    """
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

function rotate_grid(N1, N2, N3, rot, tras)
    """
    This routine change the grid according to given rotation and translation.

        Returns:
            :mapp (list): list of indexes of transformed grid

    """
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

function rotate_deriv(N1, N2, N3, mapp, ff)
    """
    This routine rotate the derivative according to the given grid.

        Returns:
            :ff_rot (np.array): array containing values of the derivative on a new frid

    """
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
