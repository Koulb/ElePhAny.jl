function check_symmetries(path_to_calc, unitcell, abs_disp)
    unitcell_phonopy = phonopy.structure.atoms.PhonopyAtoms(;symbols=unitcell[:symbols], 
    cell=pylist(pyconvert(Array,unitcell[:cell])./bohr_to_ang),#Should be in Bohr, hence conversion
    scaled_positions=unitcell[:scaled_positions],
    masses=unitcell[:masses])
    
    phonon_symm = phonopy.Phonopy(unitcell_phonopy)
    phonon_nosymm = phonopy.Phonopy(unitcell_phonopy, is_symmetry=false)  
    symm_ops = phonon_symm.symmetry.get_symmetry_operations()
    scaled_pos = phonon_symm.primitive.scaled_positions
    
    phonon_symm.generate_displacements(distance=abs_disp)
    phonon_nosymm.generate_displacements(distance=abs_disp)
    phonon_nosymm.save(path_to_calc*"displacements/phonopy_params_nosym.yaml")
    
    scaled_pos = pyconvert(Vector{Vector{Float64}},scaled_pos) 
    Uᶜʳʸˢᵗ = pyconvert(Matrix{Float64},phonon_symm.primitive.cell)
    
    dataˢʸᵐ = pyconvert(Vector{Vector{Float64}},phonon_symm.get_displacements())
    dRˢʸᵐ = [round.(transpose(Uᶜʳʸˢᵗ^-1) * vec[2:4], digits=16) for vec in dataˢʸᵐ]
    Rˢʸᵐ  = [scaled_pos[convert(Int64, vec[1])+1] for vec in dataˢʸᵐ]
    
    dataⁿᵒˢʸᵐ = pyconvert(Vector{Vector{Float64}},phonon_nosymm.get_displacements())
    dRⁿᵒˢʸᵐ = [round.(transpose(Uᶜʳʸˢᵗ^-1) * vec[2:4], digits=16) for vec in dataⁿᵒˢʸᵐ]
    Rⁿᵒˢʸᵐ  = [scaled_pos[convert(Int64, vec[1])+1] for vec in dataⁿᵒˢʸᵐ]
    
    trans_list = []
    rot_list  = []
    ineq_atoms_list = []
    index = 1
    
    for inosym in 1:length(Rⁿᵒˢʸᵐ)
        R2 = Rⁿᵒˢʸᵐ[inosym] + dRⁿᵒˢʸᵐ[inosym]
        for isym in 1:length(Rˢʸᵐ)
            R1 = Rˢʸᵐ[isym] + dRˢʸᵐ[isym]
            for (tras_py, rot_py) in zip(symm_ops["translations"], symm_ops["rotations"])
                trans = pyconvert(Vector{Float64}, tras_py) 
                rot = pyconvert(Matrix{Float64}, rot_py)
                rotR1 = rot * R1 .+ trans
    
                if all(abs.(R2 - rotR1) .< 1e-8)
                    @info "Found symmetry $index out of $(length(Rⁿᵒˢʸᵐ))"
                    index += 1
                    @info "translation: $trans"
                    @info "rotation   : $rot"
                    push!(trans_list, trans)
                    push!(rot_list, rot)
                    push!(ineq_atoms_list, isym)
                    break
                end
            end
        end
    end

    return Symmetries(ineq_atoms_list, trans_list, rot_list)
end

function check_symmetries!(model::ModelQE)
    symmetries = check_symmetries(model.path_to_calc, model.unitcell, model.abs_disp)
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
                    #println(i1, i2, i3, N1, N2, N3)
                    # error("Error in folding")
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
               # if (i1 + (i2) * N1 + (i3) * N1 * N2) != mapp[ind]
                    #println("different")
                    #println(i1, i2, i3, ind, mapp[ind])
                    # error()
                #end
                ind += 1

                ff_rot[i1+1, i2+1, i3+1] = ff[i+1, j+1, k+1]
            end
        end
    end
    return ff_rot
end
