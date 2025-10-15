using LinearAlgebra, YAML, Plots, Base.Threads


"""
    run_calculations(model::ModelQE)

Performs a series of calculations for a given `ModelQE` instance related to electron and phonon interactions.

# Arguments
- `model::ModelQE`: The model containing calculation parameters and paths.

# Description
This function runs the following calculations in sequence:
1. Electron displacement calculations using `run_disp_calc`.
2. Non-self-consistent field (NSCF) calculations using `run_nscf_calc`.
3. Displacement NSCF calculations using `run_disp_nscf_calc`.

All calculations are performed in the directory specified by `model.path_to_calc * "displacements/"`, using the number of displacements (`model.Ndispalce`) and MPI ranks (`model.mpi_ranks`) specified in the model.
"""
function run_calculations(model::ModelQE)
    # Electrons calculation
    run_disp_calc(model.path_to_calc*"displacements/", model.Ndispalce, model.mpi_ranks)
    run_nscf_calc(model.path_to_calc*"displacements/", model.mpi_ranks)
    run_disp_nscf_calc(model.path_to_calc*"displacements/", model.Ndispalce, model.mpi_ranks)
end

run_calculations(model::ModelKCW) = run_disp_calc(model)

"""
    prepare_model(model::ModelQE)

Prepares the model for electron-phonon calculations based on the provided `ModelQE` object.

# Arguments
- `model::ModelQE`: The model configuration containing calculation parameters, paths, and settings.

# Description
This function performs the following steps:
- Checks for unsupported combinations of symmetry and k-point mesh in supercell calculations.
- For calculations with both k-point mesh and supercell size greater than 1, reads structural data, computes a unified grid, and prepares undistorted and distorted wave functions.
- For calculations with only supercell size greater than 1 (and k-point mesh equal to 1), prepares undistorted wave functions.
- Prepares phonon data for all cases.
"""
function prepare_model(model::ModelQE; save_dynq=true)
    # save_potential(model.path_to_calc*"displacements/", model.Ndispalce, model.sc_size, model.mpi_ranks)

    if model.use_symm == true && all(model.k_mesh .!= 1)
        @error "Symmetry usage is not implemented for supercell calculations with kpoints"
    end

    if any(model.k_mesh .!= 1) && any(model.sc_size .!= 1) && all(==(model.sc_size[1]), model.sc_size) #for now only ostropic case
        #additioanl data for creating unified grid
        data = ase_io.read(model.path_to_calc*"displacements/scf_0/scf.in")
        a = pyconvert(Float64, data.cell.get_bravais_lattice().a)
        ecutoff    = model.scf_parameters[:ecutwfc]
        mesh_scale = model.k_mesh .* model.sc_size

        miller_map = create_unified_Grid(model.path_to_calc*"displacements/", a, ecutoff, mesh_scale)
        prepare_wave_functions_undisp(model.path_to_calc*"displacements/",miller_map, model.sc_size; k_mesh=model.k_mesh)# path_to_calc=path_to_calc,kcw_chanel=kcw_chanel
        prepare_wave_functions_disp(model.path_to_calc*"displacements/", miller_map, model.Ndispalce, model.k_mesh)
    elseif  all(model.k_mesh .== 1)  && any(model.sc_size .!= 1)
        prepare_wave_functions_undisp(model.path_to_calc*"displacements/", model.sc_size; k_mesh=model.k_mesh)
    end

    prepare_phonons_data(model.path_to_calc*"displacements/",model.unitcell, model.abs_disp, model.sc_size, model.k_mesh, model.use_symm, model.Ndispalce; save_dynq=save_dynq)
end

"""
    prepare_model(model::ModelKCW)

Prepares the given `ModelKCW` model for further calculations. This function performs the following steps:
1. Calls `prepare_kcw_data(model)` to set up necessary data for the model.
2. Calls `prepare_wave_functions_undisp` with the path to the displacement calculations and the supercell size to prepare wave functions without displacements.

# Arguments
- `model::ModelKCW`: The model object containing all necessary parameters and paths for preparation.
"""
function prepare_model(model::ModelKCW)
    prepare_kcw_data(model)
    prepare_wave_functions_undisp(model.path_to_calc*"displacements/", model.sc_size)
end

function check_calculations(path_to_calc, Ndisp)
    check = false

    println("Waiting for calculations to finish:")
    while !check
        sleep(10)
        try
            command = `squeue --user=$(ENV["USER"])` # need to understand how to write the proper name
            run(command)
        catch; end

        try
            file = open(path_to_calc*"group_$Ndisp/scf.out", "r")
            lines = readlines(file)
            close(file)
            if occursin("JOB DONE.", lines[end-1])
                check = true
            end

            file = open(path_to_calc*"scf_0/scf.out", "r")
            lines = readlines(file)
            close(file)
            if occursin("JOB DONE.", lines[end-1])
                check = check && true
            end
        catch; end
    end
    println("All calculations finished")

    return
end


"""
    electron_phonon_qe(path_to_in::String, ik::Int, iq::Int, mpi_ranks::Int, path_to_qe::String)

Runs a Quantum ESPRESSO phonon calculation for electron-phonon coupling at specified k- and q-points.

# Arguments
- `path_to_in::String`: Path to the input directory containing calculation files.
- `ik::Int`: Index of the k-point to use.
- `iq::Int`: Index of the q-point to use.
- `mpi_ranks::Int`: Number of MPI ranks (processes) to use for parallel execution.
- `path_to_qe::String`: Path to the Quantum ESPRESSO installation or executable directory.

# Description
This function:
- Changes the working directory to the specified input directory.
- Constructs a dictionary of input parameters for the `ph.x` Quantum ESPRESSO module.
- Writes the input file `ph.in` for the phonon calculation.
- Executes the `ph.x` program using MPI, redirecting output and error streams to files.
"""
function electron_phonon_qe(path_to_in::String, ik::Int, iq::Int, mpi_ranks::Int, path_to_qe::String)
    dir_name = "scf_0/"
    cd(path_to_in*dir_name) do
        kpoint = determine_q_point_cart(path_to_in*dir_name,ik)
        qpoint = determine_q_point_cart(path_to_in*dir_name,iq)

        # println("kpoint = ", kpoint)
        # println("qpoint = ", qpoint)

        parameters = Dict(
                "inputph" => Dict(
                "prefix" => "'scf'",
                "outdir" => "'./tmp'",
                "fildvscf" => "'dvscf'",
                "ldisp"    => ".true.",
                "fildyn"   => "'dyn'",
                "tr2_ph"   =>  1.0e-18,
                "qplot"    => ".true.",
                "q_in_band_form" => ".true.",
                "electron_phonon" => "'epw'",
                "kx" =>  kpoint[1],
                "ky" =>  kpoint[2],
                "kz" =>  kpoint[3],
            )
        )

        # Write the ph.x input file
        open("ph.in", "w") do f
            for (section, section_data) in parameters
                write(f, "&$section\n")
                for (key, value) in section_data
                    write(f, "  $key = $value\n")
                end
                write(f, "/\n")
            end
            write(f,"1\n")
            write(f,"$(qpoint[1]) $(qpoint[2]) $(qpoint[3]) 1 #\n")
        end

        path_to_ph = path_to_qe*"test-suite/not_epw_comp/ph.x"
        command = `mpirun -np $mpi_ranks $path_to_ph -in ph.in`
        #println(command)
        run(pipeline(command, stdout="ph.out", stderr="errs_ph.txt"))
    end
end

"""
    electron_phonon_qe(model::ModelQE, ik::Int, iq::Int)

Computes electron-phonon coupling using  Quantum ESPRESSO for a given model.

# Arguments
- `model::ModelQE`: The Quantum ESPRESSO model containing calculation parameters and paths.
- `ik::Int`: Index of the k-point.
- `iq::Int`: Index of the q-point.

# Returns
- The result of the electron-phonon coupling calculation for the specified k-point and q-point.
"""
function electron_phonon_qe(model::ModelQE, ik::Int, iq::Int)
    electron_phonon_qe(model.path_to_calc*"displacements/", ik, iq, model.mpi_ranks, model.path_to_qe)
end

"""
    find_degenerate(energies, thr=1e-3)

Groups nearly degenerate energy levels within a given threshold.

# Arguments
- `energies::AbstractVector{<:Real}`: A vector of energy values, assumed to be sorted.
- `thr::Real=1e-3`: Threshold for considering two energies as degenerate (default: `1e-3`).

# Returns
- `ineq_ener::Vector{<:Real}`: Vector of unique (inequivalent) energies, each representing a group of degenerate states.
- `eq_states::Vector{Vector{Int}}`: A vector of vectors, where each subvector contains the indices of `energies` that are degenerate with the corresponding entry in `ineq_ener`.
"""
function find_degenerate(energies, thr=1e-3)
    ineq_ener = [energies[1]]
    eq_states = [[1]]

    for (i, en) in enumerate(energies[2:end])
        if abs(en - ineq_ener[end]) < thr
            push!(eq_states[end], i+1)
        else
            push!(ineq_ener, en)
            push!(eq_states, [i+1])
        end
    end

    return ineq_ener, eq_states
end


"""
    parse_ph(file_name, nbands, nfreq)

Parse electron-phonon matrix elements from a ph.out file.

# Arguments
- `file_name::AbstractString`: Path to the input file containing electron-phonon data.
- `nbands::Int`: Number of electronic bands.
- `nfreq::Int`: Number of phonon frequencies.

# Returns
- `elph_dfpt::Array{ComplexF64,3}`: A 3D array of size `(nbands, nbands, nfreq)` containing the parsed electron-phonon matrix elements (in meV).
"""
function parse_ph(file_name, nbands, nfreq)
    file = open(file_name, "r")
    # Initialize a flag to check if word is found
    found_data = false

    # Initialize a counter for the next 10 lines
    lines_to_read = nbands*nbands*nfreq+2
    current_line = 1
    elph_dfpt = zeros(ComplexF64,(nbands, nbands, nfreq))

    # Read the file line by line
    for line in eachline(file)
        if found_data
            if current_line > 1
                split_line = split(line)
                i, j, iph = parse(Int64,split_line[1]), parse(Int64,split_line[2]), parse(Int64,split_line[3])
                elph_dfpt[i,j,iph]= parse(Float64,split_line[end])/1e3 # from meV to eV
            end

            current_line += 1
            if current_line == lines_to_read
                break
            end
        elseif occursin("ibnd     jbnd", line)
            found_data = true
        end
    end

    return elph_dfpt
end


function load_wf_debug(path_to_in::String)
    wfc_list = Dict()
    for iband in 1:32
        wfc_re, = read_potential(path_to_in*"wfun_$(iband)_1_re"; skiprows=1)
        wfc_im, = read_potential(path_to_in*"wfun_$(iband)_1_im"; skiprows=1)

        wfc = wfc_re .+ 1im .* wfc_im
        wfc_list["wfc$iband"] = wfc
    end

    return wfc_list
end

function load_wf_u_debug(path_to_in::String, ik)
    wfc_list = Dict()
    for iband in 1:4
        result = pyconvert(Array{Complex{Float64}},np.load(path_to_in*"wfun_phase_k$(ik)_$(iband)_1.npy"))
        wfc_list["wfc$iband"] = result
    end

    return wfc_list
end


"""
    electron_phonon(
        path_to_in::String,
        abs_disp,
        Nat,
        ik,
        iq,
        sc_size,
        k_mesh,
        ϵkᵤ_list,
        ϵₚ_list,
        ϵₚₘ_list,
        k_list,
        U_list,
        V_list,
        M_phonon,
        ωₐᵣᵣ_ₗᵢₛₜ,
        εₐᵣᵣ_ₗᵢₛₜ,
        mₐᵣᵣ;
        save_epw::Bool=false
    )

Compute electron-phonon matrix elements using the frozen-phonon approach.

# Arguments
- `path_to_in::String`: Path to the input calculation directory.
- `abs_disp`: Absolute displacement used in frozen-phonon calculations.
- `Nat`: Number of atoms in the unit cell.
- `ik`: Index of the k-point.
- `iq`: Index of the q-point.
- `sc_size`: Supercell size.
- `k_mesh`: Number of k-points in the mesh.
- `ϵkᵤ_list`: List of eigenvalues for each k-point.
- `ϵₚ_list`: List of perturbed eigenvalues for positive displacement.
- `ϵₚₘ_list`: List of perturbed eigenvalues for negative displacements.
- `k_list`: List of k-points.
- `U_list`: List of braket matrices for positive displacements.
- `V_list`: List of braket matrices for negative displacements.
- `M_phonon`: List of phonon rotation matrices for each atom.
- `ωₐᵣᵣ_ₗᵢₛₜ`: List of phonon frequencies for each q-point.
- `εₐᵣᵣ_ₗᵢₛₜ`: List of phonon eigenvectors for each q-point.
- `mₐᵣᵣ`: List of atomic masses.
- `save_epw::Bool=false`: If `true`, saves the rotated braket list for EPW.

# Returns
- If `save_epw` is `true`, returns the rotated braket list (`braket_list_rotated`).
- Otherwise, returns the symmetrized electron-phonon matrix elements (`symm_elph`).

# Description
This function computes the electron-phonon coupling matrix elements by evaluating the overlap ("braket")
between electronic states perturbed by atomic displacements.
The calculation is performed for each atom and Cartesian direction, and the results are rotated into the phonon eigenmode basis.
The function also symmetrizes the resulting matrix elements over phonon and electronic states and compares them with DFPT
results if available.
"""
function electron_phonon(path_to_in::String, abs_disp, Nat, ik, iq, sc_size, k_mesh, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list , U_list, V_list, M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ; save_epw::Bool=false, save_qeraman::Bool=false)
    # cd(path_to_in)

    Ndisp_nosym = 6*Nat
    scale = ev_to_ry / abs_disp

    group = "scf_0/"
    ikq = fold_kpoint(ik,iq,k_list)

    ϵkᵤ = ϵkᵤ_list[ik]
    ϵqᵤ = ϵkᵤ_list[ikq]
    nbands = length(ϵkᵤ)

    braket = zeros(Complex{Float64}, nbands, nbands)
    braket_list = []
    braket_list_rotated = []
    # print("Electron_phonon check:")

    for ind in 1:2:Ndisp_nosym
        ind_abs = (ind-1)÷2 + 1

        ϵₚ = ϵₚ_list[ind_abs]
        #TODO understand whether symmetry usage is properly justified
        ϵₚₘ = ϵₚₘ_list[ind_abs]

        Uk  = U_list[ind_abs][:,ik,:,:]
        Ukₘ = V_list[ind_abs][:,ik,:,:]
        
        # u_trace_check = conj(transpose(Uk))*Uk
        # u_m_trace_check = conj(transpose(Ukₘ))*Ukₘ

        # for i in 1:nbands
        #     println("Uk trace check [$i, $i] = ", u_trace_check[i,i])
        #   if real(u_trace_check[i,i]) < 0.999999 ||  real(u_m_trace_check[i,i]) < 0.999999
        #         @warn "Uk trace check [$i, $i] = ", u_trace_check[i,i]
        #         @warn "Uk_m trace check [$i, $i] = ", u_m_trace_check[i,i]
        #         @warn "U matrices are not in good shape at displacemnt $ind"
        #       end
        # end

        Uq  = U_list[ind_abs][:,ikq,:,:]
        Uqₘ = V_list[ind_abs][:,ikq,:,:]
        
        # u_trace_check = conj(transpose(Uq))*Uq
        # u_m_trace_check = conj(transpose(Uqₘ))*Uqₘ

        # for i in 1:nbands
        #   if real(u_trace_check[i,i]) < 0.999999 ||  real(u_m_trace_check[i,i]) < 0.999999
        #         @warn "Uq trace check [$i, $i] = ", u_trace_check[i,i]
        #         @warn "Uq_m trace check [$i, $i] = ", u_m_trace_check[i,i]
        #         @warn "U matrices are not in good shape at displacemnt $ind"
        #       end
        # end

        # println("Calculating brakets for group $ind")
        for i in 1:nbands
            for j in 1:nbands
                result = 0.0#(i==j && ik==ikq ? -ϵkᵤ[i] : 0.0)#0.0##TODO: check this iq or ikq
                # println(i, ' ', j, ' ', result)

                for k in 1:nbands*prod(sc_size)
                    for ip in 1:prod(k_mesh)
                    # ip = 1
                        result += Uk[ip, k, j]* conj(Uq[ip, k,i]) * ϵₚ[ip][k]
                        result -= Ukₘ[ip, k, j]* conj(Uqₘ[ip, k,i]) * ϵₚₘ[ip][k]
                        # if real(Uk[k,j]* conj(Uq[k,i])) ≈ 1.0
                        # if isapprox(real(Uk[k,j]* conj(Uq[k,i])), 1.0, atol=1e-6)
                        #     println(Uk[k,j]* conj(Uq[k,i]))
                        #     println(i, ' ', j)
                        #     # # result += ϵₚ[k]
                        #     # result = (ϵₚ[k] - ϵₚₘ[k])/2.0
                        #     # #Need to investigate even small displacements
                        #     println("ϵkᵤ = ", ϵkᵤ)
                        #     println("ϵₚ[k] = ", ϵₚ[k])
                        #     println("ϵkᵤ - ϵₚ[k] = ", (ϵₚ[k]-ϵkᵤ[i]))
                        #     println("ϵkᵤ - U^2*ϵₚ[k] = ", (Uk[k,j]* conj(Uq[k,i]) * ϵₚ[k]-ϵkᵤ[i]))

                        #    # println("ϵₚ[k] - ϵₚₘ[k] /2 = ", result)

                        #     result += Uk[k,j]* conj(Uq[k,i]) * ϵₚ[k]
                        # else
                        #     result += Uk[k,j]* conj(Uq[k,i]) * ϵₚ[k]
                        # end
                        # println(k,' ',ip, ' ',ϵₚ[ip][k], ' ',Uk[ip, k,j]* conj(Uq[ip, k,i]), ' ', result)
                        # println(k,' ',ip, ' ', ϵₚₘ[ip][k], ' ',Ukₘ[ip, k,j]* conj(Uqₘ[ip, k,i]), ' ', result)
                    end
                end

                braket[i,j] = result/2.0
            end
            # println("_____________________________________________________________")
            # exit(3)
        end


        push!(braket_list, transpose(conj(braket))*scale*prod(sc_size))

    end
    # println("Braket list unrotated")
    # println(scale*sc_size^3)
    # println(braket_list)

    for iat in 1:Nat
        U_inv = M_phonon[iat]
        temp_iat::Int = 1 + 3 * (iat - 1)
        braket_temp = braket_list[temp_iat:temp_iat+2]
        push!(braket_list_rotated, U_inv * braket_temp) #transpose(U_inv) * braket_temp
    end

    if save_epw
        #save braket_list_rotated in the file
        try
            open(path_to_in*"epw/braket_list_rotated_$(ik)_$(iq)", "w") do io
                for iat in 1:Nat
                    for i in 1:3
                        for j in 1:nbands
                            for k in 1:nbands
                                data = [iat, i, j, k, real(braket_list_rotated[iat][i][j,k]), imag(braket_list_rotated[iat][i][j,k])]
                                @printf(io, "  %5d  %5d  %5d  %5d  %16.12f  %16.12f\n", data...)
                            end
                        end
                    end
                end
            end
        catch
            @info "Could not save braket_list_rotated, check if epw folder exists in path_to_calc/displacements"
        end
        return braket_list_rotated
    else
        # println("Braket list rotated")
        # println(braket_list_rotated)

        ## Multiplication by phonon eigenvector and phonon frequency
        ## Compute electron-phonon vertex in normal coordinate basis
        
        ωₐᵣᵣ = ωₐᵣᵣ_ₗᵢₛₜ[iq]
        εₐᵣᵣ = εₐᵣᵣ_ₗᵢₛₜ[iq]

        #DEBUG WITH QE OUTPUT##
        #TODO Need to fix and understand the reason, probably eigenvectors
        ωₐᵣᵣ, εₐᵣᵣ = parse_qe_ph(path_to_in*"scf_0/dyn1",Nat) 
        #DEBUG WITH QE OUTPUT##

        gᵢⱼₘ_ₐᵣᵣ = Array{ComplexF64, 3}(undef, (nbands, nbands, length(ωₐᵣᵣ)))

        for i in 1:nbands
            for j in 1:nbands
                # open(path_to_in*"elph_elements/ep_$(i)_$(j)", "w") do io
                    for iph in 1:3*Nat
                        ω = ωₐᵣᵣ[1,iph] * cm1_to_ry
                        ε = εₐᵣᵣ[1,iph,:]
                        gᵢⱼₘ = 0.0
                        for iat in 1:Nat
                            braket_cart = braket_list_rotated[iat]
                            m = mₐᵣᵣ[iat] * uma_to_ry
                            disp = (ω > 0.0 ? sqrt(1/(2*m*ω)) : 0.0)#EPW convention for soft modes
                            for i_cart in 1:3
                                braket = braket_cart[i_cart]
                                temp_iat::Int = 3*(iat - 1) + i_cart

                                gᵢⱼₘ += disp*conj(ε[temp_iat])*braket[i,j]

                                #if i == 1 && j == 3
                                #   println(i,' ',j,' ',iph,' ',disp, ' ',  ω,' ', m,' ',ε[temp_iat],' ',braket[i,j], ' ',gᵢⱼₘ)
                                #end

                            end
                        end
                        gᵢⱼₘ_ₐᵣᵣ[i,j,iph] = gᵢⱼₘ/ev_to_ry
                        data = [iph, ω/cm1_to_ry, real(gᵢⱼₘ)/ev_to_ry, imag(gᵢⱼₘ)/ev_to_ry]
                        #@printf("  %5d  %10.6f  %10.6f   %10.6f\n", data...)
                        # @printf(io, "  %5d  %10.6f  %10.10f   %10.10f\n", data...)
                    end
                    #@printf("____________________________________________\n")
                # end
            end
        end

        #Acoustic sum rule for poor
        # if ik==ikq && iq == 1;
        #     gᵢⱼₘ_ₐᵣᵣ[:,:,1:3] .= 0.0
        # end

        # Symmetrization
        symm_elph = zeros(ComplexF64,(nbands, nbands, length(ωₐᵣᵣ)))#gˢʸᵐᵢⱼₘ_ₐᵣᵣ
        elph = deepcopy(gᵢⱼₘ_ₐᵣᵣ)

        thr = 1e-2#0.1#1e-4
        # symm through phonons
        for iph1 in 1:length(ωₐᵣᵣ)
            ω₁ = ωₐᵣᵣ[iph1]
            for ie in 1:nbands
                for je in 1:nbands
                    n = 0
                    g² = 0.0
                    for iph2 in 1:length(ωₐᵣᵣ)
                        ω₂ = ωₐᵣᵣ[iph2]
                        if abs(ω₁ - ω₂) < thr
                            n += 1
                            g² += conj(elph[ie,je,iph2]) * elph[ie,je,iph2]
                        end
                    end
                    g² /= n
                    symm_elph[ie, je, iph1] = real(sqrt(g²))
                end
            end
        end
        elph = deepcopy(symm_elph)

        # symm through k electrons
        for iph1 in 1:length(ωₐᵣᵣ)
            for je in 1:nbands
                for ie in 1:nbands
                    n = 0
                    g² = 0.0
                    ε₁ = ϵkᵤ[ie]
                    for ie2 in 1:nbands
                        ε₂ = ϵkᵤ[ie2]
                        if abs(ε₁ - ε₂) < thr
                            n += 1
                            g² += conj(elph[ie2, je, iph1]) * elph[ie2, je, iph1]
                        end
                    end
                    g² /= n
                    symm_elph[ie, je, iph1] = real(sqrt(g²))
                end
            end
        end
        elph = deepcopy(symm_elph)

        #symm through k+q electrons
        for iph1 in 1:length(ωₐᵣᵣ)
            for ie in 1:nbands
                for je in 1:nbands
                    n = 0
                    g² = 0.0
                    ε₁ = ϵqᵤ[je]
                    for je2 in 1:nbands
                        ε₂ = ϵqᵤ[je2]
                        if abs(ε₁ - ε₂) < thr
                            n += 1
                            g² += conj(elph[ie, je2, iph1]) * elph[ie, je2, iph1]
                        end
                    end
                    g² /= n
                    symm_elph[ie, je, iph1] = real(sqrt(g²))
                end
            end
        end

        #read dfpt data
        elph_dfpt = zeros(ComplexF64, size(symm_elph))
        ωₐᵣᵣ_DFPT = zeros(Float64, size(ωₐᵣᵣ))

        try
            elph_dfpt = parse_ph(path_to_in*"scf_0/ph.out", nbands, length(ωₐᵣᵣ))
            ωₐᵣᵣ_DFPT, _ = parse_qe_ph(path_to_in*"scf_0/dyn1", Nat)
        catch
            @info "Could not read DFPT data"
        end

        if save_qeraman
            qeraman_dir = path_to_in * "qeraman/"
            if !isdir(qeraman_dir)
                mkpath(qeraman_dir)
            end
            if !isfile(qeraman_dir*"data.elph")
                open(qeraman_dir*"data.elph", "w") do io
                    @printf(io, "      Electron-phonon matrix elements M(k,q) = sqrt(hbar/2*omega)<psi(k+q,j)|dvscf_q*psi(k,i)>\n")
                    @printf(io, "      nbnd   nmodes   nqs   nkstot   nksqtot\n")
                    @printf(io, "    %5d        %5d        %5d     %5d     %5d\n", nbands, 3*Nat, prod(sc_size), prod(k_mesh), prod(sc_size)*prod(k_mesh))
                    @printf(io, "      qx   qy   qz   weight_q   iq\n")
                    @printf(io, "      kx   ky   kz   weight_k   ik   ik+q\n")
                    @printf(io, "      ibnd  jbnd  imode  enk[eV]  enk+q[eV]  omega(q)[meV]   |M|[meV]   Re(M)[meV]   Im(M)[meV]\n")
                    @printf(io, "      ------------------------------------------------------------------------------\n")
                    @printf(io, "   0.0000000   0.0000000   0.0000000   1.0000000        1\n")#Q=Gamma case for now
                end
            end
            #need to add condition where i erase the file  


            open(qeraman_dir*"data.elph", "a") do io
                @printf(io, "   %12.6f   %12.6f   %12.6f   %12.6f  %5d  %5d\n", k_list[ik][1], k_list[ik][2], k_list[ik][3], 2/prod(k_mesh), ik, ikq)
                for i in 1:nbands
                    for j in 1:nbands
                        for iph in 1:3*Nat
                            @printf(io, "  %5d  %5d  %5d  %10.6f  %10.6f  %12.6f  %12.6f %12.12f %12.12f\n", i,j, iph, ϵkᵤ[i], ϵqᵤ[j], ωₐᵣᵣ_DFPT[1,iph]*(1e3*cm1_to_ry/ev_to_ry), abs(gᵢⱼₘ_ₐᵣᵣ[i, j, iph]*1e3), real(gᵢⱼₘ_ₐᵣᵣ[i, j, iph]*1e3),imag(gᵢⱼₘ_ₐᵣᵣ[i, j, iph]*1e3))
                        end
                    end
                end
            end
        else
            try

                #saving resulting electron phonon couplings
                # @printf("      i      j      nu      ϵkᵤ        ϵqᵤ        ωₐᵣᵣ_frozen      ωₐᵣᵣ_DFPT       g_frozen    g_DFPT\n")
                open(path_to_in*"out/comparison_$(ik)_$(iq).txt", "w") do io
                for i in 1:nbands
                    for j in 1:nbands
                            for iph in 1:3*Nat#Need to chec
                            # @printf("  %5d  %5d  %5d  %10.6f  %10.6f  %12.6f  %12.6f  %12.12f %12.12f\n", i,j, iph, ϵkᵤ[i], ϵqᵤ[j], ωₐᵣᵣ[1,iph], ωₐᵣᵣ_DFPT[1,iph], symm_elph[i, j, iph], elph_dfpt[i, j, iph])
                                @printf(io, "  %5d  %5d  %5d  %10.6f  %10.6f  %12.6f  %12.6f  %12.12f %12.12f\n", i,j, iph, ϵkᵤ[i], ϵqᵤ[j], ωₐᵣᵣ[1,iph], ωₐᵣᵣ_DFPT[1,iph], symm_elph[i, j, iph], elph_dfpt[i, j, iph])
                            end
                        end
                    end
                end
            catch
                @info "Could not save symmetrized electron-phonon matrix elements, check if out folder exists in $(path_to_in)"
            end            
        end

        return symm_elph
    end

end

"""
    plot_ep_coupling(path_to_in::String, ik::Int, iq::Int; nbnd_max=-1)

Reads electron-phonon coupling data from a specified file, generates a scatter plot comparing two coupling values, and saves the plot as a PNG image.

# Arguments
- `path_to_in::String`: Path to the input directory containing the `out/comparison_ik_iq.txt` file.
- `ik::Int`: Index of the k-point used in the filename.
- `iq::Int`: Index of the q-point used in the filename.
- `nbnd_max::Int` (optional; default: -1): If greater than 0, only data with band indices less than `nbnd_max` are included.

# Returns
- `(x, y)`: Two arrays of `Float64` containing DFPT and FP data.
"""
function plot_ep_coupling(path_to_in::String, ik::Int, iq::Int; nbnd_max=-1)
    # Initialize empty arrays for x and y
    x = Float64[]
    y = Float64[]
    filename = path_to_in*"out/comparison_$(ik)_$(iq).txt"
    # Read data from the file and populate x and y arrays
    open(filename, "r") do file
        for line in eachline(file)
            # Split the line into columns
            columns = split(line)
            if length(columns) >= 5
                # Extract the last two columns and convert them to Float64
                x_val = parse(Float64, columns[end])
                y_val = parse(Float64, columns[end-1])

                i_val = parse(Int64, columns[1])
                j_val = parse(Int64, columns[2])

                if nbnd_max > 0
                    if i_val < nbnd_max && j_val < nbnd_max
                        push!(x, x_val)
                        push!(y, y_val)
                    end
                else
                    push!(x, x_val)
                    push!(y, y_val)
                end
            end
        end
    end


    # Create a scatter plot with transparent background
    plt = scatter(x, y, xlabel="g(DFPT)", ylabel="g(FD)", title="Comparison", color="red", foreground_color=:black)
    line = LinRange(0, 1.1*maximum(max.(x,y)), 4)
    plot!(plt, line, line, color="black", legend=false)
    xlims!(plt, 0, 1.1*maximum(x))
    ylims!(plt, 0, 1.1*maximum(y))

    savefig(plt, path_to_in*"out/comparison_$(ik)_$(iq).png")

    return x, y
end

"""
    plot_ep_coupling(model::ModelQE, ik::Int=0, iq::Int=0; nbnd_max=-1)

Plots the electron-phonon coupling for a given `ModelQE` instance.

# Arguments
- `model::ModelQE`: The model containing electron-phonon calculation data.
- `ik::Int=0`: Index of the k-point to plot (default is 0).
- `iq::Int=0`: Index of the q-point to plot (default is 0).

# Keyword Arguments
- `nbnd_max::Int=-1`: Maximum number of bands to include in the plot (default is -1, which may indicate all bands).
"""
function plot_ep_coupling(model::ModelQE, ik::Int=0, iq::Int=0; nbnd_max=-1)
    plot_ep_coupling(model.path_to_calc*"displacements/", ik, iq; nbnd_max=nbnd_max)
end

"""
    electron_phonon(path_to_in::String, abs_disp, natoms, ik, iq, sc_size, electrons::AbstractElectrons, phonons::AbstractPhonons; save_epw::Bool=false, path_to_calc="", kcw_chanel="")

High-level interface for computing electron-phonon interactions when Electrons and Phonons are provided.

# Arguments
- `path_to_in::String`: Path to the input file or directory.
- `abs_disp`: Absolute displacement parameter (type depends on context).
- `natoms`: Number of atoms in the system.
- `ik`: Index of the k-point.
- `iq`: Index of the q-point.
- `sc_size`: Supercell size.
- `electrons::AbstractElectrons`: Object containing electronic structure data.
- `phonons::AbstractPhonons`: Object containing phonon data.

# Keyword Arguments
- `save_epw::Bool=false`: Whether to save the EPW (Electron-Phonon Wannier) data.
- `path_to_calc=""`: Optional path to calculation directory.
- `kcw_chanel=""`: Optional kcw channel specification.
"""
function electron_phonon(path_to_in::String, abs_disp, natoms, ik, iq, sc_size, electrons::AbstractElectrons, phonons::AbstractPhonons; save_epw::Bool=false,save_qeraman::Bool=false, path_to_calc="",kcw_chanel="")
    ϵkᵤ_list = electrons.ϵkᵤ_list
    ϵₚ_list = electrons.ϵₚ_list
    ϵₚₘ_list = electrons.ϵₚₘ_list
    k_list = electrons.k_list
    U_list = electrons.U_list
    V_list = electrons.V_list
    M_phonon = phonons.M_phonon
    ωₐᵣᵣ_ₗᵢₛₜ = phonons.ωₐᵣᵣ_ₗᵢₛₜ
    εₐᵣᵣ_ₗᵢₛₜ = phonons.εₐᵣᵣ_ₗᵢₛₜ
    mₐᵣᵣ = phonons.mₐᵣᵣ

    electron_phonon(path_to_in, abs_disp, natoms, ik, iq, sc_size, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list , U_list, V_list, M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ; save_epw=save_epw,save_qeraman=save_qeraman)

end

"""
    electron_phonon(model::AbstractModel, ik, iq, electrons::AbstractElectrons, phonons::AbstractPhonons; save_epw::Bool=false, path_to_calc="", kcw_chanel="")

Compute electron-phonon coupling properties for the given model, electron, and phonon objects.

# Arguments
- `model::AbstractModel`: The model containing system parameters and calculation paths.
- `ik`: Index or identifier for the electron k-point.
- `iq`: Index or identifier for the phonon q-point.
- `electrons::AbstractElectrons`: Object containing electronic structure data (eigenvalues, k-points, etc.).
- `phonons::AbstractPhonons`: Object containing phonon properties (modes, frequencies, etc.).

# Keyword Arguments
- `save_epw::Bool=false`: Whether to save the electron-phonon Wannier (EPW) data.
- `path_to_calc::String=""`: Path to the calculation directory (overrides model's path if provided).
- `kcw_chanel::String=""`: Optional spin chanel for KCW ("up" has KI functional and "down" has DFT).

# Description
This function prepares and calls the lower-level electron-phonon coupling calculation using the provided model, electron, and phonon data. It extracts relevant properties from the `electrons` and `phonons` objects and passes them, along with model and calculation parameters, to the core computation routine.
"""
function electron_phonon(model::AbstractModel, ik, iq, electrons::AbstractElectrons, phonons::AbstractPhonons; save_epw::Bool=false,save_qeraman::Bool=false, path_to_calc="",kcw_chanel="")
    ϵkᵤ_list = electrons.ϵkᵤ_list
    ϵₚ_list = electrons.ϵₚ_list
    ϵₚₘ_list = electrons.ϵₚₘ_list
    k_list = electrons.k_list
    U_list = electrons.U_list
    V_list = electrons.V_list
    M_phonon = phonons.M_phonon
    ωₐᵣᵣ_ₗᵢₛₜ = phonons.ωₐᵣᵣ_ₗᵢₛₜ
    εₐᵣᵣ_ₗᵢₛₜ = phonons.εₐᵣᵣ_ₗᵢₛₜ
    mₐᵣᵣ = phonons.mₐᵣᵣ

    natoms = length(model.unitcell[:symbols])

    electron_phonon(model.path_to_calc*"displacements/", model.abs_disp, natoms, ik, iq, model.sc_size, model.k_mesh, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list , U_list, V_list, M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ; save_epw=save_epw,save_qeraman=save_qeraman)

end
