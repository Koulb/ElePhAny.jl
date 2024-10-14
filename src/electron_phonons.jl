using LinearAlgebra, Printf, YAML, Plots, Base.Threads


function run_calculations(model)
    # Electrons calculation
    run_disp_calc(model.path_to_calc*"displacements/", model.Ndispalce, model.mpi_ranks)
    run_nscf_calc(model.path_to_calc, model.unitcell, model.scf_parameters, model.mesh, model.path_to_qe, model.mpi_ranks)
end

function prepare_model(model::ModelQE)
    # save_potential(model.path_to_calc*"displacements/", model.Ndispalce, model.mesh, model.mpi_ranks)
    prepare_wave_functions_undisp(model.path_to_calc*"displacements/", model.mesh;)# path_to_calc=path_to_calc,kcw_chanel=kcw_chanel
    prepare_phonons_data(model.path_to_calc*"displacements/",model.unitcell, model.abs_disp, model.mesh, model.use_symm, model.Ndispalce)
end

function prepare_model(model::ModelKCW)
    prepare_wave_functions_undisp(model.path_to_calc*"displacements/", model.mesh)
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


function electron_phonon_qe(path_to_in::String, ik::Int, iq::Int, mpi_ranks::Int, path_to_qe::String)
    dir_name = "scf_0/"
    current_directory = pwd()
    cd(path_to_in*dir_name)

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
            "tr2_ph"   =>  1.0e-12,
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
    cd(current_directory)
end

function electron_phonon_qe(model::ModelQE, ik::Int, iq::Int)
    electron_phonon_qe(model.path_to_calc*"displacements/", ik, iq, model.mpi_ranks, model.path_to_qe)
end

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
                elph_dfpt[i,j,iph]= parse(Float64,split_line[end])/1e3 # to meV
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
        wfc_re, = read_potential(path_to_in*"wfun_$(iband)_1_re";skiprows=1)
        wfc_im, = read_potential(path_to_in*"wfun_$(iband)_1_im";skiprows=1)

        wfc = wfc_re + 1im * wfc_im
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


function electron_phonon(path_to_in::String, abs_disp, Nat, ik, iq, mesh, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list , U_list, V_list, M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ; save_epw::Bool=false)
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
        ϵₚₘ = ϵₚ_list[ind_abs]#ϵₚₘ_list[ind_abs]

        Uk  = U_list[ind_abs][ik,:,:]
        Ukₘ = V_list[ind_abs][ik,:,:]
        u_trace_check = conj(transpose(Uk))*Uk
        u_m_trace_check = conj(transpose(Ukₘ))*Ukₘ

        for i in 1:nbands
          if real(u_trace_check[i,i]) < 0.999999 ||  real(u_m_trace_check[i,i]) < 0.999999
                @warn "Uk trace check [$i, $i] = ", u_trace_check[i,i]
                @warn "Uk_m trace check [$i, $i] = ", u_m_trace_check[i,i]
                @warn "U matrices are not in good shape at displacemnt $ind"
              end
        end

        Uq  = U_list[ind_abs][ikq,:,:]
        Uqₘ = V_list[ind_abs][ikq,:,:]
        u_trace_check = conj(transpose(Uq))*Uq
        u_m_trace_check = conj(transpose(Uqₘ))*Uqₘ

        for i in 1:nbands
          if real(u_trace_check[i,i]) < 0.999999 ||  real(u_m_trace_check[i,i]) < 0.999999
                @warn "Uq trace check [$i, $i] = ", u_trace_check[i,i]
                @warn "Uq_m trace check [$i, $i] = ", u_m_trace_check[i,i]
                @warn "U matrices are not in good shape at displacemnt $ind"
              end
        end

        # println("Calculating brakets for group $ind")
        for i in 1:nbands
            for j in 1:nbands
                result = 0.0#(i==j && ik==ikq ? -ϵkᵤ[i] : 0.0)#0.0##TODO: check this iq or ikq
                # println(i, ' ', j, ' ', result)

                for k in 1:nbands*mesh^3
                    result += Uk[k,j]* conj(Uq[k,i]) * ϵₚ[k]
                    result -= Ukₘ[k,j]* conj(Uqₘ[k,i]) * ϵₚₘ[k]
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
                    # println(k, ' ',ϵₚ[k], ' ',Uk[k,j]* conj(Uq[k,i]), ' ', result)
                    # println(k, ' ',ϵₚₘ[k], ' ',Ukₘ[k,j]* conj(Uqₘ[k,i]), ' ', result)

                end

                braket[i,j] = result/2.0
            end
            # println("_____________________________________________________________")
            # exit(3)
        end


        push!(braket_list, transpose(conj(braket))*scale*mesh^3)

    end
    # println("Braket list unrotated")
    # println(scale*mesh^3)
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
        ωₐᵣᵣ, εₐᵣᵣ = parse_qe_ph(path_to_in*"scf_0/dyn1")
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
                        data = [iph, ω/cm1_to_Thz, real(gᵢⱼₘ)/ev_to_ry, imag(gᵢⱼₘ)/ev_to_ry]
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

        thr = 1e-4
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
            ωₐᵣᵣ_DFPT, _ = parse_qe_ph(path_to_in*"scf_0/dyn1")

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
            @info "Could not save symmetrized electron-phonon matrix elements, check if out folder exists in path_to_calc/displacements"
        end

        return symm_elph
    end

end

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


    # Create a scatter plot
    scatter(x, y, xlabel="g_DFPT", ylabel="g_frozen", title="Comparison", color = "red")
    line = LinRange(0, 1.1*maximum(max.(x,y)), 4)
    plot!(line, line, color = "black", legend = false)
    xlims!(0, 1.1*maximum(x))
    ylims!(0, 1.1*maximum(x))

    savefig(path_to_in*"out/comparison_$(ik)_$(iq).png")

    return x, y
end

function plot_ep_coupling(model::ModelQE, ik::Int=0, iq::Int=0; nbnd_max=-1)
    plot_ep_coupling(model.path_to_calc*"displacements/", ik, iq; nbnd_max=nbnd_max)
end

function electron_phonon(path_to_in::String, abs_disp, natoms, ik, iq, mesh, electrons::AbstractElectrons, phonons::AbstractPhonons; save_epw::Bool=false, path_to_calc="",kcw_chanel="")
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

    electron_phonon(path_to_in, abs_disp, natoms, ik, iq, mesh, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list , U_list, V_list, M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ; save_epw=save_epw)

end

function electron_phonon(model::AbstractModel, ik, iq, electrons::AbstractElectrons, phonons::AbstractPhonons; save_epw::Bool=false, path_to_calc="",kcw_chanel="")
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

    electron_phonon(model.path_to_calc*"displacements/", model.abs_disp, natoms, ik, iq, model.mesh, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list , U_list, V_list, M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ; save_epw=save_epw)

end
