using EzXML, WannierIO, LinearAlgebra, Printf,  YAML

function electron_phonon_qe(path_to_in::String)
    dir_name = "scf_0/"
    current_directory = pwd()
    cd(path_to_in*dir_name)

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
            "kx" =>  0.0, 
            "ky" =>  0.0, 
            "kz" =>  0.0, 
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
        write(f,"0.0000 0.0000 0.0000 1 # Gamma\n")
    end

    command = `/home/apolyukhin/Soft/sourse/q-e/test-suite/not_epw_comp/ph.x -in ph.in`
    println(command)
    run(pipeline(command, stdout="ph.out", stderr="errs_ph.txt"))
    cd(current_directory)
end

function calculate_braket_real(bra::Array{Complex{Float64}, 3}, ket::Array{Complex{Float64}, 3})
    Nxyz = size(ket, 1)^3
    result = sum(bra .* ket) / Nxyz
    return result
end

function calculate_braket_matrix_real(bras, kets)
    result = zeros(Complex{Float64}, length(bras), length(kets))
    
    for i in 1:length(bras)
        for j in 1:length(kets)
            result[i,j] = calculate_braket_real(bras["wfc$i"],kets["wfc$j"])
        end
    end

    return result
end

function calculate_braket(bra::Array{Complex{Float64}}, ket::Array{Complex{Float64}})
    result = sum(conj(bra) .* ket)
    return result
end

function calculate_braket_matrix(bras, kets)
    result = zeros(Complex{Float64}, length(bras), length(kets))
    
    for i in 1:length(bras)
        for j in 1:length(kets)
            result[i,j] = calculate_braket(bras[i],kets[j])
        end
    end

    return result
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
            if current_line > 2
                split_line = split(line)
                i, j, iph = parse(Int64,split_line[1]), parse(Int64,split_line[2]), parse(Int64,split_line[3])
                elph_dfpt[i,j,iph]= parse(Float64,split_line[end])/1e3
                #println(i,' ', j,' ', iph,' ', elph_dfpt[i,j,iph])
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

function electron_phonon(path_to_in::String, abs_disp, Ndisp)
    cd(path_to_in)
    command = `mkdir elph_elements`
    try
        run(command);
        println(command)
    catch; end
    Nat::Int = Ndisp//6
    ik = 1 

    ev_to_ry = 1 / 13.6057039763 
    scale = ev_to_ry / abs_disp

    path_to_xml="tmp/scf.save/data-file-schema.xml"
    path_to_wf = "wfc_list.jld2"
    group = "scf_0/"
    
    #real space
    #ψᵤ = load(path_to_in*group*path_to_wf)#
    #reciprocal space
    ik=1
    file_path=path_to_in*"/scf_0/tmp/scf.save/wfc$ik.dat"
    #file_path="/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/scf_0/tmp/silicon.save/wfc1.dat"
    miller, ψᵤ = parse_fortan_bin(file_path)

    nbands = length(ψᵤ)

    #println("ψᵤ unperturbed = ",ψᵤ["wfc1"][1:5])

    ϵᵤ = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][ik]
    #println("ϵᵤ unperturbed = ",ϵᵤ)

    braket = zeros(Complex{Float64}, nbands, nbands)
    braket_list = []
    braket_list_rotated = []

    #!!!! Need to go only through even or odd dispalcements
    for ind in 1:2:Ndisp
        group   = "group_$ind/"
        ϵₚ = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][ik]
        # ψₚ = load(path_to_in*group*path_to_wf)
        # U = calculate_braket_matrix_real(ψₚ, ψᵤ)

        file_path=path_to_in*"/group_$ind/tmp/scf.save/wfc$ik.dat"
        # if ind < 10
        #     file_path = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/group_$ind/tmp_00$ind/silicon.save/wfc1.dat"
        # else
        #     file_path = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/group_$ind/tmp_0$ind/silicon.save/wfc1.dat"
        # end
        miller, ψₚ = parse_fortan_bin(file_path)
        U = calculate_braket_matrix(ψₚ, ψᵤ)

        for i in 1:nbands
            for j in 1:nbands
                result = (i==j ? -ϵᵤ[i] : 0.0)
                #println(i, ' ', j, ' ', result)

                for k in 1:nbands
                    result += U[k,j]* conj(U[k,i]) * ϵₚ[k]
                    #println(k, ' ',ϵₚ[k], ' ',U[k,j]* conj(U[k,i]), ' ', result)
                end
                
                braket[i,j] = result
            end
        end
        push!(braket_list, braket*scale)
        #println("_____________________________________________________________")
    end
    #println("Braket list unrotated")
    #println(braket_list)

    phonon_params = phonopy.load("phonopy_params.yaml")
    displacements =phonon_params.displacements[pyslice(0,Ndisp,2)]

    for iat in 1:Nat
        U = []
        temp_iat::Int = 1 + 3 *(iat-1)
        for row_py in displacements[pyslice(temp_iat-1,temp_iat+2)]
            row = pyconvert(Vector,row_py)[2:end]
            push!(U,row/norm(row))
        end   
        U_inv =  vcat(U'...)^-1
        braket_temp = braket_list[temp_iat:temp_iat+2]
       
        push!(braket_list_rotated, U_inv* braket_temp) #transpose(U_inv) * braket_temp
    end

    #println("Braket list rotated")
    #println(braket_list_rotated)

    # # Multiplication by phonon eigenvector and phonon frequency
    # # Compute electron-phonon vertex in normal coordinate basis
    uma_to_ry = 911.44476959
    cm1_to_ry = 9.11259564445e-06

    εₐᵣᵣ = Array{ComplexF64, 3}(undef, (1, 3*Nat, 3*Nat))
    ωₐᵣᵣ = Array{Float64, 2}(undef, (1, 3*Nat))
    mₐᵣᵣ = pyconvert(Vector, phonon_params.masses)

    phonons = YAML.load_file("mesh.yaml")
    for (iband, phonon) in enumerate(phonons["phonon"][1]["band"])
        for iat in 1:Nat
            for icart in 1:3
                temp_iat::Int = icart + 3 *(iat-1)
                εₐᵣᵣ[1, iband, temp_iat] = phonon["eigenvector"][iat][icart][1] + 1im*phonon["eigenvector"][iat][icart][2]
            end
        end
        ωₐᵣᵣ[1, iband] = phonon["frequency"]
    end
     
    # εₐᵣᵣ = phonons["eigenvectors"]
    # ωₐᵣᵣ = phonons["frequencies"]
    # mₐᵣᵣ = phonon_params.masses

    # The problem is in the eigenvectors that are not as symertical as you could get with using phonopy withpu api
    # Need to figure 
    #So the problem seems to be in the phonopy API or the way I use it. temproray maybe better to use execution mode
    # println("εₐᵣᵣ[1,4,:] = $(εₐᵣᵣ[1,4,:])")
    # println("εₐᵣᵣ[1,5,:] = $(εₐᵣᵣ[1,5,:])")
    # println("εₐᵣᵣ[1,6,:] = $(εₐᵣᵣ[1,6,:])")
    # println("ωₐᵣᵣ[1,4] = $(ωₐᵣᵣ[1,4])")
    # println("ωₐᵣᵣ[1,5] = $(ωₐᵣᵣ[1,5])")
    # println("ωₐᵣᵣ[1,6] = $(ωₐᵣᵣ[1,6])")

    # εₐᵣᵣ[1,4,:]= [0.00000000000000 + 0.0*1im, 0.38454605536537+ 0.0*1im, -0.59340064990100+ 0.0*1im, 0.00000000000000 + 0.0*1im, -0.38454605536537+ 0.0*1im, 0.59340064990100+ 0.0*1im]
    # εₐᵣᵣ[1,5,:]= [0.00000000000000 + 0.0*1im, 0.59340064990100 + 0.0*1im, 0.38454605536537 + 0.0*1im,0.00000000000000 + 0.0*1im, -0.59340064990100 + 0.0*1im, -0.38454605536537 + 0.0*1im]
    # εₐᵣᵣ[1,6,:]= [0.70710678118655 + 0.0*1im, 0.00000000000000 + 0.0*1im, 0.00000000000000 + 0.0*1im, -0.70710678118655 + 0.0*1im, 0.00000000000000 + 0.0*1im, 0.00000000000000 + 0.0*1im]
    # ωₐᵣᵣ[1,4]  = 522.9906504387
    # ωₐᵣᵣ[1,5]  = 522.9906504387
    # ωₐᵣᵣ[1,6]  = 522.9906504387
    gᵢⱼₘ_ₐᵣᵣ = Array{ComplexF64, 3}(undef, (nbands, nbands, length(ωₐᵣᵣ)))

    for i in 1:nbands
        for j in 1:nbands
            open(path_to_in*"elph_elements/ep_$(i)_$(j)", "w") do io  
                for iph in 4:3*Nat
                    ω = ωₐᵣᵣ[1,iph] * cm1_to_ry
                    ε = εₐᵣᵣ[1,iph,:]
                    gᵢⱼₘ = 0.0
                    for iat in 1:Nat
                        braket_cart = braket_list_rotated[iat]
                        m = mₐᵣᵣ[iat] * uma_to_ry
                        disp =  sqrt(1/(2*m*ω))
                        for i_cart in 1:3
                            braket = braket_cart[i_cart]
                            temp_iat::Int = 3*(iat - 1) + i_cart
                            gᵢⱼₘ += disp*ε[temp_iat]*braket[i,j] 
                        end
                    end
                    gᵢⱼₘ_ₐᵣᵣ[i,j,iph] = gᵢⱼₘ*13.605
                    data = [iph, ω*0.124/cm1_to_ry, real(gᵢⱼₘ)*13.605, imag(gᵢⱼₘ)*13.605]
                    #@printf("  %5d  %10.6f  %10.6f   %10.6f\n", data...)
                    @printf(io, "  %5d  %10.6f  %10.6f   %10.6f\n", data...)
                end
                #@printf("____________________________________________\n")
            end
        end
    end 

    # Symmetrization
    symm_elph = zeros(ComplexF64,(nbands, nbands, length(ωₐᵣᵣ)))#gˢʸᵐᵢⱼₘ_ₐᵣᵣ
    elph = deepcopy(gᵢⱼₘ_ₐᵣᵣ)
    
    thr = 1e-3
    # symm through phonons
    for iph1 in 4:length(ωₐᵣᵣ)
        ω₁ = ωₐᵣᵣ[iph1]
        for ie in 1:nbands
            for je in 1:nbands
                n = 0
                g² = 0.0
                for iph2 in 4:length(ωₐᵣᵣ)
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
    for iph1 in 4:length(ωₐᵣᵣ)
        for je in 1:nbands
            for ie in 1:nbands
                n = 0
                g² = 0.0
                ε₁ = ϵᵤ[ie]
                for ie2 in 1:nbands
                    ε₂ = ϵᵤ[ie2]
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
    for iph1 in 4:length(ωₐᵣᵣ)
        for ie in 1:nbands
            for je in 1:nbands
                n = 0
                g² = 0.0
                ε₁ = ϵᵤ[je]
                for je2 in 1:nbands
                    ε₂ = ϵᵤ[je2]
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

    #!!!!!!!!!!!!!!!!!!#Need to figure out consistency with DFPT

    #read dfpt data 
    elph_dfpt = parse_ph(path_to_in*"scf_0/ph.out", nbands, length(ωₐᵣᵣ))

    #saving resulting electron phonon couplings 
    @printf("      i      j      nu    g_frozen   g_DFPT\n")
    open(path_to_in*"comparison", "w") do io 
    for i in 1:nbands
        for j in i:nbands
                for iph in 4:3*Nat
                    @printf("  %5d  %5d  %5d   %10.6f %10.6f\n", i,j,iph,symm_elph[i, j, iph], elph_dfpt[i, j, iph])
                    @printf(io, "  %5d  %5d  %5d   %10.6f %10.6f\n", i,j,iph,symm_elph[i, j, iph], elph_dfpt[i, j, iph])
                end
            end
        end
    end 

end
