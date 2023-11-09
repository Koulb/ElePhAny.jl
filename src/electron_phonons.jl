using EzXML, WannierIO, LinearAlgebra, Printf,  YAML, Plots

function electron_phonon_qe(path_to_in::String, ik::Int, iq::Int)
    dir_name = "scf_0/"
    current_directory = pwd()
    cd(path_to_in*dir_name)

    kpoint = determine_q_point_cart(path_to_in*dir_name,ik)
    qpoint = determine_q_point_cart(path_to_in*dir_name,iq)

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

    command = `/home/apolyukhin/Soft/sourse/q-e/test-suite/not_epw_comp/ph.x -in ph.in`
    println(command)
    run(pipeline(command, stdout="ph.out", stderr="errs_ph.txt"))
    cd(current_directory)
end

function calculate_braket_real(bra::Array{Complex{Float64}, 3}, ket::Array{Complex{Float64}, 3})
    Nxyz = size(ket, 1)^3
    result = sum(conj(bra) .* ket) / Nxyz
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

#For sc case need to modify eigenvetors with right phase + brakets should be in real space
function electron_phonon(path_to_in::String, abs_disp, Ndisp, ik, iq, mesh)
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
    group = "scf_0/"
    local ψkᵤ, ψqᵤ, ψᵤ, nbands
    if mesh > 1
        #real space
        ψkᵤ = load(path_to_in*group*"wfc_list_phase_$ik.jld2")#
        ψqᵤ = load(path_to_in*group*"wfc_list_phase_$iq.jld2")

        #Debug read from python np 
        #temp_path = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/"
        #ψkᵤ = load_wf_u_debug(temp_path*group,ik)
        #ψqᵤ = load_wf_u_debug(temp_path*group,iq)
        nbands = length(ψkᵤ)
    else
        #reciprocal space
        ik=1
        file_path=path_to_in*"/scf_0/tmp/scf.save/wfc$ik.dat"
        miller, ψᵤ = parse_fortan_bin(file_path)
        nbands = length(ψᵤ)
    end


    ϵkᵤ = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][ik]
    ϵqᵤ = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][iq]

    braket = zeros(Complex{Float64}, nbands, nbands)
    braket_list = []
    braket_list_rotated = []

    #!!!! Need to go only through even or odd dispalcements
    for ind in 1:2:Ndisp
        group   = "group_$ind/"
        ϵₚ = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][ik]
        file_path=path_to_in*"/group_$ind/tmp/scf.save/wfc$ik.dat"
       
        local ψₚ, Uk, Uq 

        if mesh > 1 
            temp_path = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/"
            #ψₚ = load_wf_debug(temp_path*group)
            ψₚ = load(path_to_in*group*"wfc_list_1.jld2")

            Uk = calculate_braket_matrix_real(ψₚ, ψkᵤ)
            Uq = calculate_braket_matrix_real(ψₚ, ψqᵤ)
        else 
            miller, ψₚ = parse_fortan_bin(file_path)
            Uk = calculate_braket_matrix(ψₚ, ψᵤ)
            Uq = Uq    
        end

        for i in 1:nbands
            for j in 1:nbands
                result = (i==j && ik==iq ? -ϵkᵤ[i] : 0.0)
                #println(i, ' ', j, ' ', result)

                for k in 1:nbands*mesh^3
                    result += Uk[k,j]* conj(Uq[k,i]) * ϵₚ[k]
                    #println(k, ' ',ϵₚ[k], ' ',Uk[k,j]* conj(Uq[k,i]), ' ', result)
                end
                
                braket[i,j] = result
            end
            #println("_____________________________________________________________")
            #exit(3)
        end
        push!(braket_list, transpose(conj(braket))*scale*mesh^3)

    end
    println("Braket list unrotated")
    #println(braket_list)

    phonon_params = phonopy.load("phonopy_params.yaml")
    displacements = phonon_params.displacements[pyslice(0,Ndisp,2)]

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

    ## Multiplication by phonon eigenvector and phonon frequency
    ## Compute electron-phonon vertex in normal coordinate basis
    uma_to_ry = 911.44476959
    cm1_to_ry = 9.11259564445e-06
    phonons = YAML.load_file("qpoints.yaml")

    εₐᵣᵣ = Array{ComplexF64, 3}(undef, (1, 3*Nat, 3*Nat))
    ωₐᵣᵣ = Array{Float64, 2}(undef, (1, 3*Nat))
    mₐᵣᵣ = pyconvert(Vector, phonon_params.masses)
    qpoint = determine_q_point(path_to_in*"scf_0/",iq)
    scaled_pos = pyconvert(Matrix, phonon_params.primitive.get_scaled_positions())
    phonon_factor = [exp(2im * π * dot(qpoint, pos)) for pos in eachrow(scaled_pos)]

    for (iband, phonon) in enumerate(phonons["phonon"][1]["band"])
        for iat in 1:Nat
            for icart in 1:3
                temp_iat::Int = icart + 3 *(iat-1)
                eig_temp = phonon["eigenvector"][iat][icart][1] + 1im*phonon["eigenvector"][iat][icart][2]
                εₐᵣᵣ[1, iband, temp_iat] = phonon_factor[iat]*eig_temp
            end
        end
        ωₐᵣᵣ[1, iband] = phonon["frequency"]
    end
     
    gᵢⱼₘ_ₐᵣᵣ = Array{ComplexF64, 3}(undef, (nbands, nbands, length(ωₐᵣᵣ)))

    for i in 1:nbands
        for j in 1:nbands
            open(path_to_in*"elph_elements/ep_$(i)_$(j)", "w") do io  
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
    elph_dfpt = parse_ph(path_to_in*"scf_0/ph.out", nbands, length(ωₐᵣᵣ))

    #saving resulting electron phonon couplings 
    @printf("      i      j      nu    g_frozen   g_DFPT\n")
    open(path_to_in*"comparison", "w") do io 
    for i in 1:nbands
        for j in i:nbands
                for iph in 1:3*Nat
                    @printf("  %5d  %5d  %5d   %10.6f %10.6f\n", i,j,iph,symm_elph[i, j, iph], elph_dfpt[i, j, iph])
                    @printf(io, "  %5d  %5d  %5d   %10.6f %10.6f\n", i,j,iph,symm_elph[i, j, iph], elph_dfpt[i, j, iph])
                end
            end
        end
    end 

end

function plot_ep_coupling(path_to_file::String)
    # Initialize empty arrays for x and y
    x = Float64[]
    y = Float64[]
    filename = path_to_file*"comparison"
    # Read data from the file and populate x and y arrays
    open(filename, "r") do file
        for line in eachline(file)
            # Split the line into columns
            columns = split(line)
            if length(columns) >= 5
                # Extract the last two columns and convert them to Float64
                x_val = parse(Float64, columns[4])
                y_val = parse(Float64, columns[5])
                push!(x, x_val)
                push!(y, y_val)
            end
        end
    end

    # Create a scatter plot
    scatter(x, y, xlabel="g_frozen", ylabel="g_DFPT", title="Comparison", color = "red")
    line = LinRange(0, 1.1*maximum(max.(x,y)), 4)
    plot!(line, line, color = "black", legend = false)
    savefig(path_to_file*"comparison.png")
end
