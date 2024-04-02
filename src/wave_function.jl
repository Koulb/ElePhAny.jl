using FortranFiles, LinearAlgebra, Base.Threads, ProgressMeter, JLD2, FFTW

function wf_from_G_fft(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz::Integer)
    reciprocal_space_grid = zeros(ComplexF64, Nxyz, Nxyz, Nxyz)
    # Determine the shift needed to map Miller indices to grid indices
    shift = div(Nxyz, 2)

    # Iterate through Miller indices and fill in the known coefficients
    for idx in 1:size(miller, 2)
        g_vector = Int.(miller[:, idx])
        # Map the Miller indices to reciprocal space grid indices
        i, j, k = ((g_vector .+ shift) .% Nxyz) .+ 1
        #println(i,' ', j,' ', k)
        coefficient = evc[idx]
        reciprocal_space_grid[i, j, k] = coefficient
    end

    # Perform the inverse FFT to obtain the real-space wave function
    wave_function = bfft(ifftshift(reciprocal_space_grid))
    return wave_function
end

function wf_from_G_opt(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz::Integer)
    x = range(0, 1-1/Nxyz, Nxyz)
    y = range(0, 1-1/Nxyz, Nxyz)
    z = range(0, 1-1/Nxyz, Nxyz)

    wave_function = zeros(ComplexF64,(Nxyz, Nxyz, Nxyz))
    #progress = Progress(Nxyz^3, dt=1.0)
    
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
                #next!(progress)
            end
        end
    end
    return wave_function
end

function wf_from_G_opt_list(miller::Matrix{Int32}, evc_list::Vector{Any}, Nxyz::Integer)#
    x = range(0, 1-1/Nxyz, Nxyz)
    y = range(0, 1-1/Nxyz, Nxyz)
    z = range(0, 1-1/Nxyz, Nxyz)
    N_evc = size(evc_list)[1]

    wave_function = zeros(ComplexF64,(N_evc,Nxyz, Nxyz, Nxyz))
    progress = Progress(N_evc*Nxyz^3, dt=1.0)

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

                for (evc_index, evc) in enumerate(evc_list)
                    wave_function[evc_index,i, j, k] = dot(exponent, evc)
                    next!(progress)
                end
            end
        end
    end
    return wave_function
end

function parse_fortan_bin(file_path::String)
    f = FortranFile(file_path)
    ik, xkx, xky, xkz, ispin = read(f, (Int32,5))
    ngw, igwx, npol, nbnd = read(f, (Int32,4))
    dummy_vector = read(f, (Float64,9))
    miller = reshape(read(f, (Int32,3*igwx)),(3, igwx))

    evc_list = []
    for n in 1:nbnd
        evc = read(f, (ComplexF64,igwx))
        push!(evc_list,evc)
    end
    return miller, evc_list
end

function prepare_wave_functions(path_to_in::String; ik::Int=1,path_to_kcw="",kcw_chanel="")
    file_path = path_to_in*"/tmp/scf.save/wfc$ik.dat"

    if kcw_chanel != ""
        file_path = path_to_kcw*"/unperturbed/TMP/kc_kcw.save/wfc$(kcw_chanel)$(ik).dat"
        println("Wave functions from $file_path")
    end
    miller, evc_list = parse_fortan_bin(file_path)

    #Determine the fft grid
    potential_file = open(path_to_in*"/Vks", "r")
    dummy_line = readline(potential_file)
    fft_line = readline(potential_file)
    N = parse(Int64, split(fft_line)[1])

    println("Transforming wave fucntions in real space:")
    wfc_list = Dict()
    for (index, evc) in enumerate(evc_list)
        #println("band # $index")
        wfc = wf_from_G_fft(miller, evc, N)
        wfc_list["wfc$index"] = wfc

    end
    println("Data saved in "*path_to_in*"wfc_list_$ik.jld2")
    save(path_to_in*"/wfc_list_$ik.jld2",wfc_list)
    wfc_list = 0 # attempt to free memory
    GC.gc()
end

function prepare_wave_functions_opt(path_to_in::String; ik::Int=1)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik.dat"
    miller, evc_list = parse_fortan_bin(file_path)
    N_evc = size(evc_list)[1]

    #Determine the fft grid
    potential_file = open(path_to_in*"/Vks", "r")
    dummy_line = readline(potential_file)
    fft_line = readline(potential_file)
    N = parse(Int64, split(fft_line)[1])

    println("Transforming wave fucntions in real space:")
    wfc_data = wf_from_G_opt_list(miller, evc_list, N)
    wfc_list = Dict()
    for index in 1:N_evc
        wfc_list["wfc$index"] = wfc_data[index, :, :, :]
    end

    println("Data saved in "*path_to_in*"wfc_list_$ik.jld2")
    save(path_to_in*"wfc_list_$ik.jld2",wfc_list)

end

#check that I read the correct wf for ik = 2 
function unfold_to_sc(path_to_in::String, mesh::Int, ik::Int)
    #Determine the fft grid
    potential_file = open(path_to_in*"/Vks", "r")
    dummy_line = readline(potential_file)
    fft_line = readline(potential_file)
    Nxyz = parse(Int64, split(fft_line)[1]) * mesh

    q_vector = determine_q_point(path_to_in,ik).* mesh
    println("q_vector = $q_vector")

    x = range(0, 1-1/Nxyz, Nxyz)
    y = range(0, 1-1/Nxyz, Nxyz)
    z = range(0, 1-1/Nxyz, Nxyz)

    exp_factor = zeros(Complex{Float64}, Nxyz, Nxyz, Nxyz)
    println("Unfolding wave function to sc")

    @threads for i in eachindex(x)  
        for j in eachindex(y)
            for k in eachindex(z)
                x_i = x[i]
                y_j = y[j]
                z_k = z[k]

                r_ijk = [x_i, y_j, z_k]
                temp = dot(transpose(r_ijk), q_vector)
                exp_factor[i, j, k] = exp.(2im * π * temp)
            end
        end
    end

    wfc_list_old = load(path_to_in*"wfc_list_$ik.jld2")
    N_evc = length(wfc_list_old)

    wfc_list = Dict()
    for index in 1:N_evc
        wfc = wfc_list_old["wfc$index"]
        wfc = repeat(wfc, outer=(mesh, mesh, mesh))
        wfc = wfc .* exp_factor 
        wfc_list["wfc$index"]  = wfc
    end

    println("Data saved in "*path_to_in*"wfc_list_phase_$ik.jld2")
    save(path_to_in*"wfc_list_phase_$ik.jld2",wfc_list)
end

function prepare_wave_functions_all(path_to_in::String, ik::Int, iq::Int, mesh::Int, Ndisp::Int)
    file_path=path_to_in*"/scf_0/"
    #prepare_wave_functions_opt(file_path;ik=ik)
    prepare_wave_functions(file_path;ik=ik)

    #need to fold in 1st Bz in case of arbitrary q
    k_list = get_kpoint_list(file_path)
    ikq = fold_kpoint(ik,iq,k_list)
    #prepare_wave_functions_opt(file_path;ik=ikq)
    prepare_wave_functions(file_path;ik=ikq)
    #Need to unfold from pc to sc and mulitply wavefunctions by exp(ikr)
    if mesh > 1
        unfold_to_sc(file_path,mesh,ik)
        unfold_to_sc(file_path,mesh,ikq)
    end

    # TODO check consistency between convetional, fft and braket.x (Why braket(wf_1,wf_1_fft)=0???)
    for i in 1:2:Ndisp
        file_path=path_to_in*"/group_$i/"
        if mesh > 1
            #prepare_wave_functions_opt(file_path)
            prepare_wave_functions(file_path,ik=1)
        else
            prepare_wave_functions(file_path,ik=ik)
        end    
    end

end

function wave_functions_to_G(path_to_in::String; ik::Int=1)
    wfc_list = load(path_to_in*"/scf_0/wfc_list_phase_$ik.jld2")
    Nxyz = size(wfc_list["wfc1"], 1)
    miller_sc, _ = parse_fortan_bin(path_to_in*"/group_1/tmp/scf.save/wfc1.dat") 
    Nevc = size(miller_sc, 2)
    g_list = Dict()

    println("Tranforming wave function to G space:")

    for (key, wfc) in wfc_list
        evc_sc = zeros(ComplexF64, size(miller_sc, 2))
        wfc_g = fftshift(fft(wfc))
        shift = div(Nxyz, 2)
        for idx in 1:Nevc
            g_vector = Int.(miller_sc[:, idx])
            i, j, k = ((g_vector .+ shift) .% Nxyz) .+ 1
            evc_sc[idx] = wfc_g[i, j, k]
        end

        norm = sqrt(1/calculate_braket(evc_sc,evc_sc))
        evc_sc = evc_sc .* norm
        g_list[key] = evc_sc
    end

    println("Data saved in "*path_to_in*"g_list_sc_$ik.jld2")
    save(path_to_in*"/scf_0/g_list_sc_$ik.jld2", g_list)
end

function prepare_wave_functions_undisp(path_to_in::String, ik::Int, mesh::Int;path_to_kcw=path_to_kcw,kcw_chanel=kcw_chanel)
    file_path=path_to_in*"/scf_0/"

    if mesh > 1
        #prepare_wave_functions_opt(file_path;ik=ik)
        prepare_wave_functions(file_path;ik=ik,path_to_kcw=path_to_kcw,kcw_chanel=kcw_chanel)
        unfold_to_sc(file_path,mesh,ik)
        wave_functions_to_G(path_to_in;ik=ik)
    end

end

function prepare_wave_functions_undisp(path_to_in::String, ik::Int, iq::Int, mesh::Int)
    file_path=path_to_in*"/scf_0/"

    if mesh > 1
        #prepare_wave_functions_opt(file_path;ik=ik)
        prepare_wave_functions(file_path;ik=ik)
        unfold_to_sc(file_path,mesh,ik)
        wave_functions_to_G(path_to_in;ik=ik)
        
        k_list = get_kpoint_list(file_path)
        ikq = fold_kpoint(ik,iq,k_list)
        if ikq != ik
            prepare_wave_functions(file_path;ik=ikq)
            unfold_to_sc(file_path,mesh,ikq)
            wave_functions_to_G(path_to_in;ik=ikq)
        end    
    end

end


function prepare_wave_functions_undisp(path_to_in::String, mesh::Int;path_to_kcw="",kcw_chanel="")
    for ik in 1:mesh^3
        prepare_wave_functions_undisp(path_to_in,ik,mesh;path_to_kcw=path_to_kcw,kcw_chanel=kcw_chanel)
        println("ik = $ik/$(mesh^3) is ready")
    end
end

function prepare_u_matrixes(path_to_in::String, Ndisplace::Int, mesh::Int)
    U_list = []
    V_list = []

    println("Preparing u matrixes:")
    for ind in 1:2:Ndisplace
        println("group_$ind")
        group   = "group_$ind/"
        group_m   = "group_$(ind+1)/"
        _, ψₚ = parse_fortan_bin(path_to_in*group*"tmp/scf.save/wfc1.dat")
        _, ψₚₘ = parse_fortan_bin(path_to_in*group_m*"tmp/scf.save/wfc1.dat")
        nbnds = Int(size(ψₚ)[1]/mesh^3)
        
        Uₖᵢⱼ = zeros(ComplexF64, mesh^3, nbnds*mesh^3, nbnds)
        Vₖᵢⱼ = zeros(ComplexF64, mesh^3, nbnds*mesh^3, nbnds)

        for ik in 1:mesh^3
            ψkᵤ_list = load(path_to_in*"/scf_0/g_list_sc_$ik.jld2")
            ψkᵤ = [ψkᵤ_list["wfc$iband"] for iband in 1:length(ψkᵤ_list)]

            Uₖᵢⱼ[ik, :, :] = calculate_braket_matrix(ψₚ, ψkᵤ)
            Vₖᵢⱼ[ik, :, :] = calculate_braket_matrix(ψₚₘ, ψkᵤ)
        end

        push!(U_list, Uₖᵢⱼ)
        push!(V_list, Vₖᵢⱼ)
    end

    # Save U_list to a text file
    writedlm(path_to_in*"/scf_0/U_list.txt", U_list)  
    writedlm(path_to_in*"/scf_0/V_list.txt", V_list)    

    return U_list, V_list
end


#TEST
#path_to_in = "/home/apolyukhin/Development/julia_tests/qe_inputs/displacements/"
# path_to_in = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/group_1/tmp_001/silicon.save"
# N = 72
# ik = 1
# prepare_wave_functions_opt(path_to_in, ik,N)
