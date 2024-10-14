using FortranFiles, LinearAlgebra, Base.Threads, ProgressMeter, JLD2, FFTW

function parse_fortan_bin(file_path::String)
    f = FortranFile(file_path)
    ik, xkx, xky, xkz, ispin = read(f, (Int32,5))
    ngw, igwx, npol, nbnd = read(f, (Int32,4))
    dummy_vector = read(f, (Float64,9))
    miller = reshape(read(f, (Int32,3*igwx)),(3, igwx))

    evc_list = []
    for _ in 1:nbnd
        evc = read(f, (ComplexF64,igwx))
        push!(evc_list,evc)
    end
    return miller, evc_list
end

function wf_from_G(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz::Integer)
    reciprocal_space_grid = zeros(ComplexF64, Nxyz, Nxyz, Nxyz)
    # Determine the shift needed to map Miller indices to grid indices
    shift = div(Nxyz, 2)

    # Iterate through Miller indices and fill in the known coefficients
    for idx in 1:size(miller, 2)
        g_vector = Int.(miller[:, idx])
        # Map the Miller indices to reciprocal space grid indices
        i, j, k = ((g_vector .+ shift) .% Nxyz) .+ 1
        coefficient = evc[idx]
        reciprocal_space_grid[i, j, k] = coefficient
    end

    # Perform the inverse FFT to obtain the real-space wave function
    wave_function = bfft(ifftshift(reciprocal_space_grid))
    return wave_function
end

function wf_from_G_slow(miller::Matrix{Int32}, evc::Vector{ComplexF64}, Nxyz::Integer)
    x = range(0, 1-1/Nxyz, Nxyz)
    y = range(0, 1-1/Nxyz, Nxyz)
    z = range(0, 1-1/Nxyz, Nxyz)

    wave_function = zeros(ComplexF64,(Nxyz, Nxyz, Nxyz))

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
            end
        end
    end
    return wave_function
end

function wf_to_G(miller::Matrix{Int32}, wfc, Nxyz::Integer)
    Nevc = size(miller, 2)

    evc_sc = zeros(ComplexF64, size(miller, 2))
    wfc_g = fftshift(fft(wfc))
    shift = div(Nxyz, 2)
    for idx in 1:Nevc
        g_vector = Int.(miller[:, idx])
        i, j, k = ((g_vector .+ shift) .% Nxyz) .+ 1
        evc_sc[idx] = wfc_g[i, j, k]
    end

    #TODO Is it possible to not calculate this norm? Could reduce the computational cost
    norm = sqrt(1/calculate_braket(evc_sc,evc_sc))
    evc_sc = evc_sc .* norm

    return evc_sc
end

function wf_pc_to_sc(wfc, mesh)
    wfc_sc = repeat(wfc, outer=(mesh, mesh, mesh))
    return wfc_sc
end

function determine_fft_grid(path_to_file::String)
    #Determine the fft grid
    scf_file = open(path_to_file, "r")
    fft_line = ""
    for line in eachline(scf_file)
        if contains(line, "FFT dimensions:")
            fft_line = line
            break
        end
    end
    close(scf_file)

    Nxyz = parse(Int64, split(fft_line)[8][1:end-1])
    return Nxyz
end

function determine_phase(q_point, Nxyz)
    x = range(0, 1-1/Nxyz, Nxyz)
    y = range(0, 1-1/Nxyz, Nxyz)
    z = range(0, 1-1/Nxyz, Nxyz)

    exp_factor = zeros(Complex{Float64}, Nxyz, Nxyz, Nxyz)
    @threads for i in eachindex(x)
        for j in eachindex(y)
            for k in eachindex(z)
                x_i = x[i]
                y_j = y[j]
                z_k = z[k]

                r_ijk = [x_i, y_j, z_k]
                temp = dot(transpose(r_ijk), q_point)
                exp_factor[i, j, k] = exp.(2im * π * temp)
            end
        end
    end

    return exp_factor
end

function prepare_unfold_to_sc(path_to_in::String, mesh::Int, ik::Int)
    Nxyz = determine_fft_grid(path_to_in*"/scf.out") * mesh
    q_vector = determine_q_point(path_to_in, ik; mesh = mesh)
    exp_factor = determine_phase(q_vector, Nxyz)

    wfc_list_old = load(path_to_in*"wfc_list_$ik.jld2")
    N_evc = length(wfc_list_old)

    wfc_list = Dict()
    for index in 1:N_evc
        wfc = wfc_list_old["wfc$index"]
        wfc = wf_pc_to_sc(wfc, mesh)
        wfc = wfc .* exp_factor
        wfc_list["wfc$index"]  = wfc
    end

    @info "Data saved in "*path_to_in*"wfc_list_phase_$ik.jld2"
    save(path_to_in*"wfc_list_phase_$ik.jld2",wfc_list)
end

function prepare_wave_functions_to_R(path_to_in::String; ik::Int=1)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik.dat"
    miller, evc_list = parse_fortan_bin(file_path)

    N = determine_fft_grid(path_to_in*"/scf.out")

    wfc_list = Dict()
    for (index, evc) in enumerate(evc_list)
        wfc = wf_from_G(miller, evc, N)
        wfc_list["wfc$index"] = wfc
    end

    @info "Data saved in "*path_to_in*"wfc_list_$ik.jld2"
    save(path_to_in*"/wfc_list_$ik.jld2",wfc_list)
    wfc_list = 0 # attempt to free memory
    GC.gc()
end

function prepare_wave_functions_to_G(path_to_in::String; ik::Int=1)
    wfc_list = load(path_to_in*"/scf_0/wfc_list_phase_$ik.jld2")
    Nxyz = size(wfc_list["wfc1"], 1)
    miller_sc, _ = parse_fortan_bin(path_to_in*"/group_1/tmp/scf.save/wfc1.dat")

    g_list = Dict()
    for (key, wfc) in wfc_list
        g_list[key] = wf_to_G(miller_sc, wfc, Nxyz)
    end

    @info "Data saved in "*path_to_in*"/scf_0/g_list_sc_$ik.jld2"
    save(path_to_in*"/scf_0/g_list_sc_$ik.jld2", g_list)
end

function prepare_wave_functions_undisp(path_to_in::String, ik::Int, mesh::Int)
    file_path=path_to_in*"/scf_0/"

    if mesh > 1
        @info "Tranforming wave functions to R space:"
        prepare_wave_functions_to_R(file_path;ik=ik)
        @info "Unfolding wave functions to supercell:"
        prepare_unfold_to_sc(file_path,mesh,ik)
        @info "Tranforming wave functions to G space:"
        prepare_wave_functions_to_G(path_to_in;ik=ik)
    end

end

function prepare_wave_functions_undisp(path_to_in::String, mesh::Int)
    for ik in 1:mesh^3
        prepare_wave_functions_undisp(path_to_in,ik,mesh)
        @info "ik = $ik/$(mesh^3) is ready"
    end
end

function prepare_u_matrixes(path_to_in::String, natoms::Int, mesh::Int; symmetries::Symmetries = Symmetries([], [], []), save_matrixes::Bool=true)
    U_list = []
    V_list = []

    Ndisplace_nosym = 6 * natoms

    N_fft = determine_fft_grid(path_to_in*"group_1/scf.out")
    ψₚ0_list = []
    ψₚ0_real_list = []
    miller_list = []
    for (ind, _) in enumerate(unique(symmetries.ineq_atoms_list))
        println("Preparing wave functions for group_$ind:")
        miller1, ψₚ0 = parse_fortan_bin(path_to_in*"/group_$ind/tmp/scf.save/wfc1.dat")
        ψₚ0_real = [wf_from_G(miller1, evc, N_fft) for evc in ψₚ0]
        push!(ψₚ0_list, ψₚ0)
        push!(ψₚ0_real_list, ψₚ0_real)
        push!(miller_list, miller1)
    end

    println("Preparing u matrixes:")
    for ind in 1:Ndisplace_nosym
        ψₚ = []

        local tras, rot
        #check if symmetries are empty
        if isempty(symmetries.trans_list) && isempty(symmetries.rot_list)
            tras = [0.0,0.0,0.0]
            rot  = [[1.0,0.0,0.0] [0.0,1.0,0.0] [0.0,0.0,1.0]]
            append!(symmetries.ineq_atoms_list, ind)
        else
            tras  = symmetries.trans_list[ind] #./mesh
            rot   = symmetries.rot_list[ind]
        end

        if all(isapprox.(tras,[0.0,0.0,0.0], atol = 1e-15)) &&
           all(isapprox.(rot, [[1.0,0.0,0.0] [0.0,1.0,0.0] [0.0,0.0,1.0]], atol = 1e-15))
            _, ψₚ = parse_fortan_bin(path_to_in*"/group_$(symmetries.ineq_atoms_list[ind])/tmp/scf.save/wfc1.dat")
        else
            ψₚ0_real = ψₚ0_real_list[symmetries.ineq_atoms_list[ind]]
            miller1 = miller_list[symmetries.ineq_atoms_list[ind]]
            map1 = rotate_grid(N_fft, N_fft, N_fft, rot, tras)
            ψₚ_real = [rotate_deriv(N_fft, N_fft, N_fft, map1, wfc) for wfc in ψₚ0_real]
            ψₚ = [wf_to_G(miller1, evc, N_fft) for evc in ψₚ_real]
        end

        nbnds = Int(size(ψₚ)[1]/mesh^3)

        Uₖᵢⱼ = zeros(ComplexF64, mesh^3, nbnds*mesh^3, nbnds)

        for ik in 1:mesh^3
            ψkᵤ_list = load(path_to_in*"/scf_0/g_list_sc_$ik.jld2")
            ψkᵤ = [ψkᵤ_list["wfc$iband"] for iband in 1:length(ψkᵤ_list)]

            Uₖᵢⱼ[ik, :, :] = calculate_braket(ψₚ, ψkᵤ)
	    @info ("ik = $ik")
        end

        if isodd(ind)
            push!(U_list, Uₖᵢⱼ)
        else
            push!(V_list, Uₖᵢⱼ)
        end
        @info ("group_$ind is ready")
    end

    # Save U_list to a hdf5-like file
    if save_matrixes == true
        save(path_to_in * "scf_0/U_list.jld2", "U_list", U_list)
        save(path_to_in * "scf_0/V_list.jld2", "V_list", V_list)
    end

    return U_list, V_list
end

function calculate_braket_real(bra::Array{Complex{Float64}, 3}, ket::Array{Complex{Float64}, 3})
    Nxyz = size(ket, 1)^3
    result = zero(Complex{Float64})

    @inbounds @simd for i in 1:Nxyz
        result += conj(bra[i]) * ket[i]
    end

    result /= Nxyz
    return result
end

function calculate_braket_real(bras::Dict{String, Array{Complex{Float64}, 3}}, kets::Dict{String, Array{Complex{Float64}, 3}})
    result = zeros(Complex{Float64}, length(bras), length(kets))

    for (i, bra) in enumerate(values(bras))
        for (j, ket) in enumerate(values(kets))
            result[i,j] = calculate_braket_real(bra, ket)
        end
    end

    return result
end

function calculate_braket(bra::Array{Complex{Float64}}, ket::Array{Complex{Float64}})
    Nevc = length(bra)
    result = zero(Complex{Float64})

    @inbounds @simd for i in 1:Nevc
        result += conj(bra[i]) * ket[i]
    end

    return result
end

function calculate_braket(bras, kets)
    result = zeros(Complex{Float64}, length(bras), length(kets))

    @threads for i in eachindex(bras)
        for j in eachindex(kets)
            result[i,j] = calculate_braket(bras[i],kets[j])
        end
    end

    return result
end


#TEST
#path_to_in = "/home/apolyukhin/Development/julia_tests/qe_inputs/displacements/"
# path_to_in = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/group_1/tmp_001/silicon.save"
# N = 72
# ik = 1
# prepare_wave_functions_opt(path_to_in, ik,N)
