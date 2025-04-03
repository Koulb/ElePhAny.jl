using FortranFiles, LinearAlgebra, Base.Threads, ProgressMeter, JLD2, FFTW, HDF5

function parse_wf(path::String)
    evc_list = []
    miller = []

    #check if path*".dat" or path*".hdf5" exists
    if isfile(path*".dat")
        miller, evc_list = parse_fortran_bin(path*".dat")
    elseif isfile(path*".hdf5")
        miller, evc_list = parse_hdf(path*".hdf5")
    else
        error("File not found: $path")
    end
    return miller, evc_list
end

function parse_hdf(path::String)
    evc_list = []
    miller = []

    miller, evc_list = h5open(path, "r") do f
        fkeys = collect(keys(f))
        miller_key = fkeys[1]
        evc_key = fkeys[2]
        data = read(f)
        miller = data[miller_key]

        evc_list_raw = data[evc_key]
        nbands = size(data[evc_key], 2)
        evc_list = []
        for ind in 1:nbands
            evc_real = evc_list_raw[1:2:end, ind]
            evc_imag = evc_list_raw[2:2:end, ind]
            push!(evc_list, evc_real .+ im * evc_imag)
        end

        (miller,evc_list)
    end

    return miller, evc_list
end

function parse_fortran_bin(file_path::String)
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

function wf_from_G_list(miller::Matrix{Int32}, evc_list::AbstractArray{Any}, Nxyz::Integer)
    evc_matrix = permutedims(hcat(evc_list...))
    N_evc = size(evc_matrix)[1]
    reciprocal_space_grid = zeros(ComplexF64, N_evc, Nxyz, Nxyz, Nxyz)
    wave_function         = zeros(ComplexF64, N_evc, Nxyz, Nxyz, Nxyz)

    # Determine the shift needed to map Miller indices to grid indices
    shift = div(Nxyz, 2)

    @threads for idx in 1:size(miller, 2)
        i = (Int(miller[1, idx]) + shift) % Nxyz + 1
        j = (Int(miller[2, idx]) + shift) % Nxyz + 1
        k = (Int(miller[3, idx]) + shift) % Nxyz + 1
        reciprocal_space_grid[:, i, j, k] .= evc_matrix[:,idx]
    end

    wave_function_raw = ifftshift(reciprocal_space_grid, (2, 3, 4))
    reciprocal_space_grid = nothing;

    wave_function = ifft(wave_function_raw, (2, 3, 4))
    wave_function_raw = nothing

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


function wf_to_G_list(miller::Matrix{Int32}, wfc::AbstractArray{ComplexF64, 4}, Nxyz::Integer)
    Ng = size(miller, 2)
    Nevc = size(wfc, 1)

    evc_sc = zeros(ComplexF64, (Nevc, Ng))
    wfc_g_raw = fft(wfc, (2, 3, 4))
    wfc_g = fftshift(wfc_g_raw, (2, 3, 4))
    wfc_g_raw = nothing

    shift = div(Nxyz, 2)
    for idx in 1:Ng
        i = (Int(miller[1, idx]) + shift) % Nxyz + 1
        j = (Int(miller[2, idx]) + shift) % Nxyz + 1
        k = (Int(miller[3, idx]) + shift) % Nxyz + 1
        evc_sc[:,idx] = wfc_g[:,i, j, k]
    end

    wfc_g = nothing

    #TODO Is it possible to not calculate this norm? Could reduce the computational cost
    # norm = sqrt(1/calculate_braket(evc_sc,evc_sc))
    # evc_sc = evc_sc .* norm

    return evc_sc
end

function wf_pc_to_sc(wfc, sc_size)
    wfc_sc = repeat(wfc, outer=(sc_size, sc_size, sc_size))
    return wfc_sc
end

function determine_fft_grid(path_to_file::String; use_xml::Bool = false)
    Nxyz = 0
    if use_xml
        # Parse the XML file
        doc = EzXML.readxml(path_to_file)

        # Navigate to the `fft_grid` node
        fft_grid_node = findfirst("/qes:espresso/output/basis_set/fft_grid", root(doc))

        if fft_grid_node === nothing
            error("fft_grid section not found in the XML file.")
        end

        # # Extract the attributes
        nr1 = parse(Int, fft_grid_node["nr1"])
        nr2 = parse(Int, fft_grid_node["nr2"])
        nr3 = parse(Int, fft_grid_node["nr3"])

        Nxyz = nr1
    else
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
    end

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

function prepare_unfold_to_sc(path_to_in::String, sc_size::Int, ik::Int)
    Nxyz = determine_fft_grid(path_to_in*"/scf.out") * sc_size
    q_vector = determine_q_point(path_to_in, ik; sc_size = sc_size)
    exp_factor = determine_phase(q_vector, Nxyz)

    wfc_list_old = load(path_to_in*"wfc_list_$ik.jld2")
    N_evc = length(wfc_list_old)

    wfc_list = Dict()
    for index in 1:N_evc
        wfc = wfc_list_old["wfc$index"]
        wfc = wf_pc_to_sc(wfc, sc_size)
        wfc = wfc .* exp_factor
        wfc_list["wfc$index"]  = wfc
    end

    @info "Data saved in "*path_to_in*"wfc_list_phase_$ik.jld2"
    save(path_to_in*"wfc_list_phase_$ik.jld2",wfc_list)
end

function wf_phase!(path_to_in::String, wfc::AbstractArray{ComplexF64, 4}, sc_size::Int, ik::Int)
    Nxyz = determine_fft_grid(path_to_in*"/scf_0/scf.out") * sc_size
    q_vector = determine_q_point(path_to_in*"/scf_0/", ik; sc_size = 1, use_sc = true)
    exp_factor = determine_phase(q_vector, Nxyz)

    N_evc = size(wfc)[1]
    for iband in 1:N_evc
        wfc[iband,:,:,:] = wfc[iband,:,:,:] .* exp_factor
    end

    return nothing
end

function prepare_wave_functions_to_R(path_to_in::String; ik::Int=1)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik"
    miller, evc_list = parse_wf(file_path)

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
    miller_sc, _ = parse_wf(path_to_in*"/group_1/tmp/scf.save/wfc1")

    g_list = Dict()
    for (key, wfc) in wfc_list
        g_list[key] = wf_to_G(miller_sc, wfc, Nxyz)
    end

    @info "Data saved in "*path_to_in*"/scf_0/g_list_sc_$ik.jld2"
    save(path_to_in*"/scf_0/g_list_sc_$ik.jld2", g_list)
end

#Block with new implementation of Folding/Unfolding

function is_within_cutoff(hi, ki, li, kpt, cutoff_radius)
    h = hi + kpt[1]
    k = ki + kpt[2]
    l = li + kpt[3]

    G_squared = 3 * (h^2 + k^2 + l^2) - 2 * (h * k + h * l + k * l)
    return G_squared <= cutoff_radius^2
end

function create_miller_index(a, Ecut, mesh_scale, kpt)
    # Constants
    a_new = mesh_scale*a  # Lattice constant in Ångstroms
    ecutoff = Ecut  # Kinetic energy cutoff in Ha

    # Reciprocal lattice vector magnitude in Å⁻¹
    g_mag = 2 * π / a_new
    miller_new_raw = []

    # Calculate cutoff radius in reciprocal lattice units
    cutoff_radius = sqrt(2 * ecutoff) / 0.529 / g_mag
    kpt_new = kpt .* mesh_scale

    # Collect points inside the cutoff
    max_index = Int(ceil(cutoff_radius))
    for h in -max_index:max_index
        for k in -max_index:max_index
            for l in -max_index:max_index
                if is_within_cutoff(h, k, l, kpt_new, cutoff_radius)
                    push!(miller_new_raw, [h, k, l])
                end
            end
        end
    end

    miller_new::Matrix{Int32} = hcat(miller_new_raw...)
    return miller_new
end

function create_unified_Grid(path_to_dat, a, ecutoff, mesh_scale )

    ## Create a grid comensurate with SCII

    ## TODO Come up with a better way to determine the cutoff
    ecutoff = ecutoff + 5#65.0/2#(ecutoff + 5) / 2 #??

    miller_list_pc_raw  = [0;0;0]

    for ik in 1:mesh_scale^3
        kpt =  ElectronPhonon.determine_q_point(path_to_dat*"scf_0", ik)
        miller_pc_ik =  create_miller_index(a, ecutoff, mesh_scale, kpt)
        miller_list_pc_raw = hcat(miller_list_pc_raw, miller_pc_ik)
    end

    #conver matrix 3xN to vector of points
    miller_list_pc_raw_points = [miller_list_pc_raw[:,i] for i in 1:size(miller_list_pc_raw, 2)]
    miller_list_pc_raw_unique = hcat(unique(miller_list_pc_raw_points)...)

    # Create a dictionary to map each point to its column index
    miller_map = Dict{Tuple{Int, Int, Int}, Int}()

    # Get the number of columns in the matrix
    num_columns = size(miller_list_pc_raw_unique, 2)

    # Fill the dictionary with points as keys and column indices as values
    for i in 1:num_columns
        point = tuple(miller_list_pc_raw_unique[:, i]...)  # Convert the column to a tuple (x, y, z)
        miller_map[point] = i
    end

    return miller_map
end

function get_unfolded_wf(miller_final_map, miller_pc_ik, wfc_pc_ik, K_init, mesh_scale)
    wfc_sc_ik_shifted = [zeros(ComplexF64, length(miller_final_map)) for _ in 1:size(wfc_pc_ik)[1]]
    K = [Int(round(K_init[1])), Int(round(K_init[2])), Int(round(K_init[3]))]
    # println("K = $K , mesh_scale = $mesh_scale , Grand = $(miller_pc_ik[:, 10]) ")
    for iG in 1:size(miller_pc_ik)[2]#[1:10]

        G = miller_pc_ik[:, iG]
        G_shifted = mesh_scale .* (G) + K
        iG_shifted = get(miller_final_map, (G_shifted[1], G_shifted[2], G_shifted[3]), nothing)

        if iG_shifted !== nothing
            for i in 1:size(wfc_pc_ik)[1]
                wfc_sc_ik_shifted[i][iG_shifted] = wfc_pc_ik[i][iG]
            end
            # wfc_sc_ik_shifted[:][iG_shifted] = wfc_pc_ik[:][iG]
        else
            @warn "Point  $G_shifted  not found in miller_list_final"
        end

    end

    return wfc_sc_ik_shifted
end

function prepare_wave_functions_undisp(path_to_in::String, miller_final_map, ik::Int, mesh_scale::Int)
    miller_pc_ik, wfc_pc_ik =  ElectronPhonon.parse_wf(path_to_in*"scf_0/tmp/scf.save/wfc$(ik)")
    K = ElectronPhonon.determine_q_point(path_to_in*"scf_0", ik; sc_size = mesh_scale)
    wfc_pc_ik1_unf = get_unfolded_wf(miller_final_map,miller_pc_ik, wfc_pc_ik, K, mesh_scale)

    g_list = Dict()
    for (index, evc) in enumerate(wfc_pc_ik1_unf)
        g_list["wfc$index"] = evc
    end

    @info "Data saved in "*path_to_in*"/scf_0/g_list_sc_$ik.jld2"
    save(path_to_in*"/scf_0/g_list_sc_$ik.jld2", g_list)
end

function prepare_wave_functions_undisp(path_to_in::String, miller_final_map, sc_size::Int; k_mesh::Int = 1)
    for ik in 1:(sc_size*k_mesh)^3
        prepare_wave_functions_undisp(path_to_in,miller_final_map,ik,sc_size*k_mesh)
        @info "ik = $ik/$((sc_size*k_mesh)^3) is ready"
    end
end

function prepare_wave_functions_disp(path_to_in::String, miller_final_map, ik::Int, Ndisplace::Int, mesh_scale::Int)
    @threads for ind in 1:Ndisplace
        miller_sc_ik, wfc_sc_ik =  ElectronPhonon.parse_wf(path_to_in*"group_$ind/tmp/scf.save/wfc$(ik)")
        K = ElectronPhonon.determine_q_point(path_to_in*"scf_0", ik; sc_size = mesh_scale, use_sc = true)
        wfc_sc_ik1_unf = get_unfolded_wf(miller_final_map, miller_sc_ik, wfc_sc_ik, K, mesh_scale)

        g_list = Dict()
        for (index, evc) in enumerate(wfc_sc_ik1_unf)
            g_list["wfc$index"] = evc
        end

        @info "Data saved in "*path_to_in*"/group_$ind/g_list_sc_$ik.jld2"
        save(path_to_in*"/group_$ind/g_list_sc_$ik.jld2", g_list)
    end
end

function prepare_wave_functions_disp(path_to_in::String, miller_final_map, Ndisplace::Int, k_mesh::Int)
    for ik in 1:(k_mesh)^3
        prepare_wave_functions_disp(path_to_in, miller_final_map, ik, Ndisplace, k_mesh)
        @info "ik = $ik/$(k_mesh^3) is ready"
    end
end

#End of block


function prepare_wave_functions_undisp(path_to_in::String, ik::Int, sc_size::Int)
    file_path=path_to_in*"/scf_0/"

    if sc_size > 1
        @info "Tranforming wave functions to R space:"
        prepare_wave_functions_to_R(file_path;ik=ik)
        @info "Unfolding wave functions to supercell:"
        prepare_unfold_to_sc(file_path,sc_size,ik)
        @info "Tranforming wave functions to G space:"
        prepare_wave_functions_to_G(path_to_in;ik=ik)
    end

end

function prepare_wave_functions_undisp(path_to_in::String, sc_size::Int; k_mesh::Int = 1)
    for ik in 1:(sc_size*k_mesh)^3
        prepare_wave_functions_undisp(path_to_in,ik,sc_size)
        @info "ik = $ik/$((sc_size*k_mesh)^3) is ready"
    end
end

function prepare_wave_functions_disp(path_to_in::String, ik::Int, Ndisplace::Int, sc_size::Int, k_mesh::Int)

    @threads for ind in 1:Ndisplace
        path_to_data = path_to_in*"group_$ind/"
        miller, evc_list_sc = parse_wf(path_to_data*"tmp/scf.save/wfc$ik")
        N_evc = size(evc_list_sc)[1]
        N_g   = length(evc_list_sc[1])
        N = determine_fft_grid(path_to_data*"tmp/scf.save/data-file-schema.xml"; use_xml = true)
        println("N = $N")

        wave_function_result = Array{ComplexF64, 4}(undef, N_evc, N, N, N)
        evc_list_phase = Array{ComplexF64, 2}(undef, N_evc, N_g)

        Nchunk = sc_size^3
        chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]
        evc_chunks = chunk(evc_list_sc[1:N_evc], Nchunk)

        for (index, evc_list_chunk) in enumerate(evc_chunks)
            wave_function_result[(index-1)*Nchunk+1:index*Nchunk,:,:,:] = wf_from_G_list(miller, evc_list_chunk, N)
            GC.gc()

            wf_phase!(path_to_in, wave_function_result[(index-1)*Nchunk+1:index*Nchunk,:,:,:], sc_size, ik)

            evc_list_phase[(index-1)*Nchunk+1:index*Nchunk,:] = wf_to_G_list(miller, wave_function_result[(index-1)*Nchunk+1:index*Nchunk,:,:,:], N)
            GC.gc()
        end

        #save resulting evc list
        save(path_to_data*"/evc_list_sc_$ik.jld2", "evc_list_phase", evc_list_phase)
        @info "idisp = $ind/$(Ndisplace) is ready"
    end
end

function prepare_wave_functions_disp(path_to_in::String, Ndisplace::Int, sc_size::Int, k_mesh::Int)
    for ik in 1:(k_mesh)^3
        prepare_wave_functions_disp(path_to_in,ik, Ndisplace, sc_size, k_mesh)
        @info "ik = $ik/$(k_mesh^3) is ready"
    end
end

function prepare_u_matrixes(path_to_in::String, natoms::Int, sc_size::Int, k_mesh::Int; symmetries::Symmetries = Symmetries([], [], []), save_matrixes::Bool=true)
    U_list = []
    V_list = []

    Ndisplace_nosym = 6 * natoms

    N_fft = determine_fft_grid(path_to_in*"group_1/tmp/scf.save/data-file-schema.xml"; use_xml = true)
    ψₚ0_list = []
    ψₚ0_real_list = []
    miller_list = []

    for (ind, _) in enumerate(unique(symmetries.ineq_atoms_list))
        println("Preparing wave functions for group_$ind:")
        miller1, ψₚ0 = parse_wf(path_to_in*"/group_$ind/tmp/scf.save/wfc1")
        ψₚ0_real = [wf_from_G(miller1, evc, N_fft) for evc in ψₚ0]
        push!(ψₚ0_list, ψₚ0)
        push!(ψₚ0_real_list, ψₚ0_real)
        push!(miller_list, miller1)
    end

    nbnds = length(load(path_to_in*"/scf_0/g_list_sc_1.jld2"))
    println("nbnds = $nbnds")

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
            tras  = symmetries.trans_list[ind] #./sc_size
            rot   = symmetries.rot_list[ind]
        end

        Uₚₖᵢⱼ = zeros(ComplexF64, k_mesh^3, (k_mesh*sc_size)^3, nbnds*sc_size^3, nbnds)

        for ip in 1:(k_mesh)^3
            if all(isapprox.(tras,[0.0,0.0,0.0], atol = 1e-15)) &&
            all(isapprox.(rot, [[1.0,0.0,0.0] [0.0,1.0,0.0] [0.0,0.0,1.0]], atol = 1e-15))
                if k_mesh == 1
                     _, ψₚ = parse_wf(path_to_in*"/group_$(symmetries.ineq_atoms_list[ind])/tmp/scf.save/wfc1")
                else
                    ψₚ_list = load(path_to_in*"/group_$(symmetries.ineq_atoms_list[ind])/g_list_sc_$ip.jld2")
                    ψₚ = [ψₚ_list["wfc$iband"] for iband in 1:length(ψₚ_list)]
                end
            else
                ψₚ0_real = ψₚ0_real_list[symmetries.ineq_atoms_list[ind]]
                miller1 = miller_list[symmetries.ineq_atoms_list[ind]]
                map1 = rotate_grid(N_fft, N_fft, N_fft, rot, tras)
                ψₚ_real = [rotate_deriv(N_fft, N_fft, N_fft, map1, wfc) for wfc in ψₚ0_real]
                ψₚ = [wf_to_G(miller1, evc, N_fft) for evc in ψₚ_real]
            end

            for ik in 1:(sc_size*k_mesh)^3
                ψkᵤ_list = load(path_to_in*"/scf_0/g_list_sc_$ik.jld2")
                ψkᵤ = [ψkᵤ_list["wfc$iband"] for iband in 1:length(ψkᵤ_list)]

                Uₚₖᵢⱼ[ip, ik, :, :] = calculate_braket(ψₚ, ψkᵤ)
                @info ("idisp = $(ind), ik = $ik")
            end

            @info ("ik_sc = $ip is ready")
        end

        if isodd(ind)
            push!(U_list, Uₚₖᵢⱼ)
        else
            push!(V_list, Uₚₖᵢⱼ)
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
