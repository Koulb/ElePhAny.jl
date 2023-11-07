using FortranFiles, LinearAlgebra, Base.Threads,ProgressMeter, JLD2

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

function prepare_wave_functions(path_to_in::String; ik::Int=1)
    file_path = path_to_in*"/tmp/scf.save/wfc$ik.dat"
    miller, evc_list = parse_fortan_bin(file_path)

    #Determine the fft grid
    potential_file = open(path_to_in*"/Vks", "r")
    dummy_line = readline(potential_file)
    fft_line = readline(potential_file)
    N = parse(Int64, split(fft_line)[1])

    println("Transforming wave fucntions in real space:")
    wfc_list = Dict()
    for (index, evc) in enumerate(evc_list)
        println("band # $index")
        wfc = wf_from_G_opt(miller, evc, N)
        wfc_list["wfc$index"] = wfc

    end
    println("Data saved in "*path_to_in*"wfc_list_$ik.jld2")
    save(path_to_in*"/wfc_list_$ik.jld2",wfc_list)

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

    ##change to determin_q_point
    atoms = ase_io.read(path_to_in*"scf.out")
    kpoints = atoms.calc.kpts
    q_vector = [0,0,0]

    for (index, kpoint) in enumerate(kpoints)
        if index == ik
            q_vector = pyconvert(Vector{Float64},kpoint.k .* mesh)
        end
    end

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
    prepare_wave_functions_opt(file_path;ik=ik)
    #need to fold in 1st Bz in case of arbitrary
    #fold(ik,iq)
    prepare_wave_functions_opt(file_path;ik=iq)
    #Need to unfold from pc to sc and mulitply wavefunctions by exp(ikr)
    if mesh > 1
        unfold_to_sc(file_path,mesh,ik)
        unfold_to_sc(file_path,mesh,iq)
    end

    for i in 1:2:Ndisp
        file_path=path_to_in*"/group_$i/"
        if mesh > 1
            prepare_wave_functions_opt(file_path)
        else
            prepare_wave_functions(file_path,ik=ik)
        end    
    end

end


#TEST
#path_to_in = "/home/apolyukhin/Development/julia_tests/qe_inputs/displacements/"
# path_to_in = "/home/apolyukhin/Development/frozen_phonons/elph/example/supercell_disp/group_1/tmp_001/silicon.save"
# N = 72
# ik = 1
# prepare_wave_functions_opt(path_to_in, ik,N)
