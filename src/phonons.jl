using BlockArrays, LinearAlgebra

function determine_q_point(path_to_in, iq; sc_size=[1,1,1], use_sc = false)
    file = open(joinpath(path_to_in,"kpoints.dat"), "r")
    if use_sc
        file = open(joinpath(path_to_in,"kpoints_sc.dat"), "r")
    end
    lines_qpoints = readlines(file)
    close(file)
    qpoints = [parse.(Float64, split(line)[1:end-1]) for line in lines_qpoints[3:end]]
    return qpoints[iq].*sc_size
end

function determine_q_point_cart(path_to_in,ik)
    file = open(path_to_in*"/scf.out","r")
    lines = readlines(file)
    result = 0.0

    count = 1
    for (index, line) in enumerate(lines)
        if occursin("        k(" , line)
            if ik == count
                result_str = split(line)[5:7]#
                result_str[3] = result_str[3][1:end-2]
                result = parse.(Float64, result_str)
                break
            else
                count += 1
            end
        end
    end

    return result
end

# functon for reading forces from xml
function read_forces_xml(path_to_xml::String)
    doc = readxml(path_to_xml)
    output = findfirst("/qes:espresso/output", root(doc))
    forces_raw = findfirst("forces", output)
    forces_list = 2 * parse.(Float64, split(forces_raw.content)) # factor 2 is to go from eV to Hartree ?

    forces_matrix = reshape(forces_list, (3, length(forces_list) ÷ 3))'
    return forces_matrix
end

function dislpaced_unitecells(path_to_save, unitcell, abs_disp, sc_size, use_symm)
    unitcell_phonopy = phonopy_structure_atoms.PhonopyAtoms(;symbols=unitcell[:symbols],
                                                             cell=pylist(pyconvert(Array,unitcell[:cell])./bohr_to_ang),#Should be in Bohr, hence conversion
                                                             scaled_positions=unitcell[:scaled_positions],
                                                             masses=unitcell[:masses])

    phonon = phonopy.Phonopy(unitcell_phonopy, is_symmetry=use_symm, supercell_matrix=pylist([[sc_size[1], 0, 0], [0, sc_size[2], 0], [0, 0, sc_size[3]]]))
    phonon.generate_displacements(distance=abs_disp)
    supercells_data = phonon.supercells_with_displacements
    supercells = []

    for supercell_data in supercells_data
        supercell = Dict(
            :symbols => supercell_data.get_chemical_symbols(),
            :cell => supercell_data.get_cell()*bohr_to_ang, #Back to angstroms for ase
            :scaled_positions => supercell_data.get_scaled_positions(),
            :masses => supercell_data.get_masses())
        push!(supercells, supercell)
    end

    phonon.save(path_to_save*"phonopy_params.yaml")
    return supercells
end

function collect_forces(path_to_in::String, unitcell, sc_size, Ndispalce)
  # Get a number of displacements
  # files = readdir(path_to_in; join=true)
  number_atoms = length(unitcell[:symbols])*sc_size[1]*sc_size[2]*sc_size[3]
  forces = Array{Float64}(undef, Ndispalce, number_atoms, 3)

    for i_disp in 1:Ndispalce
        dir_name = "group_"*string(i_disp)*"/"
        #atom = ase.io.read(path_to_in*dir_name * "/scf.out")
        # forces[i_disp,:,:] = pyconvert(Matrix{Float64},atom.get_forces())*(Bohr/Rydberg)
        force = nothing
        if isfile(path_to_in*dir_name*"data-file-schema-scf.xml")
            force = read_forces_xml(path_to_in*dir_name*"/tmp/scf.save/data-file-schema-scf.xml")
        else
            force = read_forces_xml(path_to_in*dir_name*"/tmp/scf.save/data-file-schema.xml")
        end
        forces[i_disp,:,:] = force
    end
    return forces
end

function save_dyn_matirx(path_to_in::String, sc_size::Vec3{Int})
    #saving the dynamic matrix
    path_to_dyn = path_to_in*"dyn_mat"
    command = `mkdir $path_to_dyn`
    try
        run(command);
        println(command)
    catch; end

    # phonons = YAML.load_file(path_to_in*"qpoints.yaml")
    phonon_params = phonopy.load(path_to_in*"phonopy_params.yaml")
    scaled_pos = pyconvert(Matrix, phonon_params.primitive.get_scaled_positions())

    nat = size(scaled_pos)[1]
    phase_block = [[3,3] for _ in 1:nat]
    phase_matrix = BlockArray{ComplexF64}(undef_blocks, phase_block...)
    masses = pyconvert(Vector{Float64},phonon_params.masses)
    #phonon_params.symmetrize_force_constants()

    for iq in 1:sc_size[1]*sc_size[2]*sc_size[3]
        # dyn_mat = reduce(hcat,phonons["phonon"][iq]["dynamical_matrix"])'# hcat(...)
        # dyn_mat = dyn_mat[:,1:2:end] + 1im*dyn_mat[:,2:2:end]

        qpoint = determine_q_point(path_to_in*"scf_0/",iq)
        dyn_mat =  pyconvert(Matrix{ComplexF64},phonon_params.get_dynamical_matrix_at_q(qpoint))

        phonon_factor = [exp(2im * π * dot(qpoint, pos)) for pos in eachrow(scaled_pos)]
        for ipos in 1:nat
            for jpos in 1:nat
                # Need to check whit non-heteroatomic systems masses factor
                phonon_block = fill(sqrt(masses[ipos]*masses[jpos])*phonon_factor[ipos]*conj(phonon_factor[jpos]),3,3)
                setblock!(phase_matrix, phonon_block, ipos, jpos)
            end
        end

        dyn_mat = Array(phase_matrix).*dyn_mat
        dyn_max_final = zeros(Float64, 3*nat, 2*3*nat)
        for i in 1:3*nat
            dyn_max_final[:,2*i-1] = real(dyn_mat[:,i])
            dyn_max_final[:,2*i] = imag(dyn_mat[:,i])
        end

        writedlm(path_to_dyn*"/dyn_mat$iq", dyn_max_final)
        #writedlm(path_to_dyn*"/dyn_mat$iq", dyn_mat)
    end

    return true
end

function prepare_phonons_data(path_to_in::String, unitcell, abs_disp, sc_size, k_mesh, use_symm, Ndispalce::Int64; save_dynq=true)
    #Get the forces
    forces = collect_forces(path_to_in, unitcell, sc_size, Ndispalce)
    prepare_phonons_data(path_to_in, unitcell, abs_disp, sc_size, k_mesh, use_symm, forces; save_dynq=save_dynq)
end

function prepare_phonons_data(path_to_in::String, unitcell, abs_disp, sc_size, k_mesh, use_symm, forces::Array{Float64}; save_dynq=true)
    #This conversions Julia to Python are getting me worried
    unitcell[:cell] = pylist(pyconvert(Array,unitcell[:cell])./bohr_to_ang)#Should be in Bohr, hence conversion
    unitcell_phonopy = phonopy_structure_atoms.PhonopyAtoms(;symbols=unitcell[:symbols],
                                                             cell=unitcell[:cell],
                                                             scaled_positions=pylist(unitcell[:scaled_positions]))

    phonon = phonopy.Phonopy(unitcell_phonopy,
                             is_symmetry=use_symm,
                             supercell_matrix=pylist([[sc_size[1], 0, 0], [0, sc_size[2], 0], [0, 0, sc_size[3]]]),
                             calculator="qe",
                             factor=pwscf_to_cm1)#from internal units to Thz and then to cm-1

    phonon.generate_displacements(distance=abs_disp)#, is_plusminus="false"

    # phonon.set_forces(Py(forces[1:2:end,:,:]).to_numpy())
    phonon.set_forces(Py(forces).to_numpy())
    phonon.produce_force_constants()
    # phonon.symmetrize_force_constants_by_space_group()#Coud it help?
    # phonon.symmetrize_force_constants()

    phonon.save(path_to_in*"phonopy_params.yaml"; settings=Dict(:force_constants => true))

    #Dumb way of using phonopy since api gives diffrent result
    current_directory = pwd()

    command = `phonopy -c phonopy_params.yaml --dim="$(sc_size[1]*k_mesh[1]) $(sc_size[2]*k_mesh[2]) $(sc_size[3]*k_mesh[3])" --eigvecs --factor $pwscf_to_cm1 -p sc_size.conf`
    file_name = "sc_size.conf"

    content = ""
    qpoint = determine_q_point(path_to_in*"scf_0/",1)
    content = "QPOINTS = $(qpoint[1]) $(qpoint[2]) $(qpoint[3])"
    for iq in 2:sc_size[1]*k_mesh[1] * sc_size[2]*k_mesh[2] * sc_size[3]*k_mesh[3]
        qpoint = determine_q_point(path_to_in*"scf_0/",iq)
        content = content*" $(qpoint[1]) $(qpoint[2]) $(qpoint[3])"
    end

    content = content*" \nWRITEDM = .TRUE."
    #content = content*" \nFC_SYMMETRY = .TRUE."


    file = open(path_to_in*file_name, "w")
    write(file, content)
    close(file)

    #run(pipeline(command))
    cd(path_to_in)
    run(pipeline(command,stdout = devnull), wait = true)
    cd(current_directory)

    if save_dynq==true
        save_dyn_matirx(path_to_in, sc_size)
    end

    return true
end


#Parse phonons eigenvalues and eigenvectors from qe output
function parse_qe_ph(path_to_dyn,Nat)
    #Read file and save lines between special line
    special_line = "**************************************************************************"

    lines = []
    open(path_to_dyn) do file
        found_special_line = false
        for line in eachline(file)
            if found_special_line
                if occursin(special_line, line)
                    break
                else
                    push!(lines, line)
                end
            elseif occursin(special_line, line)
                found_special_line = true
            end
        end
    end

    ωₐᵣᵣ_ₚₕ = transpose([parse(Float64,split(lines[i])[end-1]) for i in 1:(Nat+1):length(lines)])

    eigen_list = []
    for line in lines
        if !occursin("freq", line)
            push!(eigen_list, parse.(Float64,split(line)[2:end-1]))
        end
    end

    εₐᵣᵣ_ₚₕ = Array{ComplexF64, 3}(undef, (1, 3*Nat, 3*Nat))

    for iband in 1:3*Nat
        temp_iband = Nat*(iband - 1)+1
        eigens = vcat(eigen_list[temp_iband:temp_iband+Nat-1]...)

        for iat in 1:3*Nat
            temp_iat =  2*iat - 1
            εₐᵣᵣ_ₚₕ[1, iband, iat] = eigens[temp_iat]+1im*eigens[temp_iat+1]
        end
    end

    return [ωₐᵣᵣ_ₚₕ, εₐᵣᵣ_ₚₕ]
end

function prepare_phonons(path_to_in::String, sc_size::Vec3{Int})

    local phonon_params

    if isfile(path_to_in * "phonopy_params_nosym.yaml") # to get all the possible displacements
        phonon_params = phonopy.load(path_to_in * "phonopy_params_nosym.yaml")
    else
        phonon_params = phonopy.load(path_to_in*"phonopy_params.yaml")
    end

    Nat = Int(size(pyconvert(Vector,phonon_params.masses))[1])
    Ndisp_nosym = 6 * Nat
    displacements = phonon_params.displacements[pyslice(0,Ndisp_nosym,2)]
    M_phonon  = []
    ωₐᵣᵣ_ₗᵢₛₜ = []
    εₐᵣᵣ_ₗᵢₛₜ = []
    mₐᵣᵣ      = []

    for iat in 1:Nat
        U = []
        temp_iat::Int = 1 + 3 *(iat-1)
        for row_py in displacements[pyslice(temp_iat-1,temp_iat+2)]
            row = pyconvert(Vector,row_py)[2:end]
            push!(U,row/norm(row))
        end
        U_inv =  vcat(U'...)^-1
        push!(M_phonon, U_inv)
    end

    # writedlm(path_to_in*"/scf_0/M_phonon.txt", M_phonon)
    save(path_to_in * "scf_0/M_phonon.jld2", "M_phonon", M_phonon)

    phonons = YAML.load_file(path_to_in*"qpoints.yaml")
    mₐᵣᵣ = pyconvert(Vector, phonon_params.masses)

    save(path_to_in * "scf_0/m_arr.jld2", "m_arr", mₐᵣᵣ)

    for iq in 1:sc_size[1]*sc_size[2]*sc_size[3]
        εₐᵣᵣ = Array{ComplexF64, 3}(undef, (1, 3*Nat, 3*Nat))
        ωₐᵣᵣ = Array{Float64, 2}(undef, (1, 3*Nat))

        qpoint = determine_q_point(path_to_in*"scf_0/",iq)

        scaled_pos = pyconvert(Matrix, phonon_params.primitive.get_scaled_positions())
        phonon_factor = [exp(2im * π * dot(qpoint, pos)) for pos in eachrow(scaled_pos)]

        for (iband, phonon) in enumerate(phonons["phonon"][iq]["band"])
            for iat in 1:Nat
                for icart in 1:3
                    temp_iat::Int = icart + 3 *(iat-1)
                    eig_temp = phonon["eigenvector"][iat][icart][1] + 1im*phonon["eigenvector"][iat][icart][2]
                    εₐᵣᵣ[1, iband, temp_iat] = phonon_factor[iat] * eig_temp
                end
            end
            ωₐᵣᵣ[1, iband] = phonon["frequency"]
        end

        push!(ωₐᵣᵣ_ₗᵢₛₜ, ωₐᵣᵣ)
        push!(εₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ)
    end

    save(path_to_in * "scf_0/omega_arr_list.jld2", "omega_arr_list", ωₐᵣᵣ_ₗᵢₛₜ)
    save(path_to_in * "scf_0/eps_arr_list.jld2", "eps_arr_list", εₐᵣᵣ_ₗᵢₛₜ)

    return M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ
end

function create_phonons(path_to_in::String, sc_size::Vec3{Int})
    M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ = prepare_phonons(path_to_in, sc_size)

    return Phonons(M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ)
end

function create_phonons(model::AbstractModel)
    sc_size = model.k_mesh .* model.sc_size
    M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ = prepare_phonons(model.path_to_calc*"displacements/", sc_size)

    return Phonons(M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ)
end

function load_phonons(model::AbstractModel)
    M_phonon   = load(model.path_to_calc * "displacements/scf_0/M_phonon.jld2")["M_phonon"]
    ωₐᵣᵣ_ₗᵢₛₜ  = load(model.path_to_calc * "displacements/scf_0/omega_arr_list.jld2")["omega_arr_list"]
    εₐᵣᵣ_ₗᵢₛₜ  = load(model.path_to_calc * "displacements/scf_0/eps_arr_list.jld2")["eps_arr_list"]
    mₐᵣᵣ       = load(model.path_to_calc * "displacements/scf_0/m_arr.jld2")["m_arr"]

    return Phonons(M_phonon, ωₐᵣᵣ_ₗᵢₛₜ, εₐᵣᵣ_ₗᵢₛₜ, mₐᵣᵣ)
end
