using BlockArrays, LinearAlgebra

function determine_q_point_old(path_to_in, iq; mesh=1)
    atoms = ase_io.read(path_to_in*"/scf.out")
    kpoints = atoms.calc.kpts
    q_vector = [0,0,0]
    for (index, kpoint) in enumerate(kpoints)
        if index == iq
            q_vector = pyconvert(Vector{Float64},kpoint.k.* mesh)
            break
        end
    end
    
    return round.(q_vector;digits=4)
end

function determine_q_point(path_to_in, iq; mesh=1)
    file = open(path_to_in*"/kpoints.dat", "r")
    lines_qpoints = readlines(file)
    close(file)
    qpoints = [parse.(Float64, split(line)[1:end-1]) for line in lines_qpoints[3:end]]  
    return qpoints[iq].*mesh
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

function dislpaced_unitecells(path_to_save, unitcell, abs_disp, mesh)
    unitcell_phonopy = phonopy_structure_atoms.PhonopyAtoms(;symbols=unitcell[:symbols], 
                                                             cell=pylist(pyconvert(Array,unitcell[:cell])./pyconvert(Float64,ase.units.Bohr)),#Should be in Bohr, hence conversion
                                                             scaled_positions=unitcell[:scaled_positions],
                                                             masses=unitcell[:masses])

    phonon = phonopy.Phonopy(unitcell_phonopy, is_symmetry=false, supercell_matrix=pylist([[mesh, 0, 0], [0, mesh, 0], [0, 0, mesh]]))
    phonon.generate_displacements(distance=abs_disp)
    supercells_data = phonon.supercells_with_displacements
    supercells = []

    for supercell_data in supercells_data
        supercell = Dict(
            :symbols => supercell_data.get_chemical_symbols(),
            :cell => supercell_data.get_cell()*ase.units.Bohr, #Back to angstroms for ase
            :scaled_positions => supercell_data.get_scaled_positions(),
            :masses => supercell_data.get_masses())
        push!(supercells, supercell)    
    end

    phonon.save(path_to_save*"phonopy_params.yaml")
    return supercells
end

function calculate_phonons(path_to_in::String,unitcell,abs_disp, Ndispalce, mesh; iq=0)
    # Get a number of displacements
    files = readdir(path_to_in; join=true)
    number_atoms = length(unitcell[:symbols])*mesh^3
    forces = Array{Float64}(undef, Ndispalce, number_atoms, 3)
    Bohr = pyconvert(Float64,ase.units.Bohr)
    Rydberg = pyconvert(Float64,ase.units.Rydberg)
    PwscfToTHz = pyconvert(Float64,phonopy.units.PwscfToTHz)
    factor = PwscfToTHz*33.35641

    
    for i_disp in 1:Ndispalce
        dir_name = "group_"*string(i_disp)*"/"
        #atom = ase.io.read(path_to_in*dir_name * "/scf.out")
        # forces[i_disp,:,:] = pyconvert(Matrix{Float64},atom.get_forces())*(Bohr/Rydberg)
        
        force = read_forces_xml(path_to_in*dir_name*"/tmp/scf.save/data-file-schema.xml")
        forces[i_disp,:,:] = force
        
        # println("Forces scaled: ", forces[i_disp,:,:])
        # println(size(forces[i_disp,:,:]))
        # println("Forces: ", atom.get_forces())
        # exit(3)
    end

    #This conversions Julia to Python are getting me worried 
    unitcell[:cell] = pylist(pyconvert(Array,unitcell[:cell])./Bohr)#Should be in Bohr, hence conversion
    unitcell_phonopy = phonopy_structure_atoms.PhonopyAtoms(;symbols=unitcell[:symbols], 
                                                             cell=unitcell[:cell], 
                                                             scaled_positions=pylist(unitcell[:scaled_positions]))

    phonon = phonopy.Phonopy(unitcell_phonopy,
                             is_symmetry=false, 
                             supercell_matrix=pylist([[mesh, 0, 0], [0, mesh, 0], [0, 0, mesh]]),
                             calculator="qe",
                             factor=PwscfToTHz*33.35641)#from internal units to Thz and then to cm-1
    
    phonon.generate_displacements(distance=abs_disp)#, is_plusminus="false"

    
    # phonon.set_forces(Py(forces[1:2:end,:,:]).to_numpy())
    phonon.set_forces(Py(forces).to_numpy())
    phonon.produce_force_constants()
    # phonon.symmetrize_force_constants_by_space_group()
    # phonon.symmetrize_force_constants()

    #phonon.run_mesh(mesh = [1, 1, 1], is_gamma_center=true, with_eigenvectors=true)
    #mesh_dict = phonon.get_mesh_dict()
    phonon.save(path_to_in*"phonopy_params.yaml"; settings=Dict(:force_constants => true))

    #Dumb way of using phonopy since api gives diffrent result
    current_directory = pwd()
    cd(path_to_in)
    command = `phonopy -c phonopy_params.yaml --dim="$mesh $mesh $mesh" --eigvecs --factor $factor -p mesh.conf`
    file_name = "mesh.conf"

    content = ""
    if iq == 0
        # content = "MESH = $mesh $mesh $mesh\nGAMMA_CENTER = .TRUE."
        qpoint = determine_q_point(path_to_in*"scf_0/",1)
        content = "QPOINTS = $(qpoint[1]) $(qpoint[2]) $(qpoint[3])"
        for iq in 2:mesh^3
            qpoint = determine_q_point(path_to_in*"scf_0/",iq)
            content = content*" $(qpoint[1]) $(qpoint[2]) $(qpoint[3])"
        end
    else
        qpoint = determine_q_point(path_to_in*"scf_0/",iq)
        content = "QPOINTS = $(qpoint[1]) $(qpoint[2]) $(qpoint[3])"
    end

    content = content*" \nWRITEDM = .TRUE."
    #content = content*" \nFC_SYMMETRY = .TRUE."
    
    file = open(path_to_in*file_name, "w")
    write(file, content)
    close(file)
    
    #run(pipeline(command))
    run(pipeline(command,stdout = devnull), wait = false)

    cd(current_directory)

    #saving the dynamic matrix 
    path_to_dyn = path_to_in*"dyn_mat"
    command = `mkdir $path_to_dyn`
    try
        run(command);
        println(command)
    catch; end
    
    sleep(3)

    phonons = YAML.load_file(path_to_in*"qpoints.yaml")
    phonon_params = phonopy.load(path_to_in*"phonopy_params.yaml")

    scaled_pos = pyconvert(Matrix, phonon_params.primitive.get_scaled_positions())

    nat = size(scaled_pos)[1]
    phase_block = [[3,3] for _ in 1:nat]
    phase_matrix = BlockArray{ComplexF64}(undef_blocks, phase_block...)
    masses = pyconvert(Vector{Float64},phonon.masses)
    #phonon_params.symmetrize_force_constants()

    for iq in 1:mesh^3
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

#Parse phonons eigenvalues and eigenvectors from qe output  
function parse_qe_ph(path_to_dyn)
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

    ωₐᵣᵣ_ₚₕ = transpose([parse(Float64,split(lines[i])[end-1]) for i in 1:3:length(lines)]) 

    eigen_list = []
    for line in lines
        if !occursin("freq", line)
            push!(eigen_list, parse.(Float64,split(line)[2:end-1]))
        end
    end

    Nat =  round(Int64, length(eigen_list)/length(ωₐᵣᵣ_ₚₕ))
    εₐᵣᵣ_ₚₕ = Array{ComplexF64, 3}(undef, (1, 3*Nat, 3*Nat))

    for iband in 1:3*Nat
        temp_iband = 2*iband - 1
        eigens = vcat(eigen_list[temp_iband], eigen_list[temp_iband+1])

        for iat in 1:3*Nat
            temp_iat = 2*iat - 1
            εₐᵣᵣ_ₚₕ[1, iband, iat] = eigens[temp_iat]+1im*eigens[temp_iat+1]
        end
    end

    return [ωₐᵣᵣ_ₚₕ, εₐᵣᵣ_ₚₕ]
end

function prepare_phonons(path_to_in::String, Ndisp::Int)
    phonon_params = phonopy.load(path_to_in*"phonopy_params.yaml")
    displacements = phonon_params.displacements[pyslice(0,Ndisp,2)]
    Nat = Int(size(pyconvert(Vector,phonon_params.masses))[1])
    M_phonon = []

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

    writedlm(path_to_in*"/scf_0/M_phonon.txt", M_phonon)  
    
    return M_phonon
end