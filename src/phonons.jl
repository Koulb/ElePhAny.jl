
function dislpaced_unitecells(path_to_save, unitcell, abs_disp)
    unitcell_phonopy = phonopy_structure_atoms.PhonopyAtoms(;symbols=unitcell[:symbols], 
                                                             cell=unitcell[:cell]./ase.units.Bohr,#Should be in Bohr, hence conversion
                                                             scaled_positions=unitcell[:scaled_positions],
                                                             masses=unitcell[:masses])

    phonon = phonopy.Phonopy(unitcell_phonopy, is_symmetry=false, supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    phonon.generate_displacements(distance=abs_disp)
    supercells_data = phonon.supercells_with_displacements
    supercells = []

    for supercell_data in supercells_data
        supercell = Dict(
            :symbols => supercell_data.get_chemical_symbols(),
            :cell => supercell_data.get_cell().*ase.units.Bohr,#Bach to angstroms for ase
            :scaled_positions => supercell_data.get_scaled_positions(),
            :masses => supercell_data.get_masses())
        push!(supercells, supercell)    
    end

    phonon.save(path_to_save*"phonopy_params.yaml")
    return supercells
end

function calculate_phonons(path_to_in::String,unitcell,abs_disp, Ndispalce, mesh)
    # Get a number of displacements
    files = readdir(path_to_in; join=true)
    number_atoms = length(unitcell[:symbols])
    forces = Array{Float64}(undef, Ndispalce, number_atoms, 3)
    factor = phonopy.units.PwscfToTHz*33.35641
    
    for i_disp in 1:Ndispalce
        dir_name = "group_"*string(i_disp)*"/"
        atom = ase.io.read(path_to_in*dir_name * "/scf.out")
        forces[i_disp,:,:] = atom.get_forces()*(ase.units.Bohr/ase.units.Rydberg)
    end

    println("Calculating phonon properties:")    

    unitcell_phonopy = phonopy_structure_atoms.PhonopyAtoms(;symbols=unitcell[:symbols], 
    cell=unitcell[:cell]./ase.units.Bohr,#Should be in Bohr, hence conversion
    scaled_positions=unitcell[:scaled_positions])

    phonon = phonopy.Phonopy(unitcell_phonopy,
                             is_symmetry=false, 
                             supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             calculator="qe",
                             factor=phonopy.units.PwscfToTHz*33.35641)#from internal units to Thz and then to cm-1
    
    phonon.generate_displacements(distance=abs_disp)
    phonon.set_forces(forces)
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
    content = "MESH = $mesh $mesh $mesh\nGAMMA_CENTER = .TRUE."
    file = open(path_to_in*file_name, "w")
    write(file, content)
    close(file)
    # run(pipeline(command))
    run(pipeline(command,stdout = devnull), wait = false)

    cd(current_directory)

    return true 
end
