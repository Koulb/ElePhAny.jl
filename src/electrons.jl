using JLD2, DelimitedFiles, EzXML, StaticArrays

const Vec3{T} = SVector{3,T} where {T}
const Mat3{T} = SMatrix{3,3,T,9} where {T}

function read_qe_xml(filename::AbstractString)
    # Taken from Wannier.jl package for now
    # from qe/Modules/constants.f90
    Bohr_QE = 0.529177210903
    BOHR_RADIUS_ANGS = Bohr_QE  # Angstrom
    HARTREE_SI = 4.3597447222071e-18  # J
    ELECTRONVOLT_SI = 1.602176634e-19  # J
    AUTOEV = HARTREE_SI / ELECTRONVOLT_SI

    doc = readxml(filename)
    output = findfirst("/qes:espresso/output", root(doc))

    # atoms
    atomic_structure = findfirst("atomic_structure", output)
    alat = parse(Float64, atomic_structure["alat"])
    # from bohr to angstrom
    alat *= BOHR_RADIUS_ANGS
    n_atoms = parse(Int, atomic_structure["nat"])

    # structure info, each column is a vector for position or lattice vector
    atom_positions = Vec3{Float64}[]
    atom_labels = Vector{String}(undef, n_atoms)
    lattice = zeros(3, 3)

    for (i, atom) in enumerate(eachelement(findfirst("atomic_positions", atomic_structure)))
        pos = parse.(Float64, split(atom.content))
        push!(atom_positions, pos)
        atom_labels[i] = atom["name"]
    end
    # lattice
    for i in 1:3
        a = findfirst("cell/a$i", atomic_structure)
        lattice[:, i] = parse.(Float64, split(a.content))
    end
    # from cartesian to fractional
    inv_lattice = inv(lattice)
    atom_positions = map(atom_positions) do pos
        Vec3(inv_lattice * pos)
    end
    # from bohr to angstrom
    lattice *= BOHR_RADIUS_ANGS

    # reciprocal lattice
    recip_lattice = zeros(3, 3)
    for i in 1:3
        b = findfirst("basis_set/reciprocal_lattice/b$i", output)
        recip_lattice[:, i] = parse.(Float64, split(b.content))
    end
    # to 1/angstrom
    recip_lattice *= 2π / alat

    band_structure = findfirst("band_structure", output)
    n_kpts = parse(Int, findfirst("nks", band_structure).content)
    lsda = parse(Bool, findfirst("lsda", band_structure).content)
    # noncolin = parse(Bool, findfirst("noncolin", band_structure).content)
    spinorbit = parse(Bool, findfirst("spinorbit", band_structure).content)
    # check spin-polarized case
    if lsda && !spinorbit
        nbnd_up = parse(Int, findfirst("nbnd_up", band_structure).content)
        nbnd_dn = parse(Int, findfirst("nbnd_dw", band_structure).content)
        # they should be the same in QE
        @assert nbnd_up == nbnd_dn
        n_bands = nbnd_up
        eigenvalues_up = Vector{Float64}[]
        eigenvalues_dn = Vector{Float64}[]
    else
        n_bands = parse(Int, findfirst("nbnd", band_structure).content)
        eigenvalues = Vector{Float64}[]
    end
    kpoints = Vec3{Float64}[]

    n_electrons = parse(Float64, findfirst("nelec", band_structure).content)

    fermi_energy = 0.0
    try
        fermi_energy = parse(Float64, findfirst("fermi_energy", band_structure).content)
    catch; end

    # Hartree to eV
    fermi_energy *= AUTOEV

    inv_recip = inv(recip_lattice)
    ks_energies = findall("ks_energies", band_structure)
    for ks_energy in ks_energies
        k_point = findfirst("k_point", ks_energy)
        kpt = parse.(Float64, split(k_point.content))
        # to 1/angstrom
        kpt *= 2π / alat
        # from cartesian to fractional
        kpt = inv_recip * kpt
        push!(kpoints, kpt)

        qe_eigenvalues = findfirst("eigenvalues", ks_energy)
        if lsda && !spinorbit
            e = parse.(Float64, split(qe_eigenvalues.content))
            # Hartree to eV
            e .*= AUTOEV
            push!(eigenvalues_up, e[1:n_bands])
            push!(eigenvalues_dn, e[(n_bands + 1):end])
        else
            e = parse.(Float64, split(qe_eigenvalues.content))
            # Hartree to eV
            e .*= AUTOEV
            push!(eigenvalues, e)
        end
    end

    lattice = Mat3(lattice)
    recip_lattice = Mat3(recip_lattice)

    results = (;
        lattice,
        atom_positions,
        atom_labels,
        recip_lattice,
        kpoints,
        n_electrons,
        fermi_energy,
        alat,
    )
    if lsda && !spinorbit
        return (; results..., eigenvalues_up, eigenvalues_dn)
    end
    return (; results..., eigenvalues)
end

function create_scf_calc(path_to_scf::String, unitcell, scf_parameters)
    # Create the cell
    atoms  = pycall(ase.Atoms;unitcell...)

    # Write the input file using Quantum ESPRESSO format
    ase_io.write(path_to_scf*"scf.in",atoms; scf_parameters...)
end

function create_disp_calc(path_to_in::String, unitcell, scf_parameters, abs_disp, mesh, use_symm; from_scratch = false)
    # Change to the specified directory
    cd(path_to_in)

    # Clean the folder if nescessary
    if (from_scratch && isdir(path_to_in * "displacements"))
        run(`rm -rf displacements`)
    end

    command = `mkdir displacements`
    try
        run(command);
        println(command)
    catch; end
    path_to_in = path_to_in * "displacements/"
    cd(path_to_in)
    println("Creating folders in $path_to_in:")

    command = `mkdir scf_0 epw out elph_elements`
    try
        run(command);
        println(command)
    catch; end

    nscf_parameters       = deepcopy(scf_parameters)
    #Case of hybrids
    if haskey(scf_parameters, :nqx1)
        nscf_parameters[:nqx1] = mesh
        nscf_parameters[:nqx2] = mesh
        nscf_parameters[:nqx3] = mesh
    end

    create_scf_calc(path_to_in*"scf_0/",unitcell, nscf_parameters)

    try
        command = `cp ../run.sh scf_0`
        run(command);
        println(command)
    catch; end

    unitcells_disp = dislpaced_unitecells(path_to_in, unitcell, abs_disp, mesh, use_symm)
    Ndispalce = size(unitcells_disp)[1]

    for i_disp in 1:Ndispalce
        dir_name = "group_"*string(i_disp)*"/"
        command = `mkdir $dir_name`
        command_cp = `cp ../run.sh $dir_name`

        try
            run(command);
            println(command)

            run(command_cp);
            println(command_cp)
        catch; end

        nscf_parameters       = deepcopy(scf_parameters)
        nscf_parameters[:nbnd]= nscf_parameters[:nbnd]*mesh^3#+2*mesh^3 #need to understand how I provide aditional states to keep the projectability satisfied
        nscf_parameters[:kpts]= pytuple((1, 1, 1))

        create_scf_calc(path_to_in*dir_name,unitcells_disp[i_disp], nscf_parameters)
    end

    return Ndispalce
end


function create_disp_calc!(model::ModelQE; from_scratch = false)
    # Clean the folder if nescessary
    if (from_scratch && isdir(model.path_to_calc * "displacements"))
        run(`rm -rf $(model.path_to_calc)/displacements`)
    end
    command = `mkdir $(model.path_to_calc)/displacements`
    try
        run(command);
        println(command)
    catch; end

    if model.use_symm
        check_symmetries!(model)
    else
        model.Ndispalce = 6 * length(pyconvert(Vector{Vector{Float64}}, model.unitcell[:scaled_positions]))
        println("No symmetries used")
        println("Number of displacements: $(model.Ndispalce)")
    end

    Ndispalce = create_disp_calc(model.path_to_calc, model.unitcell, model.scf_parameters, model.abs_disp, model.mesh, model.use_symm)
    if Ndispalce != model.Ndispalce
        @error "Inconsistend amount of displacement between phonopy ($Ndispalce) and symmetries calcuation ($(model.Ndisplace)) "
    end
end


function create_disp_calc(model::ModelKCW; from_scratch = false)
    # Clean the folder if nescessary     # Will it work ??
    if (from_scratch && isdir(model.path_to_calc * "displacements"))
        run(`rm -rf $(model.path_to_calc)/displacements`)
    end
    command = `mkdir displacements`
    try
        run(command);
        println(command)
    catch; end

    path_to_scf = model.path_to_calc * "displacements/scf_0"
    path_to_wfc_out = path_to_scf*"/tmp/scf.save"

    if (!isdir(path_to_scf))
        command = `mkdir -p $path_to_wfc_out`
        run(command);
        println(command)
    end

    command = `mkdir $(model.path_to_calc)displacements/epw $(model.path_to_calc)displacements/elph_elements`
    try
        run(command);
        println(command)
    catch; end

    for ind in range(1, model.mesh^3)
        file =  model.path_to_calc * "unperturbed/TMP/kc_kcw.save/wfc$(model.spin_channel)$(ind).dat"
        command = `cp $file $path_to_wfc_out/wfc$(ind).dat`
        run(command);
    end

    file =  model.path_to_calc * "unperturbed/TMP/kc_kcw.save/data-file-schema.xml"
    command = `cp $file $path_to_wfc_out/`
    run(command);

    file =  model.path_to_calc * "unperturbed/wannier/nscf.pwo"
    command = `cp $file $path_to_scf/scf.out`
    run(command);

    command = `$(model.path_to_qe)/W90/utility/kmesh.pl $(model.mesh) $(model.mesh) $(model.mesh)`
    run(pipeline(command, stdout="$path_to_scf/kpoints.dat", stderr="$path_to_scf/kmesherr.txt"))

    dislpaced_unitecells(model.path_to_calc*"displacements/", model.unitcell, model.abs_disp, model.mesh)

    for i_disp in 1:model.Ndispalce
        dir_name =  model.path_to_calc * "displacements/group_$(i_disp)/tmp/scf.save"
        command = `mkdir -p $dir_name`
        try
            run(command);
            println(command)
        catch; end

        file =  model.path_to_calc * "perturbed$(i_disp)/TMP/kc_kcw.save/wfc$(model.spin_channel)1.dat"
        command = `ln -s $(file) $(dir_name)/wfc1.dat`
        try
            run(command);
            println(command)
        catch; end

        file =  model.path_to_calc * "perturbed$(i_disp)/TMP/kc_kcw.save/data-file-schema.xml"
        command = `ln -s $(file) $(dir_name)/data-file-schema.xml`
        try
            run(command);
            println(command)
        catch; end
    end
end

function run_scf(path_to_in::String, mpi_ranks::Int = 0)
    # Change to the specified directory
    cd(path_to_in)

    # Execute the command
    if mpi_ranks > 0
        command = `mpirun -np $mpi_ranks pw.x -in scf.in`
    else
        command = `pw.x -in scf.in`
    end

    println(command)
    run(pipeline(command, stdout="scf.out", stderr="errs.txt"))
end

function run_scf_cluster(path_to_in::String)
    # Change to the specified directory
    cd(path_to_in)

    command = `sbatch run.sh`

    println(command)
    run(pipeline(command, stdout="run.out", stderr="errs.txt"))
end

function run_nscf_calc(path_to_in::String, unitcell, scf_parameters, mesh, path_to_kmesh, mpi_ranks)
    println("Ceating nscf:")
    cd(path_to_in*"displacements/scf_0/")

    command = `$path_to_kmesh/W90/utility/kmesh.pl $mesh $mesh $mesh`
    println(command)
    run(pipeline(command, stdout="kpoints.dat", stderr="kmesherr.txt"))

    atoms  = pycall(ase.Atoms;unitcell...)
    scf_parameters[:calculation] = "nscf"

    nscf_parameters       = deepcopy(scf_parameters)
    #Case of hybrids
    if haskey(scf_parameters, :nqx1)
        nscf_parameters[:nqx1] = mesh
        nscf_parameters[:nqx2] = mesh
        nscf_parameters[:nqx3] = mesh
    end

    # Write the input file using Quantum ESPRESSO format
    ase_io.write(path_to_in*"displacements/scf_0/nscf.in",atoms; nscf_parameters...)

    # Change the kpoints without ASE (not implemented yet)
    file = open(path_to_in*"displacements/scf_0/kpoints.dat", "r")
    lines_kpoints = readlines(file)
    close(file)

    file = open(path_to_in*"displacements/scf_0/nscf.in", "r")
    lines_nscf = readlines(file)
    close(file)

    index_kpoints = 0

    for (index, line) in enumerate(lines_nscf)
        if occursin("K_POINTS", line)
            index_kpoints = index
            break
        end
    end

    deleteat!(lines_nscf,index_kpoints+1)
    splice!(lines_nscf, index_kpoints, lines_kpoints)

    file = open(path_to_in*"displacements/scf_0/nscf.in", "w")
    writedlm(file, lines_nscf)
    close(file)

    # Try to copy cluster run file
    path_to_copy = path_to_in*"displacements/scf_0/run_nscf.sh"
    try
        command = `cp ./run.sh $path_to_copy`
        run(command);
        println(command)

        # Open the run_nscf.sh file and replace "scf.in" with "nscf.in"
        file = open(path_to_copy, "r")
        lines = readlines(file)
        close(file)

        for (index, line) in enumerate(lines)
            if occursin("scf.in", line)
                lines[index] = replace(line, "scf.in" => "nscf.in")
            end
        end

        file = open(path_to_copy, "w")
        writedlm(file, lines, quotes=false)
        close(file)

    catch; end

    println("Running nscf:")
    if isfile(path_to_copy)
        command = `sbatch run_nscf.sh`

        println(command)
        run(pipeline(command, stdout="run_nscf.out", stderr="nerrs.txt"))
    else
        if mpi_ranks > 0
            command = `mpirun -np $mpi_ranks pw.x -in nscf.in`
        else
            command = `pw.x -in nscf.in`
        end
        println(command)
        run(pipeline(command, stdout="scf.out", stderr="nerrs.txt"))
    end

    return true
end

function run_disp_calc(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0)
    # Change to the specified directory
    #FIXME Only run scf in the DFT case (not Hybrids)
    println("Running scf_0:")
    if(isfile(path_to_in*"scf_0/"*"run.sh"))
        run_scf_cluster(path_to_in*"scf_0/")
    else
        run_scf(path_to_in*"scf_0/", mpi_ranks)
    end
    # Get a number of displacements
    files = readdir(path_to_in; join=true)

    for i_disp in 1:Ndispalce
        println("Running displacement # $i_disp:")
        dir_name = "group_"*string(i_disp)*"/"
        if(isfile(path_to_in*dir_name*"run.sh"))
            run_scf_cluster(path_to_in*dir_name)
        else
            run_scf(path_to_in*dir_name, mpi_ranks)
        end
    end

    return true
end

###TODO Need to check consistenct for the reading of the potetial
function read_potential(path_to_file::String;skiprows=0)
    rw = Float64[]
    N1, N2, N3 = 0, 0 ,0

    start = 1
    open(path_to_file) do file
        lines = readlines(file)

        #println(path_to_file)
        for line in lines[2:end]
            #println(line)
            if  length(split(line)) == 8
                break
            else
                skiprows +=  1
            end
        end

        line = split(lines[1+skiprows])
        N1 = parse(Int, line[1])
        N2 = parse(Int, line[2])
        N3 = parse(Int, line[3])
        Nat = parse(Int, line[7])

        for line in lines[1+skiprows:end]
            split_line = split(line)
            if length(split_line) == 5 && parse(Int,split_line[1]) == Nat
                #println(line)
                #println(start)
                start += 1 + skiprows
                break
            else
                start += 1
            end
        end

        for i in start:length(lines)
            line = split(lines[i])
            for j in line
                push!(rw, parse(Float64, j))
            end
        end
    end

    ff = zeros(Float64, N1, N2, N3)

    for i in 1:N1, j in 1:N2, k in 1:N3
        ff[i, j, k] = rw[i + N1 * (j - 1) + N1 * N2 * (k - 1)]
    end

    return ff, N1, N2, N3
end

function save_potential(path_to_in::String, Ndispalce, mesh, mpi_ranks)
    # Get a number of displacements
    files = readdir(path_to_in; join=true)

    # Create a dictionary to store the pp.x input parameters
    parameters = Dict(
        "INPUTPP" => Dict(
            "prefix" => "'scf'",
            "outdir" => "'./tmp'",
            "filplot" => "'Vks'",
            "plot_num" => 1,
            "spin_component" => 1,
        )
    )

    if mpi_ranks > 0
        command = `mpirun -np $mpi_ranks pp.x -in pp.in`
    else
        command = `pp.x -in pp.in_nosym`
    end

    println("Saving potential: ")

    for i_disp in 1:Ndispalce+1

        if i_disp > Ndispalce
            dir_name = "scf_0/"
        else
            dir_name = "group_"*string(i_disp)*"/"
        end
        cd(path_to_in*dir_name)

        # Write the pp.x input file
        open("pp.in", "w") do f
            for (section, section_data) in parameters
                write(f, "&$section\n")
                for (key, value) in section_data
                    write(f, "  $key = $value\n")
                end
                write(f, "/\n")
            end
        end

        if i_disp >Ndispalce
            println("Undisplaced")
        else
            println("Displacement #$i_disp")
        end

        println(command)
        run(pipeline(command, stdout="pp.out", stderr="errs_pp.txt"))

        if mesh > 1 &&  dir_name == "scf_0/"
            Upot_pc, = read_potential(path_to_in*dir_name*"Vks",skiprows=1)
            Upot_sc = repeat(Upot_pc, outer=(mesh, mesh, mesh))
            save(path_to_in*dir_name*"Vks.jld2", "Upot_sc", Upot_sc)
        end
        #Need to check consistency between python and julia potential (lot or ?)
    end

end

function get_kpoint_list(path_to_in)
    file = open(path_to_in*"/kpoints.dat", "r")
    lines_kpoints = readlines(file)
    close(file)
    klist = [parse.(Float64, split(line)[1:end-1]) for line in lines_kpoints[3:end]]
    return klist
end

function get_kpoint_list_old(path_to_in)
    k_list = []
    atoms = ase_io.read(path_to_in*"scf.out")

    for (index, kpt) in enumerate(atoms.calc.kpts)
        push!(k_list, round.( pyconvert(Vector,kpt.k), digits=6))
    end
    return k_list
end

function fold_kpoint(ik, iq, k_list)
    k_point = k_list[ik]
    q_point = k_list[iq]

    kq_point = k_point + q_point
    fold_indices = findall(abs.(kq_point) .>= 1)

    for index in fold_indices
        kq_point[index] -= sign(kq_point[index])
    end
    ikq = 1

    for (index, k_point) in enumerate(k_list)
        if all(isapprox.(kq_point, k_point, atol=1e-5))
            ikq = index
            break
        end
    end

    return ikq
end

function prepare_eigenvalues(path_to_in::String, natoms::Int; Ndisplace::Int = 6*natoms, ineq_atoms_list::Vector{Int}=[], spin_channel::String="")
    path_to_xml="tmp/scf.save/data-file-schema.xml"
    group = "scf_0/"
    ϵₚ_list_raw  = []
    ϵₚ_list  = []
    ϵₚₘ_list = []

    if spin_channel == "up"
        ϵkᵤ_list = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues_up]
    elseif spin_channel == "dw"
        ϵkᵤ_list = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues_dn]
    else
        ϵkᵤ_list = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues]
    end

    k_list = get_kpoint_list(path_to_in*group)
    for ind in 1:Ndisplace
        group = "group_$ind/"

        if spin_channel == "up"
            ϵₚ = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues_up][1]
        elseif spin_channel == "dw"
            ϵₚ = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues_dn][1]
        else
            ϵₚ = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][1]
        end

        push!(ϵₚ_list_raw,ϵₚ)
    end

    for ind in 1:2:6*natoms
        ϵₚ  = Ndisplace != 6 * natoms ? ϵₚ_list_raw[ineq_atoms_list[ind]] : ϵₚ_list_raw[ind]
        ϵₚₘ = Ndisplace != 6 * natoms ? ϵₚ_list_raw[ineq_atoms_list[ind+1]] : ϵₚ_list_raw[ind+1]
        push!(ϵₚ_list, ϵₚ)
        push!(ϵₚₘ_list, ϵₚₘ)
    end

    # Save ϵ to a hdf5-like files
    save(path_to_in * "scf_0/ek_list.jld2", "ek_list", ϵkᵤ_list)
    save(path_to_in * "scf_0/ep_list.jld2", "ep_list", ϵₚ_list)
    save(path_to_in * "scf_0/epm_list.jld2", "epm_list", ϵₚₘ_list)
    save(path_to_in * "scf_0/k_list.jld2", "k_list", k_list)

    return ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list
end

function create_electrons(path_to_in::String, natoms::Int, mesh::Int)
    U_list, V_list = prepare_u_matrixes(path_to_in, natoms, mesh)
    ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list = prepare_eigenvalues(path_to_in, natoms)

    return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end

function create_electrons(model::AbstractModel)

    spin_channel = ""
    if hasproperty(model, :spin_channel)
        spin_channel = model.spin_channel
    end

    symmetries = Symmetries([],[],[])
    if hasproperty(model, :symmetries)
        symmetries = model.symmetries
    end

    natoms = length(model.unitcell[:symbols])

    U_list, V_list = prepare_u_matrixes(model.path_to_calc*"displacements/", natoms, model.mesh; symmetries=symmetries)
    ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list = prepare_eigenvalues(model.path_to_calc*"displacements/", natoms; Ndisplace=model.Ndispalce, ineq_atoms_list=symmetries.ineq_atoms_list, spin_channel=spin_channel)

    return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end

function load_electrons(model::AbstractModel)
    U_list   = load(model.path_to_calc * "displacements/scf_0/U_list.jld2")["U_list"]
    V_list   = load(model.path_to_calc * "displacements/scf_0/V_list.jld2")["V_list"]
    ϵkᵤ_list = load(model.path_to_calc * "displacements/scf_0/ek_list.jld2")["ek_list"]
    ϵₚ_list  = load(model.path_to_calc * "displacements/scf_0/ep_list.jld2")["ep_list"]
    ϵₚₘ_list = load(model.path_to_calc * "displacements/scf_0/epm_list.jld2")["epm_list"]
    k_list   = load(model.path_to_calc * "displacements/scf_0/k_list.jld2")["k_list"]

    return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end
