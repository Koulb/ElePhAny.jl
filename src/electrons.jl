using JLD2, DelimitedFiles, Printf, EzXML, JSON3

"""
    read_qe_xml(filename::AbstractString)

Reads and parses a Quantum ESPRESSO XML output file, extracting structural and electronic information.

# Arguments
- `filename::AbstractString`: Path to the Quantum ESPRESSO XML file.

# Returns
A named tuple containing:
- `lattice::Mat3`: The lattice vectors in Angstroms.
- `atom_positions::Vector{Vec3{Float64}}`: Atomic positions in fractional coordinates.
- `atom_labels::Vector{String}`: Atomic species labels.
- `recip_lattice::Mat3`: Reciprocal lattice vectors in 1/Angstrom.
- `kpoints::Vector{Vec3{Float64}}`: List of k-points in fractional coordinates.
- `n_electrons::Float64`: Number of electrons.
- `fermi_energy::Float64`: Fermi energy in eV.
- `alat::Float64`: Lattice parameter in Angstroms.
- `eigenvalues_up::Vector{Vector{Float64}}` and `eigenvalues_dn::Vector{Vector{Float64}}`:
    (If spin-polarized) Eigenvalues for spin-up and spin-down bands, in eV.
- `eigenvalues::Vector{Vector{Float64}}`: (If not spin-polarized) Eigenvalues for each band, in eV.
"""
function read_qe_xml(filename::AbstractString)
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


"""
    create_scf_calc(path_to_scf::String, unitcell, scf_parameters)

Creates a self-consistent field (SCF) calculation input for Quantum ESPRESSO.

# Arguments
- `path_to_scf::String`: The file path where the SCF input file will be written.
- `unitcell`: The unit cell parameters or structure information, passed as keyword arguments to `ase.Atoms`.
- `scf_parameters`: Parameters for the SCF calculation, passed as keyword arguments to `ase_io.write`.
"""
function create_scf_calc(path_to_scf::String, unitcell, scf_parameters; sanitize = true)
    # Create the cell
    atoms  = pycall(ase.Atoms;unitcell...)

    # Write the input file using Quantum ESPRESSO format
    ase_io.write(path_to_scf,atoms; scf_parameters...)

    if sanitize # in case ase/phonopy didn't manage to correctly save file
        content = read(path_to_scf, String)
        rx = r"np\.float64\((.*?)\)"s
        sanitized = replace(content, rx => s"\1")
        open(path_to_scf, "w") do io
            write(io, sanitized)
        end
    end
end

"""
    generate_kpoints(n1::Int, n2::Int, n3::Int; omit_weight::Bool=false, out_file::String="")

Generates a Monkhorst-Pack grid of k-points for a crystal structure.

# Arguments
- `n1::Int`: Number of k-points along the first reciprocal lattice direction. Must be > 0.
- `n2::Int`: Number of k-points along the second reciprocal lattice direction. Must be > 0.
- `n3::Int`: Number of k-points along the third reciprocal lattice direction. Must be > 0.

# Keyword Arguments
- `omit_weight::Bool=false`: If `false`, outputs k-points with weights in Quantum ESPRESSO format. If `true`, outputs only the k-point coordinates in a custom format.
- `out_file::String=""`: If provided, writes the output to the specified file. Otherwise, prints to standard output.

# Output
- If `omit_weight` is `false`, prints or writes the k-points in Quantum ESPRESSO format, including weights.
- If `omit_weight` is `true`, prints or writes the k-points in a custom format without weights (W90 style).
"""
function generate_kpoints(n1::Int, n2::Int, n3::Int; omit_weight::Bool=false, out_file::String="")
    # Validate inputs
    if n1 <= 0
        println("n1 must be >0")
        return
    end
    if n2 <= 0
        println("n2 must be >0")
        return
    end
    if n3 <= 0
        println("n3 must be >0")
        return
    end

    totpts = n1 * n2 * n3

    # Open file if needed, otherwise use stdout
    io = stdout
    if out_file != ""
        io = open(out_file, "w")
    end

    if !omit_weight
        println(io, "K_POINTS crystal")
        println(io, totpts)
        for x in 0:(n1-1), y in 0:(n2-1), z in 0:(n3-1)
            @printf(io, "%16.12f%16.12f%16.12f%16.8e\n", x/n1, y/n2, z/n3, 1.0/totpts)
        end
    else
        @printf(io, "mp_grid = %6d %6d %6d\n", n1, n2, n3)
        println(io, "begin kpoints")
        for x in 0:(n1-1), y in 0:(n2-1), z in 0:(n3-1)
            @printf(io, "%16.12f%16.12f%16.12f\n", x/n1, y/n2, z/n3)
        end
        println(io, "end kpoints")
    end

    # Close file if it was opened
    if out_file != ""
        close(io)
    end
end

"""
    include_kpoins(path_to_nscf::String, paht_to_kpts::String)

Replaces the K_POINTS section in a Quantum ESPRESSO input file with new k-points from a separate file.

# Arguments
- `path_to_nscf::String`: Path to the NSCF input file to be modified.
- `paht_to_kpts::String`: Path to the file containing new k-points.

"""
function include_kpoins(path_to_nscf::String, paht_to_kpts::String)
    # Change the kpoints without ASE (not implemented yet)
    file = open(paht_to_kpts, "r")
    lines_kpoints = readlines(file)
    close(file)

    file = open(path_to_nscf, "r")
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

    file = open(path_to_nscf, "w")
    writedlm(file, lines_nscf)
    close(file)
end

"""
    create_disp_calc(path_to_in::String, path_to_qe::String, unitcell, scf_parameters, abs_disp, sc_size, k_mesh, use_symm; from_scratch = false)

Set up the directory structure and input files for a displacement calculation, used in electron-phonon coupling workflows.

# Arguments
- `path_to_in::String`: Path to the working directory where calculations will be set up.
- `path_to_qe::String`: Path to the Quantum ESPRESSO installation or utilities.
- `unitcell`: The unit cell structure object.
- `scf_parameters`: Dictionary of parameters for the self-consistent field (SCF) calculation.
- `abs_disp`: Magnitude of atomic displacements to apply.
- `sc_size`: Supercell size (integer).
- `k_mesh`: Number of k-points in each direction for the k-point mesh.
- `use_symm`: Boolean or flag to indicate whether to use symmetry in generating displacements.
- `from_scratch` (optional): If `true`, cleans and recreates the displacement directory. Default is `false`.

# Description
This function:
- Creates a directory structure for displacement calculations.
- Generates k-point files for SCF and NSCF calculations.
- Prepares input files for SCF and NSCF calculations for the undistorted and displaced structures.
- Copies necessary scripts (e.g., `run.sh`) into calculation directories.
- Handles special cases for hybrid functionals (e.g., `nqx1`, `nqx2`, `nqx3`).
- Returns the number of displaced structures generated.

# Returns
- `Ndispalce::Int`: The number of displaced structures (and corresponding calculation groups) created.
"""
function create_disp_calc(path_to_in::String, path_to_qe::String, unitcell, scf_parameters, abs_disp, sc_size, k_mesh, use_symm; from_scratch = false)
    Ndispalce = 0
    cd(path_to_in) do
        # Clean the folder if nescessary
        if (from_scratch && isdir(path_to_in * "displacements"))
            run(`rm -rf displacements`)
        end

        command = `mkdir displacements`
        try
            run(command);
            println(command)
        catch; end
    end
    path_to_in = path_to_in * "displacements/"
    cd(path_to_in) do
        println("Creating folders in $path_to_in:")

        command = `mkdir scf_0 epw out`
        try
            run(command);
            println(command)
        catch; end

        nscf_parameters       = deepcopy(scf_parameters)
        hybrids = haskey(scf_parameters, :nqx1)
        hybrids_in_unicell = hybrids && (sc_size[1] != 1 || sc_size[2] != 1 || sc_size[3] != 1)

        #Case of hybrids
        if hybrids_in_unicell
            nscf_parameters[:nqx1] = k_mesh[1]
            nscf_parameters[:nqx2] = k_mesh[2]
            nscf_parameters[:nqx3] = k_mesh[3]
        end
        
        if !hybrids
            pop!(nscf_parameters, :nbnd)
        end
        
        create_scf_calc(path_to_in*"scf_0/scf.in",unitcell, nscf_parameters)

        if !hybrids
            nscf_parameters[:calculation] = "nscf"
        end

        #create nscf calculation as well
        nscf_parameters[:nbnd]= scf_parameters[:nbnd]
        create_scf_calc(path_to_in*"scf_0/nscf.in",unitcell, nscf_parameters)
        include_kpoins(path_to_in*"scf_0/nscf.in", path_to_in*"scf_0/kpoints.dat")

        try
            command = `cp ../run.sh scf_0`
            run(command);
            println(command)
        catch; end

        unitcells_disp = dislpaced_unitecells(path_to_in, unitcell, abs_disp, sc_size, use_symm)
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
            pop!(nscf_parameters, :nbnd)

            nscf_parameters[:kpts]= pytuple((k_mesh[1], k_mesh[2], k_mesh[3]))
            create_scf_calc(path_to_in*dir_name*"scf.in",unitcells_disp[i_disp], nscf_parameters)

            #create nscf calculation as well
            if !hybrids
                nscf_parameters[:calculation] = "nscf"
            end

            nscf_parameters[:nbnd]= scf_parameters[:nbnd]*prod(sc_size)#+2*sc_size^3 #need to understand how I provide aditional states to keep the projectability satisfied
            create_scf_calc(path_to_in*dir_name*"nscf.in",unitcells_disp[i_disp], nscf_parameters)
            if k_mesh[1] != 1 || k_mesh[2] != 1 || k_mesh[3] != 1
                include_kpoins(path_to_in*"group_$i_disp/nscf.in", path_to_in*"scf_0/kpoints_sc.dat")
            end
        end
    end

    return Ndispalce
end


"""
    create_disp_calc!(model::ModelQE; from_scratch = false)

Prepares the displacement calculation environment for a given `ModelQE` instance.
This function manages the setup of the directory structure required for displacement calculations,
optionally cleaning up any existing data if `from_scratch` is set to `true`.

# Arguments
- `model::ModelQE`: The model object containing calculation parameters and paths.
- `from_scratch::Bool=false`: If `true`, removes the existing `displacements` directory before creating a new one.

# Behavior
- Cleans the `displacements` directory if `from_scratch` is `true`.
- Creates a new `displacements` directory if it does not exist.
- If `model.use_symm` is `true`, checks and applies symmetries to reduce the number of displacements.
- Calls `create_disp_calc` to generate the required displacement calculations.
- Verifies consistency between the number of displacements calculated by Phonopy and the symmetry calculation.
"""
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

    cd("$(model.path_to_calc)/displacements") do
        command = `mkdir scf_0 epw out`
        try
            run(command);
            println(command)
        catch; end
    end

    path_to_in = model.path_to_calc * "displacements/"

    generate_kpoints(model.k_mesh[1]*model.sc_size[1], model.k_mesh[2]*model.sc_size[2], model.k_mesh[3]*model.sc_size[3]; out_file=path_to_in*"scf_0/kpoints.dat")
    generate_kpoints(model.k_mesh[1], model.k_mesh[2], model.k_mesh[3]; out_file=path_to_in*"scf_0/kpoints_sc.dat")

    if model.use_symm
        check_symmetries!(model)
    else
        model.Ndispalce = 6 * length(pyconvert(Vector{Vector{Float64}}, model.unitcell[:scaled_positions]))
        println("No symmetries used")
        println("Number of displacements: $(model.Ndispalce)")
    end

    Ndispalce = create_disp_calc(model.path_to_calc, model.path_to_qe, model.unitcell, model.scf_parameters, model.abs_disp, model.sc_size, model.k_mesh, model.use_symm)
    if Ndispalce != model.Ndispalce
        @error "Inconsistend amount of displacement between phonopy ($Ndispalce) and symmetries calcuation ($(model.Ndispalce)) "
    end
end

"""
    create_perturbed_kcw(pristine_data, unitcell)

Updates the atomic positions in the `pristine_data` dictionary with the scaled positions from the `unitcell`.

# Returns
- The updated `pristine_data` dictionary with atomic positions replaced by the corresponding scaled positions from `unitcell`.
"""
function create_perturbed_kcw(pristine_data, unitcell)
    for (index, positions) in enumerate(unitcell[:scaled_positions])
        pristine_data["atoms"]["atomic_positions"]["positions"][index][2:end] = pyconvert(Vector{Float64}, positions)
    end
    return pristine_data
end

"""
    create_disp_calc(model::ModelKCW; from_scratch = false)

Creates displacement calculations for a given `ModelKCW` object by preparing directories and input files for each displaced structure.

# Arguments
- `model::ModelKCW`: The model containing calculation parameters and paths.
- `from_scratch::Bool=false`: (Optional) If true, forces recreation of displacement calculations from scratch.

# Description
This function:
1. Checks for the existence of required files (`koopmans.json` and `koopmans_sc.json`) in the model's calculation path.
2. Creates an `unperturbed` directory and copies necessary files (`run.sh`, `koopmans.json`) into it.
3. Generates displaced unit cells using `dislpaced_unitecells`.
4. For each displacement:
    - Creates a new directory (`perturbedN/`).
    - Copies the `run.sh` script into the new directory.
    - Reads the pristine `koopmans_sc.json` file.
    - Creates a perturbed version of the JSON data using `create_perturbed_kcw`.
    - Writes the perturbed JSON to the corresponding directory.

# Returns
- `Ndispalce::Int`: The number of displaced unit cells generated.
"""
function create_disp_calc(model::ModelKCW; from_scratch = false)
    #Check is model.path_to_calc contains koopmans.json and koopmans_sc.json files
    if !isfile(model.path_to_calc*"koopmans.json")
        @error "No koopmans.json file found in $(model.path_to_calc)"
    end

    if !isfile(model.path_to_calc*"koopmans_sc.json")
        @error "No koopmans_sc.json file found in $(model.path_to_calc)"
    end

    command = `mkdir $(model.path_to_calc)/unperturbed`
    try
        run(command);
        println(command)
    catch; end

    command = `cp $(model.path_to_calc)/run.sh $(model.path_to_calc)/unperturbed`
    try
        run(command);
        println(command)
    catch; end

    command = `cp $(model.path_to_calc)/koopmans.json $(model.path_to_calc)/unperturbed`
    try
        run(command);
        println(command)
    catch; end

    path_to_in = model.path_to_calc * "displacements/"
    unitcells_disp = dislpaced_unitecells(path_to_in, model.unitcell, model.abs_disp, model.sc_size, model.use_symm)
    Ndispalce = size(unitcells_disp)[1]

    for i_disp in 1:Ndispalce
        dir_name = "perturbed"*string(i_disp)*"/"
        command = `mkdir $dir_name`
        command_cp = `cp ./run.sh $dir_name`

        try
            run(command);
            println(command)

            run(command_cp);
            println(command_cp)
        catch; end

        #Read json file
        #pristine_data = JSON3.read("koopmans_sc.json", Dict{String, Any})
        pristine_data = JSON3.read(model.path_to_calc*"koopmans_sc.json", Dict{String, Any})
        perturbed_data =  create_perturbed_kcw(pristine_data, unitcells_disp[i_disp])
        #save json file
        JSON3.write(model.path_to_calc*dir_name*"koopmans_sc.json", perturbed_data)
    end

    return Ndispalce
end

"""
    create_disp_calc!(model::ModelKCW; from_scratch = false)

Prepares the displacement calculation environment for a given `model` of type `ModelKCW`.

# Arguments
- `model::ModelKCW`: The model object containing calculation parameters and paths.
- `from_scratch::Bool=false`: If `true`, removes the existing `displacements` directory before creating a new one.

# Description
- Optionally cleans the `displacements` directory if `from_scratch` is set.
- Creates a new `displacements` directory in the model's calculation path.
- If symmetry usage is enabled (`model.use_symm`), checks and applies symmetries.
- If symmetries are not used, calculates the number of displacements as `6 * number of atoms` and prints relevant information.
- Calls `create_disp_calc(model)` to generate displacements and checks for consistency in the number of displacements between Phonopy and symmetry calculations.
"""
function create_disp_calc!(model::ModelKCW; from_scratch = false)
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

    Ndispalce = create_disp_calc(model)
    if Ndispalce != model.Ndispalce
        @error "Inconsistend amount of displacement between phonopy ($Ndispalce) and symmetries calcuation ($(model.Ndisplace)) "
    end

end

"""
    run_scf(path_to_in::String, mpi_ranks::Int = 0)

Runs a self-consistent field (SCF) calculation using Quantum ESPRESSO's `pw.x` executable.

# Arguments
- `path_to_in::String`: Path to the directory containing the `scf.in` input file.
- `mpi_ranks::Int=0`: Number of MPI ranks to use. If greater than 0, runs the calculation in parallel using `mpirun`; otherwise, runs in serial.
"""
function run_scf(path_to_in::String, mpi_ranks::Int = 0)
    cd(path_to_in) do
        if mpi_ranks > 0
            command = `mpirun -np $mpi_ranks pw.x -in scf.in`
        else
            command = `pw.x -in scf.in`
        end

        println(command)
        run(pipeline(command, stdout="scf.out", stderr="errs.txt"))
    end
end

"""
    run_scf_cluster(path_to_in::String)

Submits a batch job for a self-consistent field (SCF) calculation on a computing cluster.

# Arguments
- `path_to_in::String`: The path to the directory containing the input files and the `run.sh` script.

# Description
Changes the working directory to `path_to_in`, then submits the `run.sh` script as a batch job using `sbatch`. The standard output and error of the job submission command are redirected to `run.out` and `errs.txt`, respectively.
"""
function run_scf_cluster(path_to_in::String; slurm_type::String="sbatch")
    cd(path_to_in) do
        command = `$slurm_type run.sh`

        println(command)
        run(pipeline(command, stdout="run.out", stderr="errs.txt"))
    end
end

"""
    run_nscf_calc(path_to_in::String, mpi_ranks)

Runs a non-self-consistent field (NSCF) calculation for electronic structure simulations.

# Arguments
- `path_to_in::String`: The base path to the input directory containing the SCF calculation results.
- `mpi_ranks`: The number of MPI ranks to use for parallel execution. If greater than 0, runs the calculation in parallel.

# Description
This function performs the following steps:
1. Changes the working directory to the SCF calculation directory (`scf_0`).
2. Copies the `run.sh` script to `run_nscf.sh` and modifies it to use `nscf.in` instead of `scf.in`.
3. Submits the NSCF calculation using `sbatch` if the `run_nscf.sh` script exists.
4. If the script does not exist, runs the calculation directly using `pw.x` (with or without MPI, depending on `mpi_ranks`).
"""
function run_nscf_calc(path_to_in::String, mpi_ranks; slurm_type::String="sbatch")
    println("Ceating nscf:")
    cd(path_to_in*"/scf_0/") do
        path_to_copy = path_to_in*"/scf_0/run_nscf.sh"
        try
            command = `cp ./run.sh $path_to_copy`
            run(command);
            println(command)

            # Open the run_nscf.sh file and replace "scf.in" with "nscf.in" and "scf.out" with "nscf.out"
            file = open(path_to_copy, "r")
            lines = readlines(file)
            close(file)

            for (index, line) in enumerate(lines)
                if occursin("scf.in", line)
                    lines[index] = replace(lines[index], "scf.in" => "nscf.in")
                end
                if occursin("scf.out", line)
                    lines[index] = replace(lines[index], "scf.out" => "nscf.out")
                end
            end

            file = open(path_to_copy, "w")
            writedlm(file, lines, quotes=false)
            close(file)

        catch; end

        println("Running nscf:")
        if isfile(path_to_copy)
            command = `$slurm_type run_nscf.sh`

            println(command)
            run(pipeline(command, stdout="run_nscf.out", stderr="nerrs.txt"))
        else
            if mpi_ranks > 0
                command = `mpirun -np $mpi_ranks pw.x -in nscf.in`
            else
                command = `pw.x -in nscf.in`
            end
            println(command)
            run(pipeline(command, stdout="nscf.out", stderr="nerrs.txt"))
        end
    end
    return true
end

function run_nscf_calc(model::AbstractModel; slurm_type::String="sbatch")
    run_nscf_calc(model.path_to_calc*"displacements/", model.mpi_ranks; slurm_type=slurm_type)
end

"""
    run_disp_calc(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0) -> Bool

Runs self-consistent field (SCF) calculations for a set of atomic displacements in a specified directory.

# Arguments
- `path_to_in::String`: Path to the input directory containing displacement subdirectories.
- `Ndispalce::Int`: Number of displacement groups to process.
- `mpi_ranks::Int=0`: Number of MPI ranks to use for the SCF calculation (default is 0).

# Description
The function first runs an SCF calculation in the `scf_0` subdirectory. If a `run.sh` script is present, it uses `run_scf_cluster`; otherwise, it uses `run_scf`. Then, for each displacement group (from 1 to `Ndispalce`), it runs the corresponding SCF calculation in the `group_i` subdirectory, using the same logic for `run.sh`.
"""
function run_disp_calc(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0, pristine_only::Bool = false; slurm_type::String="sbatch")
    # Change to the specified directory
    #FIXME Only run scf in the DFT case (not Hybrids)

    println("Running scf_0:")
    if(isfile(path_to_in*"scf_0/"*"run.sh"))
        run_scf_cluster(path_to_in*"scf_0/";slurm_type=slurm_type)
    else
        run_scf(path_to_in*"scf_0/", mpi_ranks)
    end

    if !pristine_only
        for i_disp in 1:Ndispalce
            println("Running displacement # $i_disp:")
            dir_name = "group_"*string(i_disp)*"/"
            if(isfile(path_to_in*dir_name*"run.sh"))
                run_scf_cluster(path_to_in*dir_name;slurm_type=slurm_type)
            else
                run_scf(path_to_in*dir_name, mpi_ranks)
            end
        end
    end

    return true
end

function run_disp_calc(model::AbstractModel; slurm_type::String="sbatch")
    run_disp_calc(model.path_to_calc*"displacements/", model.Ndispalce, model.mpi_ranks; slurm_type=slurm_type)
end

"""
    run_disp_nscf_calc(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0)

Runs non-self-consistent field (NSCF) calculations for a series of atomic displacements.

# Arguments
- `path_to_in::String`: Path to the input directory containing displacement groups.
- `Ndispalce::Int`: Number of displacement groups to process.
- `mpi_ranks::Int=0`: Number of MPI ranks to use for parallel execution. If set to 0, runs in serial mode.

# Description
For each displacement group (from 1 to `Ndispalce`), the function:
1. Changes the working directory to the corresponding displacement group directory.
2. Attempts to copy the SCF XML file containing forces to a backup file.
3. Copies and modifies the `run.sh` script to create a `run_nscf.sh` script, replacing occurrences of `"scf.in"` with `"nscf.in"`.
4. If a `run_nscf.sh` script exists, submits it as a batch job using `sbatch`, redirecting output and errors.
5. If not, runs the NSCF calculation directly using `pw.x` (with or without MPI, depending on `mpi_ranks`), redirecting output and errors.
"""
function run_disp_nscf_calc(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0; slurm_type::String="sbatch")
    for i_disp in 1:Ndispalce
        println("Running displacement # $i_disp:")
        dir_name = "group_"*string(i_disp)*"/"
        cd(path_to_in*dir_name) do
            #save scf xml file that has forces
            try
                command = `cp $(path_to_in*dir_name)/tmp/scf.save/data-file-schema.xml $(path_to_in*dir_name)/tmp/scf.save/data-file-schema-scf.xml`
                run(command)
                println(command)
            catch; end

            # Try to copy cluster run file
            path_to_copy = path_to_in*"/group_$i_disp/run_nscf.sh"
            try
                command = `cp ./run.sh $path_to_copy`
                run(command);
                println(command)

                # Open the run_nscf.sh file and replace "scf.in" with "nscf.in" and "scf.out" with "nscf.out"
                file = open(path_to_copy, "r")
                lines = readlines(file)
                close(file)

                for (index, line) in enumerate(lines)
                    if occursin("scf.in", line)
                        lines[index] = replace(lines[index], "scf.in" => "nscf.in")
                    end
                    if occursin("scf.out", line)
                        lines[index] = replace(lines[index], "scf.out" => "nscf.out")
                    end
                end

                file = open(path_to_copy, "w")
                writedlm(file, lines, quotes=false)
                close(file)

            catch; end

            if(isfile(path_to_in*dir_name*"run_nscf.sh"))
                command = `$slurm_type run_nscf.sh`
                println(command)
                run(pipeline(command, stdout="run_nscf.out", stderr="errs.txt"))
            else
                # Execute the command
                if mpi_ranks > 0
                    command = `mpirun -np $mpi_ranks pw.x -in nscf.in`
                else
                    command = `pw.x -in nscf.in`
                end

                println(command)
                run(pipeline(command, stdout="nscf.out", stderr="nerrs.txt"))
            end
        end
    end

    return true
end

function run_disp_nscf_calc(model::AbstractModel; slurm_type::String="sbatch")
    run_disp_nscf_calc(model.path_to_calc*"displacements/", model.Ndispalce, model.mpi_ranks; slurm_type=slurm_type)
end

"""
    run_disp_calc(model::ModelKCW) -> Bool

Runs self-consistent field (SCF) calculations for both unperturbed and perturbed configurations of a given model.

# Arguments
- `model::ModelKCW`: The model object containing calculation parameters, including the path to calculation directories and the number of displacements (`Ndispalce`).

# Description
This function performs the following steps:
1. Runs an SCF calculation in the "unperturbed" subdirectory of the specified calculation path.
2. Iterates over the number of displacements (`Ndispalce`) defined in the model, running SCF calculations in each corresponding "perturbed" subdirectory (e.g., "perturbed1/", "perturbed2/", etc.).
"""
function run_disp_calc(model::ModelKCW)
    # Change to the specified directory
    path_to_in = model.path_to_calc
    println("Running unperturbed:")
    run_scf_cluster(path_to_in*"unperturbed/")

    # Get a number of displacements
    for i_disp in 1:model.Ndispalce
        println("Running perturbed # $i_disp:")
        dir_name = "perturbed"*string(i_disp)*"/"
        run_scf_cluster(path_to_in*dir_name)
    end

    return true
end

"""
    prepare_kcw_data(model::ModelKCW) -> Int

Prepares the necessary directory structure and files for Koopmans calculation for a given `ModelKCW` instance.

# Arguments
- `model::ModelKCW`: The model object containing all relevant paths, parameters, and settings for the calculation.

# Description
This function performs the following steps:
1. Creates required directories for displacements, and electron-phonon elements.
2. Sets up the self-consistent field (SCF) calculation directory and copies necessary wavefunction and data files from the unperturbed calculation.
3. Copies the non-self-consistent field (NSCF) output file to the SCF directory.
4. Generates a k-point mesh file for the supercell.
5. Calls `dislpaced_unitecells` to generate displaced unit cells based on the model parameters.
6. For each displacement, creates the appropriate directory and links the corresponding wavefunction and data files from the perturbed calculations.

# Returns
- `Int`: The number of displacements (`model.Ndispalce`) processed.
"""
function prepare_kcw_data(model::ModelKCW)
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

    command = `mkdir $(model.path_to_calc)displacements/epw`
    try
        run(command);
        println(command)
    catch; end

    for ind in range(1, model.sc_size[1]*model.sc_size[2]*model.sc_size[3])
        file =  model.path_to_calc * "unperturbed/TMP/kc_kcw.save/wfc$(model.spin_channel)$(ind).dat"
        command = `cp $file $path_to_wfc_out/wfc$(ind).dat`
        run(command);
    end

    file =  model.path_to_calc * "unperturbed/TMP/kc_kcw.save/data-file-schema.xml"
    command = `cp $file $path_to_wfc_out/`
    run(command);

    file =  model.path_to_calc * "unperturbed/wannier/nscf.pwo"
    command = `cp $file $path_to_scf/scf.out`
    println(command)
    run(command);

    generate_kpoints(model.sc_size, model.sc_size, model.sc_size; out_file="$path_to_scf/kpoints.dat")
    # command = `$(model.path_to_qe)/W90/utility/kmesh.pl $(model.sc_size) $(model.sc_size) $(model.sc_size)`
    # run(pipeline(command, stdout="$path_to_scf/kpoints.dat", stderr="$path_to_scf/ksc_sizeerr.txt"))

    dislpaced_unitecells(model.path_to_calc*"displacements/", model.unitcell, model.abs_disp, model.sc_size, model.use_symm)

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

    return model.Ndispalce
end


###TODO Need to check consistenct for the reading of the potetial
"""
    read_potential(path_to_file::String; skiprows=0)

Reads a KS potential data file and returns a 3D array of Float64 values along with its dimensions.

# Arguments
- `path_to_file::String`: Path to the file containing the potential data.
- `skiprows`: (Optional) Number of initial rows to skip in the file. Default is 0.

# Returns
- `ff::Array{Float64,3}`: 3D array of potential values with dimensions `(N1, N2, N3)`.
- `N1::Int`: Size of the first dimension.
- `N2::Int`: Size of the second dimension.
- `N3::Int`: Size of the third dimension.
"""
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

"""
    save_potential(path_to_in::String, Ndispalce, sc_size, mpi_ranks)

Saves the electronic potential for a series of displaced structures by running Quantum ESPRESSO's `pp.x` post-processing tool.

# Arguments
- `path_to_in::String`: Path to the directory containing input files for each displacement.
- `Ndispalce`: Number of displacement configurations to process.
- `sc_size`: Size of the supercell. If greater than 1, the potential is repeated accordingly.
- `mpi_ranks`: Number of MPI ranks to use for parallel execution. If greater than 0, runs `pp.x` with MPI.

# Description
For each displacement (and the undisplaced structure), this function:
- Changes to the appropriate directory.
- Writes a `pp.in` input file for `pp.x` with predefined parameters.
- Runs `pp.x` (optionally in parallel) to generate the potential file.
- For the undisplaced structure and if `sc_size > 1`, reads the potential, repeats it to match the supercell size, and saves it in JLD2 format.
"""
function save_potential(path_to_in::String, Ndispalce, sc_size, mpi_ranks)
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
        cd(path_to_in*dir_name) do

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

            if sc_size > 1 &&  dir_name == "scf_0/"
                Upot_pc, = read_potential(path_to_in*dir_name*"Vks",skiprows=1)
                Upot_sc = repeat(Upot_pc, outer=(sc_size[1], sc_size[2], sc_size[3]))
                save(path_to_in*dir_name*"Vks.jld2", "Upot_sc", Upot_sc)
            end
        end
    end

end

"""
    get_kpoint_list(path_to_in)

Reads a list of k-points from a file named `kpoints.dat` located in the directory specified by `path_to_in`.

# Arguments
- `path_to_in::AbstractString`: Path to the directory containing the `kpoints.dat` file.

# Returns
- `klist::Vector{Vector{Float64}}`: A vector of k-points, where each k-point is represented as a vector of `Float64` values.
"""
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

"""
    fold_kpoint(ik, iq, k_list)

Given indices `ik` and `iq` into the list of k-points `k_list`, the function computes the index of the k-point in `k_list`
that corresponds to the sum of `k_list[ik]` and `k_list[iq]`, folded back into the first Brillouin zone.
# Arguments
- `ik::Int`: Index of the first k-point in `k_list`.
- `iq::Int`: Index of the second k-point in `k_list`.
- `k_list::AbstractVector{<:AbstractVector}`: List of k-points (each k-point is a vector).

# Returns
- `ikq::Int`: Index in `k_list` of the folded sum of `k_list[ik]` and `k_list[iq]`.
"""
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

"""
    prepare_eigenvalues(path_to_in::String, natoms::Int; Ndisplace::Int = 6*natoms, ineq_atoms_list::Vector{Int}=[], spin_channel::String="")

Prepares and saves eigenvalues for displaced atomic configurations.

# Arguments
- `path_to_in::String`: Path to the input directory containing calculation data.
- `natoms::Int`: Number of atoms in the system.
- `Ndisplace::Int=6*natoms`: Number of atomic displacements (default is 6 times the number of atoms).
- `ineq_atoms_list::Vector{Int}=[]`: List of indices for inequivalent atoms (used if `Ndisplace` differs from `6*natoms`).
- `spin_channel::String=""`: Spin channel to use; can be `"up"`, `"dw"`, or `""` for non-spin-polarized.

# Returns
A tuple containing:
- `ϵkᵤ_list`: Eigenvalues for the undistorted structure.
- `ϵₚ_list`: List of eigenvalues for positive displacements.
- `ϵₚₘ_list`: List of eigenvalues for negative displacements.
- `k_list`: List of k-points.

# Saves
- `scf_0/ek_list.jld2`: Eigenvalues for the undistorted structure.
- `scf_0/ep_list.jld2`: Eigenvalues for positive displacements.
- `scf_0/epm_list.jld2`: Eigenvalues for negative displacements.
- `scf_0/k_list.jld2`: List of k-points.
"""
function prepare_eigenvalues(path_to_in::String, natoms::Int; Ndisplace::Int = 6*natoms, ineq_atoms_list::Vector{Int}=[], ind_k_list::Vector{Vector{Int}}=[], spin_channel::String="")
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

    if isempty(ind_k_list)
        ind_k_list = [collect(1:length(k_list)) for _ in 1:Ndisplace]
    end

    for ind in 1:Ndisplace
        group = "group_$ind/"

        #Need to save values coresponding to differents k-points not only gamma
        if spin_channel == "up"
            ϵₚ = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues_up]#[1]
        elseif spin_channel == "dw"
            ϵₚ = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues_dn]#[1]
        else
            ϵₚ = read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues]#[1]
        end

        push!(ϵₚ_list_raw,ϵₚ)
    end

    for ind in 1:2:6*natoms
        ϵₚ  = Ndisplace != 6 * natoms ? ϵₚ_list_raw[ineq_atoms_list[ind]][ind_k_list[ind]] : ϵₚ_list_raw[ind]
        ϵₚₘ = Ndisplace != 6 * natoms ? ϵₚ_list_raw[ineq_atoms_list[ind+1]][ind_k_list[ind+1]] : ϵₚ_list_raw[ind+1]
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

"""
    create_electrons(path_to_in::String, natoms::Int, sc_size::Int, k_mesh::Int) -> Electrons

Creates and returns an `Electrons` object by preparing the necessary matrices and eigenvalues.

# Arguments
- `path_to_in::String`: Path to the input file or directory containing required data.
- `natoms::Int`: Number of atoms in the system.
- `sc_size::Int`: Supercell size.
- `k_mesh::Int`: Number of k-points in the mesh.

# Returns
- `Electrons`: An `Electrons` object initialized with the computed matrices and eigenvalues.
"""
function create_electrons(path_to_in::String, natoms::Int, sc_size::Int, k_mesh::Int)
    U_list, V_list = prepare_u_matrixes(path_to_in, natoms, sc_size, k_mesh)
    ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list = prepare_eigenvalues(path_to_in, natoms)

    return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end

"""
    create_electrons(model::AbstractModel)

Creates an `Electrons` object based on the provided `model`. This function extracts relevant properties from the model, such as `spin_channel` and `symmetries`, and uses them to prepare the necessary matrices and eigenvalues for the electron calculations.

# Arguments
- `model::AbstractModel`: The model containing all necessary information about the system, including unit cell, calculation paths, supercell size, k-point mesh, and optional properties like `spin_channel` and `symmetries`.

# Returns
- `Electrons`: An object containing the prepared U and V matrices, eigenvalues, and k-point list for the electron system.
"""
function create_electrons(model::AbstractModel; restart::Bool = false)

    spin_channel = ""
    if hasproperty(model, :spin_channel)
        spin_channel = model.spin_channel
    end

    symmetries = Symmetries([],[],[],[],[])
    if hasproperty(model, :symmetries)
        symmetries = model.symmetries
    end

    natoms = length(model.unitcell[:symbols])

    U_list, V_list = prepare_u_matrixes(model.path_to_calc*"displacements/", natoms, model.sc_size, model.k_mesh; symmetries=symmetries, restart=restart)
    ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list = prepare_eigenvalues(model.path_to_calc*"displacements/", natoms; Ndisplace=model.Ndispalce, ineq_atoms_list=symmetries.ineq_atoms_list, ind_k_list=symmetries.ind_k_list, spin_channel=spin_channel)

    return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end

"""
    load_electrons(model::AbstractModel) -> Electrons

Loads electron-related data from disk for the given `model`. The function reads several arrays from JLD2 files located in the `displacements/scf_0/` subdirectory of `model.path_to_calc`, including:

- `U_list`: Unitary matrices for electron states.
- `V_list`: Additional matrices for electron states.
- `ϵkᵤ_list`: Eigenvalues of undisplaced configuration.
- `ϵₚ_list`: Eigenvalues of displaced (+tau) configuration.
- `ϵₚₘ_list`: Eigenvalues of displaced (-tau) configuration.
- `k_list`: List of k-points in reciprocal space.

Returns an `Electrons` object constructed from the loaded data.
"""
function load_electrons(model::AbstractModel)
    U_list   = load(model.path_to_calc * "displacements/scf_0/U_list.jld2")["U_list"]
    V_list   = load(model.path_to_calc * "displacements/scf_0/V_list.jld2")["V_list"]
    ϵkᵤ_list = load(model.path_to_calc * "displacements/scf_0/ek_list.jld2")["ek_list"]
    ϵₚ_list  = load(model.path_to_calc * "displacements/scf_0/ep_list.jld2")["ep_list"]
    ϵₚₘ_list = load(model.path_to_calc * "displacements/scf_0/epm_list.jld2")["epm_list"]
    k_list   = load(model.path_to_calc * "displacements/scf_0/k_list.jld2")["k_list"]

    return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end
