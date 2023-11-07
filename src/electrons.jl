using JLD2

function create_scf_calc(path_to_scf::String,unitcell, scf_parameters; gamma = false)
    # Create the FCC cell for Silicon
    atoms  = pycall(ase.Atoms;unitcell...)

    #check if supercell
    if gamma
        scf_parameters[:kpts]= pytuple((1, 1, 1))
    end
        
    # Write the input file using Quantum ESPRESSO format
    ase_io.write(path_to_scf*"scf.in",atoms; scf_parameters...)
end

function create_disp_calc(path_to_in::String, unitcell, scf_parameters, abs_disp, mesh; from_scratch = false)
    # Change to the specified directory
    cd(path_to_in)

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

    command = `mkdir scf_0`        
    try
        run(command);
        println(command)
    catch; end
    create_scf_calc(path_to_in*"scf_0/",unitcell, scf_parameters)

    unitcells_disp = dislpaced_unitecells(path_to_in, unitcell, abs_disp, mesh)
    Ndispalce = size(unitcells_disp)[1]

    for i_disp in 1:Ndispalce
        dir_name = "group_"*string(i_disp)*"/"
        command = `mkdir $dir_name`        
        try
            run(command);
            println(command)
        catch; end
        create_scf_calc(path_to_in*dir_name,unitcells_disp[i_disp], scf_parameters, gamma=true)
    end

    return Ndispalce
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

function run_disp_calc(path_to_in::String, Ndispalce::Int, mpi_ranks::Int = 0)
    # Change to the specified directory
    println("Running scf_0:")
    run_scf(path_to_in*"scf_0/", mpi_ranks)

    # Get a number of displacements
    files = readdir(path_to_in; join=true)

    for i_disp in 1:Ndispalce
        println("Running displacement # $i_disp:")
        dir_name = "group_"*string(i_disp)*"/"
        run_scf(path_to_in*dir_name, mpi_ranks)
    end

    return true
end

function read_potential(path_to_file::String)
    rw = Float64[]
    N1, N2, N3 = 0, 0 ,0
    open(path_to_file) do file
        lines = readlines(file)
    
        line = split(lines[2])
        N1 = parse(Int, line[1])
        N2 = parse(Int, line[2])
        N3 = parse(Int, line[3])
        Nat = parse(Int, line[7])
        start = 1

        for line in lines
            split_line = split(line)
            if length(split_line) == 5 && parse(Int,split_line[1]) == Nat
                start += 1
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

function save_potential(path_to_in::String, Ndispalce, mesh)
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

    command = `pp.x -in pp.in`

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
            Upot_pc, = read_potential(path_to_in*dir_name*"Vks")
            Upot_sc = repeat(Upot_pc, outer=(mesh, mesh, mesh))
            save(path_to_in*dir_name*"Vks.jld2", "Upot_sc", Upot_sc)
        end
        #Need to check consistency between python and julia potential (lot or ?)
    end

end