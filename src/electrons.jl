using JLD2, DelimitedFiles

function create_scf_calc(path_to_scf::String,unitcell, scf_parameters)
    # Create the FCC cell for Silicon
    atoms  = pycall(ase.Atoms;unitcell...)
        
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

    command = `mkdir scf_0 epw out elph_elements`        
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

        nscf_parameters       = deepcopy(scf_parameters)
        nscf_parameters[:nbnd]= nscf_parameters[:nbnd]*mesh^3#+2*mesh^3 #need to understand how I provide aditional states to keep the projectability satisfied
        nscf_parameters[:kpts]= pytuple((1, 1, 1))

        create_scf_calc(path_to_in*dir_name,unitcells_disp[i_disp], nscf_parameters)
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

function run_nscf_calc(path_to_in::String, unitcell, scf_parameters, mesh, path_to_kmesh, mpi_ranks)
    println("Ceating nscf:")
    cd(path_to_in*"displacements/scf_0/")

    command = `$path_to_kmesh/W90/utility/kmesh.pl $mesh $mesh $mesh`
    println(command)
    run(pipeline(command, stdout="kpoints.dat", stderr="kmesherr.txt"))
    
    atoms  = pycall(ase.Atoms;unitcell...)
    scf_parameters[:calculation] = "nscf"
 
    # Write the input file using Quantum ESPRESSO format
    ase_io.write(path_to_in*"displacements/scf_0/nscf.in",atoms; scf_parameters...)

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

    println("Running nscf:")

    if mpi_ranks > 0
        command = `mpirun -np $mpi_ranks pw.x -in nscf.in`
    else
        command = `pw.x -in nscf.in`
    end

    println(command)
    run(pipeline(command, stdout="scf.out", stderr="nerrs.txt"))

    return true
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
        if all(isapprox.(kq_point, k_point))
            ikq = index 
            break
        end
    end

    return ikq
end

function prepare_eigenvalues(path_to_in::String, Ndisplace::Int, mesh::Int) 
    path_to_xml="tmp/scf.save/data-file-schema.xml"
    group = "scf_0/"
    ϵₚ_list  = [] 
    ϵₚₘ_list = [] 

    k_list = get_kpoint_list(path_to_in*group)
    ϵkᵤ_list = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues]

    for ind in 1:2:Ndisplace
        group = "group_$ind/"
        group_m = "group_$(ind+1)/"
        ϵₚ = WannierIO.read_qe_xml(path_to_in*group*path_to_xml)[:eigenvalues][1]
        ϵₚₘ = WannierIO.read_qe_xml(path_to_in*group_m*path_to_xml)[:eigenvalues][1]

        push!(ϵₚ_list,ϵₚ)
        push!(ϵₚₘ_list,ϵₚₘ)
    end

    return ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list
end

function create_electrons(path_to_in::String, Ndisplace::Int, mesh::Int)

# prepare_wave_functions_undisp(path_to_in, mesh;)
U_list, V_list = prepare_u_matrixes(path_to_in, Ndisplace, mesh)
ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list = prepare_eigenvalues(path_to_in, Ndisplace, mesh)


return Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)

end