using PyCall
# Call the Python module
ase = pyimport("ase")
ase_io = pyimport("ase.io")

include("wave_function.jl")
include("phonons.jl")
include("electron_phonon.jl")

function create_scf_calc(path_to_scf::String,unitcell, scf_parameters)
    # Create the FCC cell for Silicon
    atoms = ase.Atoms(;unitcell...)
        
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
        create_scf_calc(path_to_in*dir_name,unitcells_disp[i_disp], scf_parameters)
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

function save_potential(path_to_in::String, Ndispalce)
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

    end

end

# Example usage
directory_path = "/home/apolyukhin/Development/julia_tests/qe_inputs/"
mpi_ranks = 27

# Lattice constant of Silicon
a = 5.43052  # in Angstrom
angstrom_to_bohr = 1.88973 # Need to understand why
mesh = 1

unitcell = Dict(
        :symbols => ["Si","Si"],
        :cell => [[-0.5 * a, 0.0, 0.5 * a],
                  [0.0, 0.5 * a, 0.5 * a],
                  [-0.5 * a, 0.5 * a, 0.0]],
        :scaled_positions => [(0, 0, 0), (0.75, 0.75, 0.75)],
        :masses => [28.08550,28.08550]
)

# Set up the calculation parameters as a Python dictionary
scf_parameters = Dict(
    :format => "espresso-in",
    :kpts => (4, 4, 4),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/home/apolyukhin/Development/frozen_phonons/elph/example/pseudo",
    :ecutwfc => 60,
    :conv_thr => 1.6e-8,
    :pseudopotentials => Dict("Si" => "Si.upf"),
    :diagonalization => "david",
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => true,
    :tprnfor => true
)

#Wave-function index
ik = 1
abs_disp = 0.01 
Ndispalce = 12

## Electrons calculation
Ndispalce = create_disp_calc(directory_path, unitcell, scf_parameters, abs_disp, mesh; from_scratch = true)
run_disp_calc(directory_path*"displacements/", Ndispalce, mpi_ranks)
save_potential(directory_path*"displacements/", Ndispalce)
prepare_wave_functions_all(directory_path*"displacements/", ik, Ndispalce)
## Phonons calculation
calculate_phonons(directory_path*"displacements/",unitcell, abs_disp, Ndispalce)
### Electron-phonon matrix elements
electron_phonon_qe(directory_path*"displacements/")
electron_phonon(directory_path*"displacements/", abs_disp, Ndispalce)
