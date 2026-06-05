using ProgressMeter, Base.Threads, Dates,OrderedCollections,TOML,FileWatching

#------------------------------------------------------
#------------Utility functions---------------------
#------------------------------------------------------

Base.@kwdef struct ElephanyConfig
    use_symm       :: Bool           = true
    abs_disp       :: Float64        = 1e-3
    mpi_ranks      :: Int64          = 8
    path_to_calc   :: String         = "./"
    path_to_qe     :: String         = "./"
    create         :: Bool           = true
    from_scratch   :: Bool           = false
    run            :: Bool           = true
    timeout_scf    :: Period         = Hour(5)
    timeout_nscf   :: Period         = Hour(8)
    poll_interval  :: Period         = Minute(3)
    prepare        :: Bool           = true
    electrons_calc :: Bool           = true
    phonons_calc   :: Bool           = true
    calc_ep        :: Bool           = true
end

StrOrSym = Union{AbstractString, Symbol}

function format_fortran(value)
    isa(value, AbstractString) && return "'$value'"
    isa(value, Bool)           && return value ? ".true." : ".false."
    isa(value, Number)         && return string(value)
    error("format_fortran: unsupported type $(typeof(value)) for value $value")
end


function ensure_file_exists(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "Required file not found: $path\n" *
        "Make sure the preceding calculation step completed successfully."))
end

function ensure_dir_exists(path::AbstractString)
    isdir(path) || throw(ArgumentError(
        "Required directory not found: $path"))
end

function wait_for_file_creation(
    dir,
    files;
    timeout = Minute(30),
)
    # Validate the watch directory exists
    isdir(dir) || throw(ArgumentError("Watch directory does not exist: $dir"))

    pending  = Set(abspath.(files))
    deadline = now() + timeout

    filter!(f -> !isfile(f), pending)  # skip files that already exist
    isempty(pending) && return true

    function watch_directory(watch_dir)
        isdir(watch_dir) || return
        watch_folder(watch_dir) do events
            file_name, event_info = events
            if event_info.renamed || event_info.changed
                path = abspath(file_name)
                if path in pending && isfile(path)
                    delete!(pending, path)
                end
            end
        end
    end

    watch_directory(dir)

    while !isempty(pending) && now() < deadline
        for file in collect(pending)
            sub_dir = dirname(file)
            isdir(sub_dir) && watch_directory(sub_dir)
            isfile(file)   && delete!(pending, file)   # catch races
        end
        sleep(1)
    end

    if !isempty(pending)
        error("Timed out ($(timeout)) waiting for files to be created:\n  " *
              join(collect(pending), "\n  "))
    end
    return true
end

function wait_for_phrase(
    files         :: Vector{String},
    phrases       :: Vector{String};
    timeout       = Minute(30),
    poll_interval = Second(60),
)
    isempty(files)   && throw(ArgumentError("files vector must not be empty"))
    isempty(phrases) && throw(ArgumentError("phrases vector must not be empty"))

    files    = abspath.(files)
    deadline = now() + timeout
    pending  = Set(files)

    while !isempty(pending) && now() < deadline
        for file in collect(pending)
            isfile(file) || continue
            found = Set{String}()
            open(file, "r") do io
                for line in eachline(io)
                    for phrase in phrases
                        occursin(phrase, line) && push!(found, phrase)
                    end
                end
            end
            length(found) == length(phrases) && delete!(pending, file)
        end
        isempty(pending) || sleep(poll_interval)
    end

    if !isempty(pending)
        error("Timed out ($(timeout)) waiting for phrases $(phrases) in:\n  " *
              join(collect(pending), "\n  "))
    end
    return true
end


function write_namelist(
    io::IO,
    name::StrOrSym,
    params::AbstractDict
)
    println(io, "&" * string(name))

    indent = 2
    pad = " "^indent

    for (k, v) in params
        isnothing(v) && continue
        if isa(v, AbstractVector)
            for (i, vi) in enumerate(v)
                @printf(io,"%s%s(%d) = %s\n",pad,k,i,format_fortran(vi))
            end

        else
            @printf(io,"%s%s = %s\n",pad,k,format_fortran(v))
        end
    end

    println(io, "/")
end

function write_namelist(filename::AbstractString, name::StrOrSym, params::AbstractDict)
    return open(filename, "w") do io
        write_namelist(io, name, params)
    end
end
function write_namelists(io::IO, namelists::AbstractDict)
    for (name, nml) in pairs(namelists)
        write_namelist(io, name, nml)
    end
end

function write_namelists(filename::AbstractString, namelists::AbstractDict)
    return open(filename, "w") do io
        write_namelists(io, namelists)
    end
end

function generate_kpoints_file(
    n1::Int,
    n2::Int,
    n3::Int;
    omit_weight::Bool = false,
    out_file::String = "",
)

    n1 > 0 || throw(ArgumentError("n1 must be > 0, got $n1"))
    n2 > 0 || throw(ArgumentError("n2 must be > 0, got $n2"))
    n3 > 0 || throw(ArgumentError("n3 must be > 0, got $n3"))

    totpts = n1 * n2 * n3

    io = out_file == "" ? stdout : open(out_file, "w")

    try
        if !omit_weight
            println(io, "K_POINTS crystal")
            println(io, totpts)

            for x in 0:n1-1, y in 0:n2-1, z in 0:n3-1
                @printf(
                    io,
                    "%16.12f %16.12f %16.12f %16.8e\n",
                    x / n1,
                    y / n2,
                    z / n3,
                    1.0 / totpts,
                )
            end

        else
            @printf(io, "mp_grid = %6d %6d %6d\n", n1, n2, n3)

            println(io, "begin kpoints")

            for x in 0:n1-1, y in 0:n2-1, z in 0:n3-1
                @printf(
                    io,
                    "%16.12f %16.12f %16.12f\n",
                    x / n1,
                    y / n2,
                    z / n3,
                )
            end

            println(io, "end kpoints")
        end

    finally
        out_file == "" || close(io)
    end
end


function tomat(v::Vector)
    rows = [Float64.(row) for row in v]
    return reduce(vcat, [r' for r in rows])
end

function totuples(v::Vector)
    return [NTuple{3,Float64}(Float64.(pos)) for pos in v]
end


function to_ordereddict(d)
    od = OrderedDict{String, Any}()

    for (k, v) in d
        if v isa AbstractDict
            od[k] = to_ordereddict(v)
        else
            od[k] = v
        end
    end

    return od
end


function flatten_to_symbol_dict(d::AbstractDict)
    flat = Dict{Symbol,Any}()

    for (_, subdict) in d
        subdict isa AbstractDict || continue

        for (k, v) in subdict
            flat[Symbol(k)] = deepcopy(v)
        end
    end

    return flat
end

function take_out_part_of_Dict(od, drop_sections)
    new_od = OrderedDict{String, Any}()
    for (section, params) in od
        section in drop_sections && continue
        new_od[section] = deepcopy(params)
    end
    return new_od
end

function generate_scf_namelists_file(
    io             ::IO,
    path_to_scf    ::String,
    inputs         :: AbstractDict,   # your OrderedDict of namelists
)
    write_namelists(io, inputs)
    ensure_file_exists(path_to_scf)
end

function add_kpoints_scf(io::IO,inputs::AbstractDict)

    k_point_grid = inputs["K_POINTS"]["kpoint_grid_array"][1]
    if haskey(inputs["K_POINTS"],"kpoint_shift_array")
        k_point_shift = inputs["K_POINTS"]["kpoint_shift_array"][1]
    else
        k_point_shift = [0,0,0]
    end
    
    println(io, "K_POINTS {automatic}")
    @printf(io, " %i %i %i %i %i %i\n",
        k_point_grid[1],
        k_point_grid[2],
        k_point_grid[3],
        k_point_shift[1],
        k_point_shift[2],
        k_point_shift[3]
    )
    
end

function add_kpoints_nscf(io::IO,path_to_calc::String,inputs::AbstractDict)
    k_point_grid = inputs["K_POINTS"]["kpoint_grid_array"][1]
    path_to_kpoints = joinpath(path_to_calc,"kpoints.dat")
    generate_kpoints_file(k_point_grid[1],k_point_grid[2],k_point_grid[3];out_file = path_to_kpoints)

    ensure_file_exists(path_to_kpoints)

    lines_kpoints = readlines(path_to_kpoints)

    for line in lines_kpoints
        println(io, line)
    end
end

function insert_before_namelist_close!(io::IO, namelist_name::String, lines::Vector{String})
    isempty(lines) && return

    flush(io)
    seek(io, 0)
    content = read(io, String)

    inputph_start = findfirst("&$(namelist_name)", content)
    slash_match   = findnext(r"(?m)^/$", content, last(inputph_start))
    slash_pos     = first(slash_match)

    new_content = content[1:slash_pos-1] * join(lines) * content[slash_pos:end]

    seek(io, 0)
    truncate(io, 0)
    write(io, new_content)
end

#------------------------------------------------------------
#------Configuration of simulation from TOML-------------
#------------------------------------------------------------
const CALC_TYPES = (:dft, :r2scan, :hse)

function build_scf_parameters(base_parameters::Dict, calc_type::Symbol)
    calc_type ∈ CALC_TYPES || throw(ArgumentError(
        "Invalid calc_type: $calc_type. Must be one of $(CALC_TYPES)"))
    calc_type == :dft   && return base_parameters
    calc_type == :r2scan && return merge(base_parameters, Dict(:input_dft => "r2scan"))
    calc_type == :hse   && return merge(base_parameters, Dict(
        :input_dft => "HSE",
        :nqx1 => 1, :nqx2 => 1, :nqx3 => 1,
        )
    )
end

function build_unitcell_python(inputs::AbstractDict)
    species = inputs["ATOMIC_SPECIES"]["atomic_species_array"]
    positions = inputs["ATOMIC_POSITIONS"]["atomic_positions_array"]
    cell = inputs["CELL_PARAMETERS"]["crystal_lattice_vectors"]

    # Build a lookup: symbol => mass
    mass_dict = Dict(
        spec[1] => spec[2]
        for spec in species
    )

    symbols = [pos[1] for pos in positions]
    masses = [mass_dict[sym] for sym in symbols]
    scaled_positions = [pos[2:end] for pos in positions]

    unitcell = Dict(
        :symbols          => pylist(symbols),
        :cell             => pylist(cell),
        :scaled_positions => pylist(scaled_positions),
        :masses           => pylist(masses),
    )

    pseudopotentials = Dict(
        spec[1] => spec[3]
        for spec in inputs["ATOMIC_SPECIES"]["atomic_species_array"]
    )
    return unitcell, pseudopotentials
end

function parse_qe_toml(toml_file::String)

    if isfile(toml_file)
        println("Using existing '$(toml_file)' last modified on ",unix2datetime(stat(toml_file).mtime))
    else
        println("No TOML file with the name '$(toml_file)', please verify the information or create it.")
    end

    all_inputs = TOML.parsefile(toml_file)
    inputs = to_ordereddict(all_inputs)

    return inputs
end


function transform_to_elephany_format(toml_file::String)
    #create a flat Dict of inputs without specific sections
    inputs = parse_qe_toml(toml_file)

    drop_sections = ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS",
                    "FCP", "RISM", "ADDITIONAL_K_POINTS", "CONSTRAINTS", "OCCUPATIONS", 
                    "ATOMIC_VELOCITIES", "ATOMIC_FORCES","SOLVENTS", "HUBBARD"
    )

    scf_parameters = take_out_part_of_Dict(inputs, drop_sections)

    for key in collect(keys(scf_parameters["SYSTEM"]))
        startswith(key, "celldm") && delete!(scf_parameters["SYSTEM"], key)
    end

    scf_parameters = flatten_to_symbol_dict(scf_parameters)

    #add new values to the Dict
    push!(scf_parameters,:kpts => pytuple((
        inputs["K_POINTS"]["kpoint_grid_array"][1][1],
        inputs["K_POINTS"]["kpoint_grid_array"][1][2],
        inputs["K_POINTS"]["kpoint_grid_array"][1][3]
    )))
    push!(scf_parameters,:format => "espresso-in")
    push!(scf_parameters,:crystal_coordinates => true)
    scf_parameters[:prefix] = "scf"
    scf_parameters[:outdir] = "./tmp/"

    unitcell_py,pseudopotentials = build_unitcell_python(inputs)
    push!(scf_parameters,:pseudopotentials => pseudopotentials)

    return scf_parameters, unitcell_py
end

function get_model(toml_file::String,cfg::ElephanyConfig,sc_size::Vector)

   scf_parameters, unitcell_py = transform_to_elephany_format(toml_file::String)

    model = create_model(
        path_to_calc   = cfg.path_to_calc,
        abs_disp       = cfg.abs_disp,
        path_to_qe     = cfg.path_to_qe,
        mpi_ranks      = cfg.mpi_ranks,
        sc_size        = sc_size,
        k_mesh         = inputs[],
        unitcell       = unitcell_py,
        scf_parameters = scf_parameters,
        use_symm       = cfg.use_symm,
    )
    return model
end

#------------------------------------------------------------
#------Configuration of simulation from TOML-------------
#------------------------------------------------------------

function Base.show(io::IO, cfg::ElephanyConfig)
    println(io, "Elephany Workflow Configuration:")
    println(io, "─────────────────────────────────")
    for field in fieldnames(ElephanyConfig)
        println(io, rpad(string(field), 16), " = ", getfield(cfg, field))
    end
end

function validate_config(cfg::ElephanyConfig)
    if cfg.calc_ep && !(cfg.electrons_calc && cfg.phonons_calc)
        throw(ArgumentError(
            "calc_ep = true requires both electrons_calc = true and " *
            "phonons_calc = true inside ElephanyConfig."))
    end
    if cfg.calc_ep && !cfg.prepare
        @warn "calc_ep = true but prepare = false. " *
              "Assuming electrons/phonons were prepared in a previous run."
    end
    if cfg.run && !cfg.create
        @warn "run = true but create = false. " *
              "Ensure displacement directories already exist."
    end
end

function create_calc_dir_and_files(
    model       :: ElectronPhonon.ModelQE;
    from_scratch :: Bool = false,
)
    println("------------- [create] Generating displacement directories and input files -------------")
    create_disp_calc!(model; from_scratch = from_scratch)

    # Verify at least one displacement directory was created
    base_dir = joinpath(model.path_to_calc, "displacements")
    isdir(base_dir) || error(
        "[create] Expected directory not found after create_disp_calc!: $base_dir")
    println("-------------[create] Done-------------")
end


function run_scf_and_nscf(
    model         :: ElectronPhonon.ModelQE;
    path_to_calc  :: String         = "./",
    timeout_scf   :: Dates.Period   = Hour(5),
    timeout_nscf  :: Dates.Period   = Hour(8),
    poll_interval :: Dates.Period   = Minute(1),
)
    base_dir = joinpath(path_to_calc, "displacements")
    isdir(base_dir) || error(
        "[run] Displacement directory missing: $base_dir\n" *
        "Make sure create = true was run before run = true.")

    # Build the list of output files to watch
    files_to_monitor_scf = vcat(
        [joinpath(base_dir, "scf_0", "scf.out")],
        [joinpath(base_dir, "group_$(i)", "scf.out") for i in 1:model.Ndispalce],
    )

    files_to_monitor_nscf = vcat(
        [joinpath(base_dir, "scf_0", "nscf.out")],
        [joinpath(base_dir, "group_$(i)", "nscf.out") for i in 1:model.Ndispalce],
    )
    println("------------- [run] Monitoring $(model.Ndispalce + 1) SCF/NSCF output files -------------")

    # ── SCF ──────────────────────────────────────────────────────────────────
    println("------------- [run] Launching SCF (pw.x) -------------")
    run_disp_calc(model)
    wait_for_phrase(files_to_monitor_scf, ["JOB DONE"];
                    timeout = timeout_scf, poll_interval = poll_interval)

    # Quick sanity-check: warn if any output file looks unusually small
    for f in files_to_monitor_scf
        if isfile(f) && filesize(f) < 1024
            @warn "[run] SCF output is suspiciously small (< 1 kB): $f"
        end
    end
    println("------------- [run] SCF done -------------")

    # ── NSCF ─────────────────────────────────────────────────────────────────
    println("------------- [run] Launching NSCF (pw.x) -------------")
    run_nscf_calc(model)
    run_disp_nscf_calc(model)
    wait_for_phrase(files_to_monitor_nscf, ["JOB DONE"];
                    timeout = timeout_nscf, poll_interval = poll_interval)

    for f in files_to_monitor_nscf
        if isfile(f) && filesize(f) < 1024
            @warn "[run] NSCF output is suspiciously small (< 1 kB): $f"
        end
    end
    println("------------- [run] NSCF done -------------")
end


function prepare_electrons_and_phonons(
    model          :: ElectronPhonon.ModelQE;
    electrons_calc :: Bool = true,
    phonons_calc   :: Bool = true,
)
    prepare_model(model)

    if electrons_calc
        println("------------- [prepare] Creating electrons -------------")
        electrons = create_electrons(model)
        # Basic check: the returned object should be non-nothing
        isnothing(electrons) && error(
            "[prepare] create_electrons returned nothing — check SCF/NSCF convergence.")
        println("------------- [prepare] Electrons done -------------")
    end

    if phonons_calc
        println("------------- [prepare] Creating phonons -------------")
        phonons = create_phonons(model)
        isnothing(phonons) && error(
            "[prepare] create_phonons returned nothing — check SCF/NSCF convergence.")
        println("------------- [prepare] Phonons done -------------")
    end
end


function calc_electrons_phonons_coupling(model::ElectronPhonon.ModelQE)
    println("------------- [calc_ep] Loading electrons and phonons -------------")
    electrons = load_electrons(model)
    phonons   = load_phonons(model)

    isnothing(electrons) && error("[calc_ep] load_electrons returned nothing.")
    isnothing(phonons)   && error("[calc_ep] load_phonons returned nothing.")

    ik_list = collect(1:prod(model.k_mesh .* model.sc_size))
    iq_list = [1]   # extend this list as needed
    n_total = length(ik_list) * length(iq_list)

    println("------------- [calc_ep] Computing e-ph matrix elements for $n_total k·q points -------------")
    progress = Progress(n_total, dt = 5.0)

    for ik in ik_list
        for iq in iq_list
            electron_phonon(model, ik, iq, electrons, phonons; save_epw = true)
            next!(progress)
        end
    end
    println("------------- [calc_ep] Done -------------")
end


function top_launcher(
    model       :: ElectronPhonon.ModelQE,
    config_calc :: ElephanyConfig,
)
    println(config_calc)
    validate_config(config_calc)
    println("══════ Starting Elephany workflow ══════")

    if config_calc.create
        create_calc_dir_and_files(model; from_scratch = config_calc.from_scratch)
    end

    if config_calc.run
        run_scf_and_nscf(
            model;
            path_to_calc  = config_calc.path_to_calc,
            timeout_scf   = config_calc.timeout_scf,
            timeout_nscf  = config_calc.timeout_nscf,
            poll_interval = config_calc.poll_interval,
        )
    end

    if config_calc.prepare
        prepare_electrons_and_phonons(
            model;
            electrons_calc = config_calc.electrons_calc,
            phonons_calc   = config_calc.phonons_calc,
        )
    end

    if config_calc.calc_ep
        calc_electrons_phonons_coupling(model)
    end

    println("══════ Elephany workflow complete ══════")
end

function launch_elephany_calculation(toml_file::String,cfg::ElephanyConfig,sc_size::Vector)
    model = get_model(toml_file,cfg,sc_size)
    top_launcher(model,cfg)
end

#------------------------------------------------------------
#------PBE input files generator: scf,nscf,ph from TOML-------------
#------------------------------------------------------------

function adding_atomic_non_namelists(io::IO,inputs::AbstractDict)


    if haskey(inputs, "ATOMIC_SPECIES")
        atomic_species   = inputs["ATOMIC_SPECIES"]
        println(io, "ATOMIC_SPECIES")
        for species in atomic_species["atomic_species_array"]
            @printf(io, "  %-2s %8.3f  %s\n",
                species[1],
                species[2],
                species[3]
            )
        end 
    end
    

    if haskey(inputs, "ATOMIC_POSITIONS")
        atomic_positions = inputs["ATOMIC_POSITIONS"]
        println(io, "ATOMIC_POSITIONS {", atomic_positions["atomic_positions_unit"],"}")
        for positions in atomic_positions["atomic_positions_array"]
            @printf(io, "  %-2s %8.10f  %8.10f %8.10f\n",
                positions[1],
                positions[2],
                positions[3],
                positions[4]
            )
        end 
    end
    
    ibrav = get(inputs["SYSTEM"], "ibrav", 0)
    if ibrav == 0
        cell_parameters = inputs["CELL_PARAMETERS"]

        println(io, "CELL_PARAMETERS angstrom")
        for i in 1:3
            @printf(io, "  %16.10f %16.10f %16.10f\n", cell_parameters[i,1], cell_parameters[i,2], cell_parameters[i,3])
        end
        println(io)
    end

    if haskey(inputs,"OCCUPATIONS")
        println(io,"OCCUPATIONS")
        for f_inp in inputs["OCCUPATIONS"]["f_inp1"]
            @printf(io,"%f ",f_inp)
        end
        @printf("\n")
        for f_inp in inputs["OCCUPATIONS"]["f_inp2"]
            @printf(io,"%f ",f_inp)
        end
        @printf("\n")
    end

    if haskey(inputs,"ATOMIC_VELOCITIES")
        velocities = inputs["ATOMIC_VELOCITIES"]
        println(io,"ATOMIC_VELOCITIES {a.u}")

        for velocity in velocities["atomic_velocities_array"]
            @printf(io, "  %-2s %8.3f  %8.3f %8.3f\n",
            velocity[1],
            velocity[2],
            velocity[3],
            velocity[4]
        )
        end
    end
    
    if haskey(inputs,"ATOMIC_FORCES")
        forces = inputs["ATOMIC_FORCES"]
        println(io,"ATOMIC_FORCES")

        for force in forces["atomic_forces_array"]
            @printf(io, "  %-2s %8.3f  %8.3f %8.3f\n",
            force[1],
            force[2],
            force[3],
            force[4]
        )
        end

    end

end

function adding_special_format_namelist_scf_parameters(io::IO,inputs::AbstractDict)
    inputs_system = inputs["SYSTEM"]
    lines = String[]

    indent = 2
    pad = " "^indent

    if haskey(inputs_system,"celldm")
        for celldm in inputs_system["celldm"]
            push!(lines,@sprintf("%scelldm(%i) = %.10f\n",pad,celldm[1],celldm[2]))
        end
    end

    for (key, labels) in [
        ("nr",  ["nr1",  "nr2",  "nr3" ]),
        ("nrs", ["nr1s", "nr2s", "nr3s"]),
        ("nqx", ["nqx1", "nqx2", "nqx3"]),
    ]
        if haskey(inputs_system, key)
            val = inputs_system[key]
            if length(val[1]) != 3
                @warn "Expected 3 values for '$key', got $(length(val)), skipping."
                continue
            end
            for (label, v) in zip(labels, val)
                push!(lines, @sprintf("%s%s = %i\n", pad,label, v[1]))
            end
        end
    end

    isempty(lines) && return

    try
        insert_before_namelist_close!(io, "SYSTEM", lines)
    catch e
        @warn "Could not insert special SYSTEM parameters: $e"
    end
end

function adding_title_line_ph(io::IO,inputs::AbstractDict)
    if haskey(inputs,"TITLE_LINE")
        println(io,inputs["TITLE_LINE"]["title_line"])
    end
end

function adding_non_namelist_ph_parameters(io::IO,inputs::AbstractDict)
    if !haskey(inputs, "INPUTPH")
        @warn "No INPUTPH namelist found in inputs, skipping non-namelist parameters."
        return
    end

    pad = "  "

    if haskey(inputs["INPUTPH"],"ldisp") && haskey(inputs["INPUTPH"],"qplot")
        if inputs["INPUTPH"]["ldisp"] == false && inputs["INPUTPH"]["qplot"] == false
            additional_qpoints = inputs["XQ_LINE"]
            if !isempty(additional_qpoints["xq"])
                @printf(io,"[ %8.3f  %8.3f %8.3f]\n", 
                additional_qpoints["xq"][1],
                additional_qpoints["xq"][2],
                additional_qpoints["xq"][3]
                )
            end
        elseif inputs["INPUTPH"]["qplot"] == true
            additional_qpoints = inputs["QPOINTS_SPECS"]
            @printf(io,"[ %i \n", size(additional_qpoints["qpoints"])[1])
            i = 1
            for qpoint in additional_qpoints["qpoints"]
                if i ==  size(additional_qpoints["qpoints"])[1]
                    @printf(io,"%s%f  %8.3f %8.3f %i ]\n",
                    pad,
                    qpoint[1],
                    qpoint[2],
                    qpoint[3],
                    qpoint[4])
                else
                    @printf(io,"%s%f  %8.3f %8.3f %i \n",
                    pad,
                    qpoint[1],
                    qpoint[2],
                    qpoint[3],
                    qpoint[4])
                    i += 1
                end
                
            end
        end
     end

    if haskey(inputs["INPUTPH"],"nat_todo")
        atoms = inputs["ATOM_LIST"]
        @printf(io,"[")
        for atom in atoms["atoms"]
            @printf(io,"%i ",atom[1])
        end
        @printf(io,"]\n")
    end
end

function adding_special_format_namelist_ph_parameters(io::IO,inputs::AbstractDict)
    inputs_ph = inputs["INPUTPH"]
    lines     = String[]

    indent = 2
    pad = " "^indent

    if get(inputs_ph, "ldisp", false)
        if !haskey(inputs_ph, "q-points")
            @warn "ldisp is true but no 'q-points' found in INPUTPH, skipping."
        elseif length(inputs_ph["q-points"][1]) != 3
            @warn "Expected 3 values for 'q-points', got $(length(inputs_ph["q-points"])), skipping."
        else
            for (label, val) in zip(["nq1", "nq2", "nq3"], inputs_ph["q-points"][1])
                push!(lines, @sprintf("%s%s = %i\n",pad,label, val))
            end
        end
    end

    if haskey(inputs_ph, "k_point")
        if length(inputs_ph["k_point"][1]) != 6
            @warn "Expected 6 values for 'k_point', got $(length(inputs_ph["k_point"])), skipping."
        else
            for (label, val) in zip(["nk1", "nk2", "nk3", "k1", "k2", "k3"], inputs_ph["k_point"][1])
                push!(lines, @sprintf("%s%s = %i\n",pad, label, val))
            end
        end
    end

    isempty(lines) && return

    try
        insert_before_namelist_close!(io, "INPUTPH", lines)
    catch e
        @warn "Could not insert special INPUTPH parameters: $e"
    end
end

function generate_scf_and_nscf(
    base_parameters :: AbstractDict,   # OrderedDict of namelists
    path_to_calc    :: String,
    prefix          :: String,
)
    drop_sections = ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS")
    ensure_dir_exists(path_to_calc)
    # ── SCF ──────────────────────────────────────────────────────────────────

    println("-------------Generating scf.in-------------")
    path_to_scf = joinpath(path_to_calc, "scf.in")
    scf_parameters = take_out_part_of_Dict(base_parameters, drop_sections)
    
    pop!(scf_parameters["SYSTEM"], "celldm", nothing)
    pop!(scf_parameters["SYSTEM"], "nbnd", nothing)   # nbnd not needed for SCF
    scf_parameters["CONTROL"]["prefix"] = prefix
    scf_parameters["CONTROL"]["outdir"] = "./"
    
    
    open(path_to_scf, "w+") do io
        generate_scf_namelists_file(io,path_to_scf,scf_parameters)
        adding_atomic_non_namelists(io,base_parameters)
        adding_special_format_namelist_scf_parameters(io,base_parameters)
        add_kpoints_scf(io,base_parameters)
    end

    println("-------------scf.in done-------------")

    # ── NSCF ─────────────────────────────────────────────────────────────────

    println("-------------Generating nscf.in-------------")
    path_to_nscf = joinpath(path_to_calc, "nscf.in")
    nscf_parameters = take_out_part_of_Dict(base_parameters, drop_sections)

    nscf_parameters["CONTROL"]["calculation"] = "nscf"
    nscf_parameters["CONTROL"]["prefix"]      = prefix
    nscf_parameters["CONTROL"]["outdir"]      = "./"
    pop!(nscf_parameters["SYSTEM"], "celldm", nothing)

    open(path_to_nscf, "w+") do io
        generate_scf_namelists_file(io, path_to_nscf,nscf_parameters)
        adding_atomic_non_namelists(io,base_parameters)
        adding_special_format_namelist_scf_parameters(io,base_parameters)
        add_kpoints_nscf(io,path_to_calc ,base_parameters)
    end
    println("-------------nscf.in done-------------")
end


function generate_ph(path_to_calc::AbstractString, inputs::AbstractDict)
    path_to_ph = joinpath(path_to_calc, "ph.in")
    println("-------------Generating ph.in-------------")

    drop_sections = ("XQ_LINE","QPOINTS_SPECS","ATOM_LIST","TITLE_LINE")
    ph_parameters = take_out_part_of_Dict(inputs, drop_sections)
    pop!(ph_parameters["INPUTPH"], "q-points", nothing)
    pop!(ph_parameters["INPUTPH"], "k_point",  nothing)

    open(path_to_ph, "w+") do io
        adding_title_line_ph(io,inputs)
        write_namelists(io, ph_parameters)
        adding_special_format_namelist_ph_parameters(io,inputs)
        adding_non_namelist_ph_parameters(io,inputs)
    end
    println("-------------ph.in done-------------")
end

function generate_scf_nscf_from_toml(path_to_toml_file::String,path_to_calc::String,prefix::String)

    ensure_file_exists(path_to_toml_file)
    inputs = parse_qe_toml(path_to_toml_file);

    generate_scf_and_nscf(inputs,path_to_calc,prefix)
    
end

function generate_ph_from_toml(path_to_toml_file::String,path_to_calc::String)
    ensure_file_exists(path_to_toml_file)
    inputs = parse_qe_toml(path_to_toml_file)
    generate_ph(path_to_calc,inputs)
end