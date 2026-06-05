using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads
using OrderedCollections, Dates

config_calc = ElephanyConfig(
    use_symm      = true,
    abs_disp      = 1e-3,
    mpi_ranks     = 8,
    path_to_calc  = "./",
    path_to_qe    = "./",
    create        = true,
    from_scratch  = false,
    run           = false,
    timeout_scf   = Hour(5),
    timeout_nscf  = Hour(8),
    poll_interval = Minute(3),
    prepare       = false,
    electrons_calc = true,
    phonons_calc  = true,
    calc_ep      = false,
)

toml_file = "./parameters_of_simulations_si.toml"
sc_size = [1,1,1]

launch_elephany_calculation(toml_file,config_calc,sc_size)