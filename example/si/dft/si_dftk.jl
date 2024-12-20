using ElectronPhonon, PythonCall, ProgressMeter, Base.Threads
using DFTK, Unitful, UnitfulAtomic, AtomsIOPython
using JLD2


#flags
create = false
run = false
prepare = false
dftk_prepare = true
calc_ep = true

ha_to_eV = 27.211386245988

# Example usage
path_to_calc = "/scratch/apolyukhin/julia_tests/si_dftk/"
abs_disp = 1e-3  #1e-4 #0.0005

println("Displacement: $abs_disp")
directory_path = "$path_to_calc"#
path_to_qe= "/home/apolyukhin/Soft/sourse/q-e/"
mpi_ranks = 8

#Params
sc_size = 2 #important to change !
k_mesh  = 1
# Ndispalce = 12

# Lattice constant of Silicon
a = 5.43052  # in Angstrom

unitcell = Dict(
    :symbols =>  pylist(["Si", "Si"]),
    :cell => pylist([[-0.5 * a, 0.0, 0.5 * a],
    [0.0, 0.5 * a, 0.5 * a],
    [-0.5 * a, 0.5 * a, 0.0]]),
    :scaled_positions => pylist([(0, 0, 0), (0.75, 0.75, 0.75)]),
    :masses => pylist([28.08550, 28.08550])
)

# Set up the calculation parameters as a Python dictionary
scf_parameters = Dict(
    :format => "espresso-in",
    :kpts => pytuple((k_mesh*sc_size, k_mesh*sc_size, k_mesh*sc_size)),
    :calculation =>"scf",
    :prefix => "scf",
    :outdir => "./tmp/",
    :pseudo_dir => "/home/apolyukhin/Development/frozen_phonons/elph/example/pseudo",
    :ecutwfc => 60,
    :conv_thr =>1.e-13,# 1e-16,# #1.e-20,#5.e-30
    :pseudopotentials => Dict("Si" => "Si.upf"),
    :diagonalization => "ppcg",#"ppcg",#"ppcg",#david
    :mixing_mode => "plain",
    :mixing_beta => 0.7,
    :crystal_coordinates => true,
    :verbosity => "high",
    :tstress => false,
    :ibrav => 2,
    :tprnfor => true,
    :nbnd => 4,
    :electron_maxstep => 1000,
    :nosym=> true,
    :noinv=> true#,
    # :input_dft => "HSE",
    # :nqx1 => 1,
    # :nqx2 => 1,
    # :nqx3 => 1
)

# So for the HSE we need to do scf with nq1=nq2=nq3=2 and provide the kpoints in the scf.in file (check if nsf is an option)
# For the supercell nq1=nq2=nq3=1 to be consitnent ?
use_symm = false

model = create_model(path_to_calc = path_to_calc,
                      abs_disp = abs_disp,
                      path_to_qe = path_to_qe,
                      mpi_ranks = mpi_ranks,
                      sc_size = sc_size,
                      k_mesh  = k_mesh,
                      unitcell = unitcell,
                      scf_parameters = scf_parameters,
                      use_symm = use_symm)


if create
    create_disp_calc!(model; from_scratch = true)
end

if run
    run_calculations(model)
end


if prepare
    prepare_model(model)
    electrons = create_electrons(model)
    # # ### Phonons calculation
    phonons = create_phonons(model)
end

if dftk_prepare
    #DFTK part
    nbnds = model.scf_parameters[:nbnd]
    k_list = load(model.path_to_calc*"displacements/scf_0/k_list.jld2")["k_list"]
    k_grid_dftk = ExplicitKpoints(k_list)
    #unpertubed calc
    system_pc = load_system(model.path_to_calc*"displacements/scf_0/scf.in")
    system_pc = attach_psp(system_pc, Si=model.scf_parameters[:pseudo_dir]*"/Si.upf")

    model_dftk_pc = model_PBE(system_pc; symmetries=false)
    Ecut_raw_pc = model.scf_parameters[:ecutwfc]
    k_dftk_pc = pyconvert(Any,model.scf_parameters[:kpts][1])
    basis_pc  = PlaneWaveBasis(model_dftk_pc; Ecut=(Ecut_raw_pc)u"Ry", kgrid=k_grid_dftk)
    nbandsalg_pc = AdaptiveBands(basis_pc.model; n_bands_converge=nbnds)
    scfres_pc = self_consistent_field(basis_pc; nbandsalg=nbandsalg_pc)

    #Saving eigenvalues
    ϵkᵤ_list = []
    for eigs in scfres_pc.eigenvalues
        push!(ϵkᵤ_list, eigs[1:nbnds]*ha_to_eV)
    end
    #Saving wavefunctions
    ψkᵤ_list = []
    ψkᵤ_list_pc = scfres_pc.ψ[1:k_dftk_pc^3]
    Nxyz = basis_pc.fft_size[1]

    ϵₚ_list_raw = []
    ϵₚ_list  = []
    ϵₚₘ_list = []
    wfcp_list = []
    global miller_sc = []
    #perturbed calculations
    for i in 1:12
        system = load_system(model.path_to_calc*"displacements/group_$i/scf.in")
        system = attach_psp(system, Si=model.scf_parameters[:pseudo_dir]*"/Si.upf")
        model_dftk = model_PBE(system; symmetries=false)
        Ecut_raw = model.scf_parameters[:ecutwfc]
        k_dftk = pyconvert(Any,model.scf_parameters[:kpts][1])
        basis = PlaneWaveBasis(model_dftk; Ecut=(Ecut_raw)u"Ry", kgrid=(1, 1, 1))
        nbandsalg = AdaptiveBands(basis.model; n_bands_converge=nbnds*k_dftk^3)
        scfres = self_consistent_field(basis; nbandsalg)

        if i == 1
            miller_sc_raw = G_vectors(basis, basis.kpoints[1])
            global miller_sc = permutedims(Int32.(reduce(hcat, miller_sc_raw)'))
        end

        #Saving eigenvalues
        append!(ϵₚ_list_raw, scfres.eigenvalues)
        #Saving wavefunctions
        append!(wfcp_list, scfres.ψ)
    end

    natoms = 2

    for ind in 1:2:12
        ϵₚ  =  [ϵₚ_list_raw[ind][1:nbnds*sc_size^3]*ha_to_eV]
        ϵₚₘ =  [ϵₚ_list_raw[ind+1][1:nbnds*sc_size^3]*ha_to_eV]
        push!(ϵₚ_list, ϵₚ)
        push!(ϵₚₘ_list, ϵₚₘ)
    end

    # unfolding wfc to the supercell
    for (ind_k, ψkᵤ_pc) in enumerate(ψkᵤ_list_pc)
        miller_pc_raw = G_vectors(basis_pc, basis_pc.kpoints[ind_k])
        miller_pc = permutedims(Int32.(reduce(hcat, miller_pc_raw)'))
        ψkᵤ_pc_list = [ψkᵤ_pc[:,i] for i in 1:nbnds]
        ψkᵤ_pc_real = [ElectronPhonon.wf_from_G(miller_pc,ψkᵤ_pc, Nxyz) for ψkᵤ_pc in ψkᵤ_pc_list]
        ψkᵤ_sc_real = [ElectronPhonon.wf_pc_to_sc(ψkᵤ_pc_val, sc_size) for ψkᵤ_pc_val in ψkᵤ_pc_real]
        #phase factor multuplication if figure how to provide explicit k-points to match QE
        #TODO understand why it doesn'y work (Projectability condition is not satisfied)
        q_vector = ElectronPhonon.determine_q_point(model.path_to_calc*"displacements/scf_0/", ind_k)
        exp_factor = ElectronPhonon.determine_phase(q_vector, sc_size*Nxyz)

        ψkᵤ_sc_real_check = [ψkᵤ_sc_real_val .* exp_factor for ψkᵤ_sc_real_val in ψkᵤ_sc_real]

        N_evc = size(ψkᵤ_sc_real)[1]
        for iband in 1:N_evc
            ψkᵤ_sc_real[iband] = ψkᵤ_sc_real[iband] .* exp_factor
        end

        ψkᵤ_sc = [ElectronPhonon.wf_to_G(miller_sc,ψkᵤ_sc_val, sc_size*Nxyz) for ψkᵤ_sc_val in ψkᵤ_sc_real]
        append!(ψkᵤ_list, [ψkᵤ_sc])
    end

    U_list = []
    V_list = []

    for ind in 1:12
        Uₚₖᵢⱼ = zeros(ComplexF64, model.k_mesh^3, (model.k_mesh*model.sc_size)^3, nbnds*model.sc_size^3, nbnds)

        # for ip in 1:(model.k_mesh)^3
        ip = 1
        ψₚ = wfcp_list[ind]#[ip]
        ψₚ = [ψₚ[:,i] for i in 1:nbnds*model.sc_size^3]

        for ik in 1:(sc_size*k_mesh)^3
            ψkᵤ = ψkᵤ_list[ik]
            Uₚₖᵢⱼ[ip, ik, :, :] = ElectronPhonon.calculate_braket(ψₚ, ψkᵤ)#
            @info ("idisp = $(ind), ik = $ik")
        end

        @info ("ik_sc = $ip is ready")
        # end

        if isodd(ind)
            push!(U_list, Uₚₖᵢⱼ)
        else
            push!(V_list, Uₚₖᵢⱼ)
        end
        @info ("group_$ind is ready")
    end


    global electrons = Electrons(U_list, V_list, ϵkᵤ_list, ϵₚ_list, ϵₚₘ_list, k_list)
end

if calc_ep
    # Loading option instead of calculation
    # electrons = load_electrons(model)
    phonons = load_phonons(model)

    # # # #### Electron-phonon matrix elements
    ik_list = [1]#[1,2,3] # [i for i in 1:sc_size^3] ##[1,2]##
    iq_list = [2]#[1,2,3] # [i for i in 1:sc_size^3] ##[1,2]##

    progress = Progress(length(ik_list)*length(iq_list), dt=5.0)

    println("Calculating electron-phonon matrix elements for $(length(ik_list)*length(iq_list)) points:")
    for ik in ik_list #@threads
        for iq in iq_list
            electron_phonon_qe(model, ik, iq)
            electron_phonon(model, ik, iq, electrons, phonons;) #save_epw = true
            plot_ep_coupling(model, ik, iq, nbnd_max = 8)
            next!(progress)
        end
    end
end
