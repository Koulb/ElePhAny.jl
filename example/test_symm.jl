using ElectronPhonon, Base.Threads, ProgressMeter, JLD2

trans_arry  = [
    # [0.0, -1.1102230246251565e-16, 0.0],
    # [1.1102230246251565e-16, -1.1102230246251565e-16, 0.0],
    # [0.0, 0.0, 0.0],
    # [0.0, -1.1102230246251565e-16, 1.1102230246251565e-16],
    # [0.0, 0.0, 0.0],
    [0.75, 0.7499999999999998, 0.7500000000000002],
    [0.75, 0.7499999999999999, 0.75],
    [0.75, 0.7499999999999998, 0.7500000000000002],
    [0.7500000000000002, 0.7499999999999998, 0.75],
    [0.75, 0.7499999999999999, 0.75],
    [0.7500000000000002, 0.7499999999999998, 0.75]] ./ 4.0

rot_array   = [
    # [-1.0 -1.0 -1.0; 0.0 0.0 1.0; 0.0 1.0 0.0],
    # [0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0],
    # [0.0 1.0 0.0; -1.0 -1.0 -1.0; 0.0 0.0 1.0],
    # [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0],
    # [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 -1.0 -1.0],
    [1.0 1.0 1.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0],
    [-1.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 -1.0 0.0],
    [0.0 0.0 -1.0; 1.0 1.0 1.0; 0.0 -1.0 0.0],
    [0.0 -1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 -1.0],
    [0.0 -1.0 0.0; 0.0 0.0 -1.0; 1.0 1.0 1.0],
    [0.0 0.0 -1.0; 0.0 -1.0 0.0; -1.0 0.0 0.0]]

path_to_in = "/scratch/apolyukhin/julia_tests/si_4_old/displacements"
N_fft = 180
Ndisplace = 11 - 5

miller1, ψₚ0 = ElectronPhonon.parse_fortan_bin(path_to_in*"/group_1/tmp/scf.save/wfc1.dat")
Nevc = size(ψₚ0)[1]

ψₚ_rotated_list = Array{Any}(undef,(Ndisplace, length(ψₚ0)))
# ψₚ_rotated_list[1,:] = ψₚ0

progress = Progress((Ndisplace)*Nevc, dt=3.0)

println("Start rotating wave functions")

map1_list = []

for ind in 1:Ndisplace
    tras  = trans_arry[ind]
    rot   = rot_array[ind]
    map1 = ElectronPhonon.rotate_grid(N_fft, N_fft, N_fft, rot, tras)
    append!(map1_list, [map1])
end

# println(map1_list[1])

@threads for ind_combined in 1:(Ndisplace * Nevc)
    ind = div(ind_combined - 1, Nevc) + 1
    ind_ev = mod(ind_combined - 1, Nevc) + 1
    # println("ind = $ind, ind_ev = $ind_ev`")

    map1 = map1_list[ind]
    evc = ψₚ0[ind_ev]
    ψₚ0_real = ElectronPhonon.wf_from_G_fft(miller1, evc, N_fft)
    ψₚ_real = ElectronPhonon.rotate_deriv(N_fft, N_fft, N_fft, map1, ψₚ0_real)
    ψₚ_rotated = ElectronPhonon.wave_function_to_G(miller1,[ψₚ_real], N_fft)
    ψₚ_rotated_list[ind, ind_ev] = ψₚ_rotated
    next!(progress)

end

# println("Data saved in wfc_rotated.jld2")
# save("wfc_rotated.jld2","wfc", ψₚ_rotated_list)
