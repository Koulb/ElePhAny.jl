module ElectronPhononCUDAExt

using ElectronPhonon
using CUDA 

import ElectronPhonon: _has_cuda_flag

# Mark that CUDA is available
function __init__()
    @info "ElectronPhononCUDAExt: CUDA extension loaded"
    _has_cuda_flag[] = true
end

"""
GPU-accelerated version of calculate_braket.
"""
function ElectronPhonon.calculate_braket_gpu(ψₚ::Vector{Vector{ComplexF64}}, ψkᵤ::Vector{Vector{ComplexF64}})
    nbnds1 = length(ψₚ)
    nbnds2 = length(ψkᵤ)

    Ng = length(ψₚ[1])

    Ψₚ = reshape(reduce(hcat, ψₚ), Ng, nbnds1)
    Ψkᵤ = reshape(reduce(hcat, ψkᵤ), Ng, nbnds2)

    Ψₚ_d  = CuArray(Ψₚ)
    Ψkᵤ_d = CuArray(Ψkᵤ)

    U_d = adjoint(Ψₚ_d) * Ψkᵤ_d  
    return Array(U_d)             
end

end # module
