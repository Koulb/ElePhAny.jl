module ElectronPhononCUDAExt

using ElectronPhonon
using CUDA, LinearAlgebra 

import ElectronPhonon: _has_cuda_flag

# Mark that CUDA is available
function __init__()
    @info "ElectronPhononCUDAExt: CUDA extension loaded"
    _has_cuda_flag[] = true
end

const _bras_d_ref = Ref{Union{Nothing,CuArray{ComplexF64,2}}}(nothing)
const _kets_d_ref = Ref{Union{Nothing,CuArray{ComplexF64,2}}}(nothing)
const _U_d_ref    = Ref{Union{Nothing,CuArray{ComplexF64,2}}}(nothing)

"""
GPU-accelerated version of calculate_braket.
"""
function ElectronPhonon.calculate_braket_gpu(bras::Vector{Vector{ComplexF64}}, kets::Vector{Vector{ComplexF64}})
    nbnds1 = length(bras)
    nbnds2 = length(kets)

    Ng = length(bras[1])

    bras_mat = reshape(reduce(hcat, bras), Ng, nbnds1)
    kets_mat = reshape(reduce(hcat, kets), Ng, nbnds2)

    bras_d  = CuArray(bras_mat)
    kets_d = CuArray(kets_mat)

    U_d = adjoint(bras_d) * kets_d  
    return Array(U_d)             
end

function ElectronPhonon.calculate_braket_gpu!(U::AbstractMatrix{ComplexF64}, bras::Vector{Vector{ComplexF64}}, kets::Vector{Vector{ComplexF64}})
    nbnds1 = length(bras)
    nbnds2 = length(kets)
    Ng = length(bras[1])

    bras_mat = reshape(reduce(hcat, bras), Ng, nbnds1)
    kets_mat = reshape(reduce(hcat, kets), Ng, nbnds2)

    if _bras_d_ref[] === nothing || size(_bras_d_ref[]) != size(bras_mat)
        _bras_d_ref[] = CuArray{ComplexF64}(undef, size(bras_mat))
    end
    if _kets_d_ref[] === nothing || size(_kets_d_ref[]::CuArray{ComplexF64,2}) != size(kets_mat)
        _kets_d_ref[] = CuArray{ComplexF64}(undef, size(kets_mat))
    end
    if _U_d_ref[] === nothing || size(_U_d_ref[]) != (nbnds1, nbnds2)
        _U_d_ref[] = CuArray{ComplexF64}(undef, nbnds1, nbnds2)
    end

    bras_d  = _bras_d_ref[]
    kets_d = _kets_d_ref[]
    U_d   = _U_d_ref[]

    copyto!(bras_d,  bras_mat)
    copyto!(kets_d, kets_mat)

    mul!(U_d, adjoint(bras_d), kets_d) 
    copyto!(U, Array(U_d))
    return U            
end

end # module
