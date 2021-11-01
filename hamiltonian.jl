using SparseArrays, LinearAlgebra

function hamiltonian(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    hamiltonian_heisenberg(i, x, y, model)
end

function hamiltonian_heisenberg(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    -c.J * (1.0 + (2.0 * exp(yflip - y) - 3.0) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)) / 4.0
end

function hamiltonian_psi(i::Integer, x::Vector{S}, model::GPmodel) where {T<:Real, S<:Real}
    hamiltonian_heisenberg_psi(i, x, model)
end

function hamiltonian_heisenberg_psi(i::Integer, x::Vector{S}, model::GPmodel) where {T<:Complex, S<:Real}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    c.J * (1.0 + (2.0 * exp(yflip) - 3.0) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)) / 4.0
end

S⁰ = sparse([1.0+0.0im 0.0; 0.0 1.0+0.0im])
S¹ = sparse([0.0 1.0+0.0im; 1.0+0.0im 0.0]) / 2.0
S² = sparse([0.0 -1.0im; 1.0im 0.0]) / 2.0
S³ = sparse([1.0+0.0im 0.0; 0.0 -1.0+0.0im]) / 2.0

function ⊗(A,B,C...)
    A = kron(A, B)
    for Ci in C
        A = kron(A, Ci)
    end
    return A
end

function set_spins(N, sites, Ss)
    list_mats = fill(S⁰, N)
    for (site, S) in zip(sites, Ss)
        list_mats[site] = S
    end
    return list_mats
end

function hamiltonian_heisenberg(N::T) where {T<:Integer}
    H  = c.J .* ⊗(set_spins(N, [N,1], [S¹, S¹])...)
    H += c.J .* ⊗(set_spins(N, [N,1], [S², S²])...)
    H += c.J .* ⊗(set_spins(N, [N,1], [S³, S³])...)
    for i in 1:N-1
        H += c.J .* ⊗(set_spins(N, [i,i+1], [S¹, S¹])...)
        H += c.J .* ⊗(set_spins(N, [i,i+1], [S², S²])...)
        H += c.J .* ⊗(set_spins(N, [i,i+1], [S³, S³])...)
    end
    return H ./ c.NSpin
end

function hamiltonian(N::T) where {T<:Integer}
    hamiltonian_heisenberg(N)
end

function setmatrix()
    T = ComplexF64
    S = Int64
    h::SparseMatrixCSC{T, S} = SparseMatrixCSC(hamiltonian(c.NSpin))
    I::SparseMatrixCSC{T, S} = SparseMatrixCSC(⊗(fill(S⁰, c.NSpin)...))
    A::SparseMatrixCSC{T, S} = SparseMatrixCSC(c.l * I - h)
    h, I, A
end

const h, I, A = setmatrix()
