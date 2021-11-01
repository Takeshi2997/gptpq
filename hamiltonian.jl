using SparseArrays, LinearAlgebra

function hamiltonian(i::Integer, x::State, y::T, model::GPmodel) where {T<:Complex}
#    hamiltonian_ising(i, x, y, model)
    hamiltonian_heisenberg(i, x, y, model)
#    hamiltonian_XY(i, x, y, model)
end

function hamiltonian_heisenberg(i::Integer, x::State, y::T, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    -c.J * (1.0 + (2.0 * exp(yflip - y) - 3.0) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)) / 4.0
end

function hamiltonian_ising(i::Integer, x::State, y::T, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    -x.spin[i] * x.spin[i%c.NSpin+1] / 4.0 - c.H * exp(yflip - y) / 2.0
end

function hamiltonian_XY(i::Integer, x::State, y::T, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    c.t * exp(yflip - y) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)
end

function hamiltonian_psi(i::Integer, x::State, model::GPmodel) where {T<:Real}
#    hamiltonian_ising_psi(i, x, model)
    hamiltonian_heisenberg_psi(i, x, model)
#    hamiltonian_XY_psi(i, x, model)
end

function hamiltonian_ising_psi(i::Integer, x::State, model::GPmodel) where {T<:Real}
    y = predict(x, model)
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    -x.spin[i] * x.spin[i%c.NSpin+1] / 4.0 * exp(y) - c.H * exp(yflip) / 2.0
end

function hamiltonian_heisenberg_psi(i::Integer, x::State, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    c.J * (1.0 + (2.0 * exp(yflip) - 3.0) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)) / 4.0
end

function hamiltonian_XY_psi(i::Integer, x::State, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    c.t * exp(yflip) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)
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

function hamiltonian_ising(N::T) where {T<:Integer}
    H = ⊗(set_spins(N, [N,1], [S³, S³])...)
    for i in 1:N-1
        H += ⊗(set_spins(N, [i,i+1], [S³,S³])...)
        H += -c.H * ⊗(set_spins(N, [i], [S¹])...)
    end
    return H ./ c.NSpin
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
#     hamiltonian_ising(N)
    hamiltonian_heisenberg(N)
#    hamiltonian_XY(N)
end

function setmatrix()
    T = ComplexF64
    h::SparseMatrixCSC{T} = SparseMatrixCSC(hamiltonian(c.NSpin))
    I::SparseMatrixCSC{T} = SparseMatrixCSC(⊗(fill(S⁰, c.NSpin)...))
    A::SparseMatrixCSC{T} = SparseMatrixCSC(c.l * I - h)
    h, I, A
end

const h, I, A = setmatrix()
