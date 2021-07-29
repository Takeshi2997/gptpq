function hamiltonian(i::Integer, x::State, y::T, ψ0::T, model::GPmodel) where {T<:Complex}
    hamiltonian_ising(i, x, y, ψ0, model)
#    hamiltonian_heisenberg(x, y, model)
#    hamiltonian_XY(x, y, model)
end

function hamiltonian_heisenberg(i::Integer, x::State, y::T, ψ0, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    -c.J * (1.0 + (2.0 * exp(yflip - y) * randn(typeof(y)) / ψ0 - 3.0) * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)) / 4.0
end

function hamiltonian_ising(i::Integer, x::State, y::T, ψ0, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    -x.spin[i] * x.spin[i%c.NSpin+1] / 4.0 - c.H * exp(yflip - y) * randn(typeof(y)) / ψ0 / 2.0
end

function hamiltonian_XY(i::Integer, x::State, y::T, ψ0, model::GPmodel) where {T<:Complex}
    xflip_spin = copy(x.spin)
    xflip_spin[i] *= -1
    xflip = State(xflip_spin)
    yflip = predict(xflip, model)
    c.t * exp(yflip - y) * randn(typeof(y)) / ψ0 * (x.spin[i] * x.spin[i%c.NSpin+1] < 0)
end


