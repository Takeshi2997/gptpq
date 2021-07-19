function hamiltonian(i::Integer, x::State, y::T, model::GPmodel) where {T<:Complex}
    hamiltonian_ising(i, x, y, model)
#    hamiltonian_heisenberg(x, y, model)
#    hamiltonian_XY(x, y, model)
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


