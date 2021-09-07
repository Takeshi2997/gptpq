include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds, FastGaussQuadrature

const t, w = gausslegendre(11)

function imaginarytime(model::GPmodel)
    data_x, data_y, ψ0 = model.data_x, model.data_y, model.ψ0
    at0 = a.t
    at1 = at0 * exp(-1/c.NSpin)
    @threads for i in 1:c.NData
        x = data_x[i]
        y = data_y[i]
        epsi = [localenergy_func(t0, x, model) for t0 in t]
        data_y[i] = log(exp(y) * ψ0[i]^at0 - c.Δτ / 2.0 * dot(w, epsi)) - at1 * log(ψ0[i])
    end
    data_y ./= norm(data_y)
    GPmodel(data_x, data_y, ψ0)
end

function localenergy_func(t::T, x::State, model::GPmodel) where {T<:Real}
    τ = (t + 1.0) / 2.0
    f = exp(-τ * c.Δτ)
    epsi = 0.0im
    @simd for i in 1:c.NSpin
        ep = hamiltonian_psi(i, x, f, model)
        epsi += ep
    end
    epsi
end

function tryflip(x::State, model::GPmodel, eng::MersenneTwister)
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, model)
    xflip_spin = copy(x.spin)
    xflip_spin[pos] *= -1
    xflip = State(xflip_spin)
    y_new = predict(xflip, model)
    prob = exp(2 * real(y_new - y))
    x.spin[pos] *= ifelse(rand(eng) < prob, -1, 1)
    State(x.spin)
end

function physicalvals(x::State, model::GPmodel)
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, model)
        eloc += e / c.NSpin
    end
    eloc
end

function energy(x_mc::Vector{State}, model::GPmodel)
    @threads for i in 1:c.NMC
        @simd for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    ene = Folds.sum(physicalvals(x, model) for x in x_mc)
    real(ene / c.NMC)
end
