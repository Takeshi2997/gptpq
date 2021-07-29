include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function imaginarytime(model::GPmodel)
    data_x, data_y = model.data_x, model.data_y
    ψ = copy(data_y)
    ψ0 = randn(typeof(ψ[1]), c.NData)
    @threads for i in 1:c.NData
        e = localenergy(data_x[i], data_y[i], ψ0[i], model)
        ψ[i] = (c.l - e / c.NSpin) * exp(data_y[i]) * ψ0[i]
    end
    data_y = log.(ψ ./ ψ0)
    # v = sum(ψ) / c.NData
    # data_y .-= log(v)
    GPmodel(data_x, data_y)
end

function tryflip(x::State, model::GPmodel, eng::MersenneTwister)
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, model)
    xflip_spin = copy(x.spin)
    xflip_spin[pos] *= -1
    xflip = State(xflip_spin)
    y_new = predict(xflip, model)
    prob = exp(2 * real(y_new - y)) * abs2(randn(typeof(y)) / randn(typeof(y)))
    x.spin[pos] *= ifelse(rand(eng) < prob, -1, 1)
    State(x.spin)
end

function localenergy(x::State, y::T, ψ0::T, model::GPmodel) where {T<:Complex}
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, ψ0, model)
        eloc += e
    end
    eloc
end

function physicalvals(x::State, model::GPmodel)
    y = predict(x, model)
    ψ0 = randn(typeof(y))
    eloc = 0.0im
    vloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, ψ0, model)
        eloc += e
        vloc += (c.l - e) * conj(c.l - e)
    end
    [eloc, vloc]
end

function energy(x_mc::Vector{State}, model::GPmodel)
    @threads for i in 1:c.NMC
        @simd for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    enesvec = Folds.sum(physicalvals(x, model) for x in x_mc)
    ene  = enesvec[1]
    vene = enesvec[2]
    real(ene / c.NMC), real(vene / c.NMC)
end

