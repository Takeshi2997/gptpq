include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function imaginarytime(model::GPmodel)
    data_x, data_y, ψ0 = model.data_x, model.data_y, model.ψ0
    data_y .*= log.(ψ0)
    ψ = copy(data_y)
    @threads for i in 1:c.NData
        e = localenergy(data_x[i], data_y[i], model)
        ψ[i] = (c.l - e / c.NSpin) * exp(data_y[i])
    end
    ψ ./= norm(ψ)
    data_y = log.(ψ ./ ψ0)
    GPmodel(data_x, data_y, ψ0)
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

function localenergy(x::State, y::T, model::GPmodel) where {T<:Complex}
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, model)
        eloc += e
    end
    eloc
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
