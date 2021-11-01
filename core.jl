include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function update(model::GPmodel)
    ρ = model.ρ
    ρ1 = A * (A * ρ)'
    ρ1  = ρ1 ./ tr(ρ1)
    GPmodel(model, ρ1)
end

function imaginarytime(model::GPmodel)
    model = update(model)
    data_x, data_ψ, ρ = model.data_x, model.data_y, model.ρ
    @threads for i in 1:c.NData
        x = data_x[i]
        ψ = data_ψ[i]
        epsi = localenergy_psi(x, model)
        data_ψ[i] = (c.l * ψ - epsi)
    end
    data_ψ ./= sum(data_ψ) / c.NData
    GPmodel(data_x, data_y, ρ)
end

function localenergy_func(x::Vector{T}, model::GPmodel) where {T<:Real}
    epsi = 0.0im
    @simd for i in 1:c.NSpin
        ep = hamiltonian_psi(i, x, model)
        epsi += ep
    end
    epsi
end

function tryflip(x::Vector{T}, model::GPmodel, eng::MersenneTwister) where {T<:Real}
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

function physicalvals(x::Vector{T}, model::GPmodel) where {T<:Real}
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, model)
        eloc += e / c.NSpin
    end
    eloc
end

function energy(x_mc::Vector{Vector{T}}, model::GPmodel) where {T<:Real}
    @threads for i in 1:c.NMC
        @simd for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    ene = Folds.sum(physicalvals(x, model) for x in x_mc)
    real(ene / c.NMC)
end
