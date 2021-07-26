include("./setup.jl")

mutable struct State{T<:Real}
    spin::Vector{T}
end

mutable struct GPmodel{T<:Complex}
    data_x::Vector{State}
    data_y::Vector{T}
    pvec::Vector{T}
    KI::Array{T}
end
function GPmodel(data_x::Vector{State}, data_y::Vector{T}) where {T<:Complex}
    KI = Array{T}(undef, c.NData, c.NData)
    makematrix(KI, data_x)
    makeinverse(KI)
    pvec = KI * data_y
    GPmodel(data_x, data_y, pvec, KI)
end

function kernel(x1::State, x2::State)
    v = norm(x1.spin - x2.spin)^2
    v /= c.NSpin
    c.B * exp(-v / c.A)
end

function makematrix(K::Array{T}, data_x::Vector{State}) where{T<:Complex}
    for i in 1:length(data_x)
        for j in i:length(data_x)
            K[i, j] = kernel(data_x[i], data_x[j])
            K[j, i] = K[i, j]
        end
    end
end 

function makeinverse(KI::Array{T}) where {T<:Complex}
    # KI[:, :] = inv(KI)
    U, Δ, V = svd(KI)
    invΔ = Diagonal(1.0 ./ Δ .* (Δ .> 1e-6))
    KI[:, :] = V * invΔ * U'
end

function predict(x::State, model::GPmodel)
    data_x, data_y, pvec, KI = model.data_x, model.data_y, model.pvec, model.KI

    # Compute mu var
    kv = map(x1 -> kernel(x1, x), data_x)
    k0 = kernel(x, x)
    mu = kv' * pvec
    var = k0 - kv' * KI * kv

    # sample from gaussian
    log((exp(sqrt(var) * randn(typeof(mu)) + mu) - 1.0))
end


