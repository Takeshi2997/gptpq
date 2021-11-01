include("./setup.jl")
using SparseArrays

function tovector(x::Vector{T}) where {T<:Real}
    out = 1
    n = 0
    for x0 in x
        if x0 > 0.0
            out += 2^n
        end
        n += 1
    end
    ξ = zeros(T, 2^c.NSpin)
    ξ[out] += 1.0
    ξ
end

mutable struct GPmodel{T<:Complex, S<:Real}
    data_x::Vector{S}
    data_ψ::Vector{T}
    ρ::SparseMatrixCSC{T, Int64}
    pvec::Vector{T}
    KI::Array{T}
end
function GPmodel(data_x::Vector{Vector{S}}, data_ψ::Vector{T}, ρ::SparseMatrixCSC{T, Int64}) where {T<:Complex, S<:Real}
    KI = Array{T}(undef, c.NData, c.NData)
    makematrix(KI, ρ, data_x)
    makeinverse(KI)
    pvec = KI * data_ψ
    GPmodel(data_x, data_ψ, ρ, pvec, KI)
end
function GPmodel(model::GPmodel, ρ::SparseMatrixCSC{T, Int64}) where {T<:Complex}
    data_x, data_ψ = model.data_x, model.data_ψ
    GPmodel(data_x, data_ψ, ρ)
end

function kernel(ρ::SparseMatrixCSC{T, Int64}, x1::Vector{S}, x2::Vector{S})  where {T<:Complex, S<:Real}
    ξ1 = tovector(x1)
    ξ2 = tovector(x2)
    dot(ξ1, ρ * ξ2)
end

function makematrix(K::Array{T}, ρ::SparseMatrixCSC{T, Int64}, data_x::Vector{S}) where {T<:Complex, S<:Real}
    for i in 1:length(data_x)
        for j in i:length(data_x)
            K[i, j] = kernel(ρ, data_x[i], data_x[j])
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

function predict(x::Vector{T}, model::GPmodel) where {T<:Real}
    data_x, ρ, data_ψ, pvec, KI = model.data_x, model.ρ, model.data_ψ, model.pvec, model.KI

    # Compute mu var
    kv = map(x1 -> kernel(ρ, x, x1), data_x)
    k0 = kernel(ρ, x, x)
    mu = kv' * pvec
    var = k0 - kv' * KI * kv

    # sample from gaussian
    log(sqrt(var) * randn(typeof(mu)) + mu)
end
