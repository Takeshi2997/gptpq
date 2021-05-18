module GPcore
include("./setup.jl")
using .Const, Random, LinearAlgebra, Distributions, Base.Threads

struct Trace{T <: Real, S <: Complex}
    xs::Vector{Vector{T}}
    ys::Vector{S}
    invK::Array{S}
end

function model(trace::Trace, x::Vector{T}) where {T <: Real}
    # Compute mu var
    mu, var = statcalc(trace, x)

    # sample from gaussian
    y = var * randn(Complex{Float32}) + mu
end

function kernel(x::Vector{T}, y::Vector{T}) where {T <: Real}
    r = norm(x - y) / 2f0 / (Const.dimB + Const.dimS)
    Const.θ₁ * exp(-2f0 * π * r^2 / Const.θ₂)
end

function covar(xs::Vector{Vector{T}}) where {T <: Real}
    n = length(xs)
    K = zeros(Complex{T}, n, n)
    for j in 1:n
        y = xs[j]
        for i in 1:n
            x = xs[i]
            K[i, j] = kernel(x, y)
        end
    end
    return K
end

function statcalc(trace::Trace, x::Vector{T}) where {T <: Real}
    xs, ys, invK = trace.xs, trace.ys, trace.invK

    kv = [kernel(xs[i], x) for i in 1:length(xs)]
    k0 = kernel(x, x)
    
    mu = kv' * invK * ys
    var = abs(k0 - kv' * invK * kv)
    return  mu, var
end

end
