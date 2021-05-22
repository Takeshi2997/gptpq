module Func
include("./gpcore.jl")
include("./setup.jl")
using .Const, .GPcore, LinearAlgebra, Random, Distributions

struct Flip
    flip::Vector{Diagonal{Float32}}
end
function Flip()
    flip = Vector{Diagonal{Float32}}(undef, Const.dim)
    for i in 1:Const.dim
        o = Diagonal(ones(Float32, Const.dim))
        o[i, i] *= -1f0
        flip[i] = o
    end
    Flip(flip)
end
const a = Flip()

function update(trace::GPcore.Trace)
    xs, ys = trace.xs, trace.ys
    x = xs[end]
    y = ys[end]
    n = Const.dim
    rng = MersenneTwister(1234)
    randomnum = rand(rng, Float32, n)
    for ix in 1:n
        xflip = copy(x)
        xflip[ix] *= -1f0
        yflip = GPcore.model(trace, xflip)
        prob  = abs2(yflip / y)
        if randomnum[ix] < prob
            x = xflip
            y = yflip
        end
    end
    return x, y
end

function hamiltonianS(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace, ix::Integer) where {T <: Real}
    out = 0f0im
    ixnext = ix + 1
    if x[ix] != x[ixnext]
        yflip = GPcore.model(trace, a.flip[ixnext] * a.flip[ix] * x)
        out  += yflip / y
    end
    return -Const.t * out / 4f0
end

function energyS(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace) where {T <: Real}
    out = 0f0im
    for ix in 1:Const.dimS-1
        out += hamiltonianS(x, y, trace, ix)
    end
    return out
end

function hamiltonianB(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace, iy::Integer) where {T <: Real}
    out = 0f0im
    iynext = iy%Const.dim + 1
    if x[iy] != x[iynext]
        yflip = GPcore.model(trace, a.flip[iynext] * a.flip[iy] * x)
        out  += yflip / y
    end
    return -Const.t * out
end

function energyB(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace) where {T <: Real}
    out = 0f0im
    for iy in 1:Const.dim
        out += hamiltonianB(x, y, trace, iy)
    end
    return out
end

function energy(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace) where {T <: Real}
    eS = 0f0im
    e  = 0f0im
    eS = energyS(x, y, trace)
    e  = energyB(x, y, trace)
    return eS, e
end

function imaginary_evolution(trace::GPcore.Trace)
    xs, ys, invK = trace.xs, trace.ys, trace.invK
    xs′ = copy(xs)
    ys′ = copy(ys)
    for n in 1:Const.init
        x = xs[n]
        y = ys[n]
        e = energyB(x, y, trace)
        ys′[n] = (Const.l - e / Const.dim) * y
    end 
    outtrace = GPcore.Trace(xs′, ys′, invK)
    return outtrace
end

end
