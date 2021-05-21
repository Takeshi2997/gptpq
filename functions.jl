module Func
include("./gpcore.jl")
include("./setup.jl")
using .Const, .GPcore, LinearAlgebra, Random, Distributions

struct Flip
    flip::Vector{Diagonal{Float32}}
end
function Flip()
    flip = Vector{Diagonal{Float32}}(undef, Const.dimB+Const.dimS)
    for i in 1:Const.dimB+Const.dimS
        o = Diagonal(ones(Float32, Const.dimB+Const.dimS))
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
    n = Const.dimB + Const.dimS
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
    ixnext = Const.dimB + (ix - Const.dimB) % Const.dimS + 1
    if x[ix] != x[ixnext]
        yflip = GPcore.model(trace, a.flip[ixnext] * a.flip[ix] * x)
        out  += 2f0 * yflip / y - 1f0
    else
        out += 1f0
    end
    return -Const.J * out / 4f0
end

function energyS(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace) where {T <: Real}
    out = 0f0im
    for ix in Const.dimB+1:Const.dimB+Const.dimS
        out += hamiltonianS(x, y, trace, ix)
    end
    return out
end

function hamiltonianB(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace, iy::Integer) where {T <: Real}
    out = 0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        yflip = GPcore.model(trace, a.flip[iynext] * a.flip[iy] * x)
        out  += yflip / y
    end
    return -Const.t * out
end

function energyB(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace) where {T <: Real}
    out = 0f0im
    for iy in 1:Const.dimB
        out += hamiltonianB(x, y, trace, iy)
    end
    return out
end

function hamiltonianI(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace, ix::Integer, iy::Integer) where {T <: Real}
    out = 0f0im
#    out = -x[ix] * x[iy]
    yflip = GPcore.model(trace, a.flip[iy] * a.flip[ix] * x)
    out  += yflip / y
    return out / 2f0
end

function energyI(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace, λ::T) where {T <: Real}
    out = 0f0im
    for iy in 1:Const.dimI
        for ix in Const.dimB+1:Const.dimB+Const.dimI
            out += hamiltonianI(x, y, trace, ix, iy)
        end
    end
    return λ * out
end

function energy(x::Vector{T}, y::Complex{T}, trace::GPcore.Trace, λ::T) where {T <: Real}
    eS = 0f0im
    eB = 0f0im
    eI = 0f0im
    eS = energyS(x, y, trace)
    eB = energyB(x, y, trace)
    eI = energyI(x, y, trace, λ)
    return eS, eB, eI
end

function imaginary_evolution(trace::GPcore.Trace, λ::T) where {T <: Real}
    xs, ys, invK = trace.xs, trace.ys, trace.invK
    xs′ = copy(xs)
    ys′ = copy(ys)
    for n in 1:Const.init
        x = xs[n]
        y = ys[n]
        eS, eB, eI = energy(x, y, trace, λ)
        ys′[n] = (Const.l - (eS + eB + eI) / (Const.dimB + Const.dimS)) * y
    end 
    outtrace = GPcore.Trace(xs′, ys′, invK)
    return outtrace
end

end
