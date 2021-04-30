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
    n = length(x)
    rng = MersenneTwister(1234)
    randomnum = rand(rng, Float32, n)
    for ix in 1:n
        xflip = copy(x)
        xflip[ix] *= -1f0
        yflip = GPcore.model(trace, xflip)
        prob  = abs2(yflip / y)
        a = x[ix]
        if randomnum[ix] < prob
            x = xflip
            y = yflip
        end
    end
    return x, y
end

function hamiltonianS(x::Vector{Float32}, y::Complex{Float32}, 
                      trace::GPcore.Trace, ix::Integer)
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

function energyS(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    out = 0f0im
    for ix in Const.dimB+1:Const.dimB+Const.dimS
        out += hamiltonianS(x, y, trace, ix)
    end
    return out
end

function hamiltonianB(x::Vector{Float32}, y::Complex{Float32}, 
                      trace::GPcore.Trace, iy::Integer)
    out = 0f0im
    iynext = iy%Const.dimB + 1
    if x[iy] != x[iynext]
        yflip = GPcore.model(trace, a.flip[iynext] * a.flip[iy] * x)
        out  += yflip / y
    end
    return -Const.t * out
end

function energyB(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    out = 0f0im
    for iy in 1:Const.dimB
        out += hamiltonianB(x, y, trace, iy)
    end
    return out
end

function hamiltonianI(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace, ix::Integer, iy::Integer)
    out = 0f0im
    out = -x[ix] * x[iy]
#    if x[ix] != x[iy]
#        yflip = GPcore.model(trace, a.flip[iy] * a.flip[ix] * x)
#        out  += yflip / y
#    end
    return Const.λ * out / 2f0
end

function energyI(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    out = 0f0im
    for iy in 1:Const.dimI
        ix = Const.dimB + iy
        out += hamiltonianI(x, trace, ix, iy)
    end
    return out
end

function energy(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    eS = 0f0im
    eB = 0f0im
    eI = 0f0im
    eS = energyS(x, y, trace)
    eB = energyB(x, y, trace)
    eI = energyI(x, y, trace)
    return eS, eB, eI
end

function imaginary_evolution(trace::GPcore.Trace)
    xs, ys = trace.xs, trace.ys
    xs′ = copy(xs)
    ys′ = copy(ys)
    for n in 1:Const.init
        x = xs[n]
        y = ys[n]
        eS, eB, eI = energy(x, y, trace)
        ys′[n] = (Const.l - (eS + eB + eI) / (Const.dimB + Const.dimS)) * y
    end 
    outtrace = GPcore.Trace(xs′, ys′)
    return outtrace
end

end
