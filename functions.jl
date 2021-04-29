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
        xflip = x
        xflip[ix] *= -1f0
        yflip = GPcore.model(trace, xflip)
        prob  = abs2(yflip / y)
        (x, y) = ifelse(randomnum[ix] < prob, (xflip, yflip), (x, y))
        append!(xs, [x])
        append!(ys, [y])
        trace = GPcore.Trace(xs, ys)
    end
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

function hamiltonianI(x::Vector{Float32}, ix::Integer, iy::Integer)
    out  = 0f0im
    out += -x[ix] * x[iy]
    return Const.λ * out / 4f0
end

function energyI(x::Vector{Float32})
    out = 0f0im
    for iy in 1:Const.dimI
        ix = Const.dimB + iy
        out += hamiltonianI(x, ix, iy)
    end
    return out
end

function energy(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    eS = energyS(x, y, trace)
    eB = energyB(x, y, trace)
    eI = energyI(x)
    return eS, eB, eI
end

function imaginary_evolution(trace::GPcore.Trace)
    xs, ys = trace.xs, trace.ys
    traceinit = GPcore.Trace(xs[1:Const.init], ys[1:Const.init])
    ys′ = Vector{Float32}(undef, Const.init)
    for n in length(xs)-Const.init+1:length(xs)
        x = xs[n]
        y = ys[n]
        eS, eB, eI = energy(x, y, traceinit)
        ys′[n] = (Const.l - (eS + eB + eI)) * y
    end 
    outtrace = GPcore.Trace(xs, ys′)
    return outtrace
end

end
