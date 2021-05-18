module MCMC
include("./setup.jl")
include("./functions.jl")
include("./legendreTF.jl")
using .Const, .Func, .LegendreTF, Distributions, Base.Threads, LinearAlgebra

function imaginary(dirname::String, filename1::String, λ::T) where {T <: Real}
    # Initialize Traces
    traces = Vector{Func.GPcore.Trace}(undef, Const.batchsize)
    for n in 1:Const.batchsize
        xs = [rand([1f0, -1f0], Const.dimS+Const.dimB) for i in 1:Const.init]
        bimu = zeros(Float32, 2 * Const.init)
        K  = Func.GPcore.covar(xs)
        biK1 = vcat(real.(K)/2f0,  imag.(K)/2f0)
        biK2 = vcat(-imag.(K)/2f0, real.(K)/2f0)
        biK  = hcat(biK1, biK2)
        U, Δ, V = svd(K)
        invΔ = Diagonal(1f0 ./ Δ .* (Δ .> 1f-6))
        invK = V * invΔ * U'
        biys = rand(MvNormal(bimu, biK))
        ys = biys[1:Const.init] .+ im * biys[Const.init+1:end]
        ys ./= norm(ys)
        traces[n] = Func.GPcore.Trace(xs, ys, invK)
    end

    # Imaginary roop
    for it in 1:Const.iT
        # Initialize Physical Value
        eS = zeros(Complex{T}, Const.batchsize)
        eB = zeros(Complex{T}, Const.batchsize)
        eI = zeros(Complex{T}, Const.batchsize)
        e  = zeros(Complex{T}, Const.batchsize)
        nB = zeros(T, Const.batchsize)
        @threads for n in 1:Const.batchsize
            eS[n], eB[n], eI[n], e[n], nB[n] = sampling(traces[n], λ)
        end
        energyS  = real(sum(eS)) / Const.iters / Const.batchsize
        energyB  = real(sum(eB)) / Const.iters / Const.batchsize
        energyI  = real(sum(eI)) / Const.iters / Const.batchsize
        energy   = real(sum(e))  / Const.iters / Const.batchsize
        numberB  = sum(nB) / Const.iters / Const.batchsize

        # Write Data
        open(filename1, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(energy / (Const.dimB + Const.dimS)))
            write(io, "\t")
            write(io, string(energyS / Const.dimS))
            write(io, "\t")
            write(io, string(energyB / Const.dimB))
            write(io, "\t")
            write(io, string(energyI / Const.dimS))
            write(io, "\t")
            write(io, string(numberB / Const.dimB))
            write(io, "\n")
        end

        # Trace Update!
        for n in 1:Const.batchsize
            traces[n] = Func.imaginary_evolution(traces[n], λ)
        end
    end 
end

function sampling(trace::Func.GPcore.Trace, λ::T) where {T <: Real}
    energyS = 0f0im
    energyB = 0f0im
    energyI = 0f0im
    energy  = 0f0im
    numberB = 0f0
    # Metropolice sampling
    xs = Vector{Vector{T}}(undef, Const.iters)
    ys = Vector{Complex{T}}(undef, Const.iters)
    mh(trace, xs, ys)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        eS, eB, eI = Func.energy(x, y, trace, λ)
        nB = sum(@views x[1:Const.dimB])
        energyS += eS
        energyB += eB
        energyI += eI
        energy  += eS + eB + eI
        numberB += nB
    end
    return energyS, energyB, energyI, energy, numberB
end

function mh(trace::Func.GPcore.Trace, xs::Vector{Vector{T}}, ys::Vector{Complex{T}}) where {T <: Real}
    for i in 1:Const.burnintime
        outxs, outys = Func.update(trace)
    end
    for i in 1:Const.iters
        outxs, outys = Func.update(trace)
        xs[i] = outxs
        ys[i] = outys
    end
end
end
