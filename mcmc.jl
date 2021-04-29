module MCMC
include("./setup.jl")
include("./functions.jl")
include("./legendreTF.jl")
using .Const, .Func, .LegendreTF, Distributions, Base.Threads

function imaginary(dirname::String, filename1::String, filename2::String)
    # Initialize Traces
    traces = Vector{Func.GPcore.Trace}(undef, Const.batchsize)
    for n in 1:Const.batchsize
        xs = [rand([1f0, -1f0], Const.dimB+Const.dimS) for i in 1:Const.init]
        mu = zeros(Float32, Const.init)
        K  = Func.GPcore.covar(xs)
        ys = rand(MvNormal(mu, K)) .+ im .* rand(MvNormal(mu, K))
        traces[n] = Func.GPcore.Trace(xs, ys)
    end

    # Imaginary roop
    for it in 1:Const.iT
        # Initialize Physical Value
        eS = zeros(Complex{Float32}, Const.batchsize)
        eB = zeros(Complex{Float32}, Const.batchsize)
        eI = zeros(Complex{Float32}, Const.batchsize)
        e  = zeros(Complex{Float32}, Const.batchsize)
        nB = zeros(Float32, Const.batchsize)
        @threads for n in 1:Const.batchsize
            eS[n], eB[n], eI[n], e[n], nB[n] = sampling(traces[n])
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

        if energyB / Const.dimB < 0.1f0
            # Calculate inverse Temperature
            β = LegendreTF.calc_temperature(energyB / Const.dimB)

            # Write Energy-Temperature
            open(filename2, "a") do io
                write(io, string(β))
                write(io, "\t")
                write(io, string(energyS / Const.dimS))
                write(io, "\n")
            end
        end

        # Trace Update!
        for n in 1:Const.batchsize
            traces[n] = Func.imaginary_evolution(traces[n])
        end
    end 
end

function sampling(trace::Func.GPcore.Trace)
    traceinit = trace
    energyS = 0f0im
    energyB = 0f0im
    energyI = 0f0im
    energy  = 0f0im
    numberB = 0f0
    # Metropolice sampling
    xs, ys = mh(trace)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        eS, eB, eI = Func.energy(x, y, traceinit)
        nB = sum(@views x[1:Const.dimB])
        energyS += eS
        energyB += eB
        energyI += eI
        energy  += eS + eB + eI
        numberB += nB
    end
    return energyS, energyB, energyI, energy, numberB
end

function mh(trace::Func.GPcore.Trace)
    outxs = Vector{Vector{Float32}}(undef, Const.iters)
    outys = Vector{Complex{Float32}}(undef, Const.iters)
    for i in 1:Const.iters
        xs, ys = Func.update(trace)
        outxs[i] = xs
        outys[i] = ys
    end
    return outxs, outys
end
end
