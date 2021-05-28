module MCMC
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Distributions, Base.Threads, Serialization, LinearAlgebra

function imaginary(dirname::String, filename1::String)
    # Initialize Traces
    traces = Vector{Func.GPcore.Trace}(undef, Const.batchsize)
    for n in 1:Const.batchsize
        xs = [rand([1f0, -1f0], Const.dim) for i in 1:Const.init]
        bimu = zeros(Float32, 2 * Const.init)
        biI  = Array(Diagonal(ones(Float32, 2 * Const.init)))
        biys = rand(MvNormal(bimu, biI))
        ys = log.(biys[1:Const.init] .+ im * biys[Const.init+1:end])
        traces[n] = Func.GPcore.makedata(xs, ys)
    end

    logvenergy0 = log(Const.l^2)
    # Imaginary roop
    for it in 1:Const.iT
        # Initialize Physical Value
        e  = zeros(Complex{Float32}, Const.batchsize)
        ve = zeros(Complex{Float32}, Const.batchsize)
        h  = zeros(Float32, Const.batchsize)
        @threads for n in 1:Const.batchsize
            e[n], ve[n], h[n] = sampling(traces[n])
        end
        energy  = real(sum(e))  / Const.iters / Const.batchsize
        venergy = real(sum(ve)) / Const.iters / Const.batchsize
        magnet  = sum(h) / Const.iters / Const.batchsize

        entropy = logvenergy0 / Const.dim - 2f0 * it / Const.dim * log(Const.l - energy)
        # Write Data
        open(filename1, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(energy))
            write(io, "\t")
            write(io, string(entropy))
            write(io, "\t")
            write(io, string(magnet))
            write(io, "\n")
        end

        logvenergy0 += log(venergy)
        # Trace Update!
        for n in 1:Const.batchsize
            traces[n] = Func.imaginary_evolution(traces[n])
        end
    end 

    # Write Data
    out = Vector(undef, Const.batchsize)
    for n in 1:Const.batchsize
        out[n] = (traces[n].xs, traces[n].ys)
    end
    open(io -> serialize(io, out), dirname * "/gsdata.dat", "w")
end

function sampling(trace::Func.GPcore.Trace)
    energy  = 0f0im
    venergy = 0f0im
    magnet = 0f0
    # Metropolice sampling
    xs, ys = mh(trace)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        e = Func.energy(x, y, trace) / Const.dim
        h = sum(@views x[1:Const.dim]) / Const.dim
        energy  += e
        venergy += (Const.l - e) * conj(Const.l - e)
        magnet  += h
    end
    return energy, venergy, magnet
end

function mh(trace::Func.GPcore.Trace)
    outxs = Vector{Vector{Float32}}(undef, Const.iters)
    outys = Vector{Complex{Float32}}(undef, Const.iters)
    for i in 1:Const.burnintime
        x, y = Func.update(trace)
    end
    for i in 1:Const.iters
        x, y = Func.update(trace)
        outxs[i] = x
        outys[i] = y
    end
    return outxs, outys
end
end
