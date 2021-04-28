include("./setup.jl")
include("./functions.jl")
using .Const, Distributions, Base.Threads

function main()
    # Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    # Imaginary time development
    imaginary(dirname)
end

function imaginary(dirname::String)

    for it in 1:Const.iT
        # Initialize Physical Value
        energyS = 0f0im
        energyB = 0f0im
        energyI = 0f0im
        energy  = 0f0im
        numberB = 0f0
        @threads for n in 1:Const.batchsize
            sampling(energyS, energyB, energyI, energy, numberB)
        end
        energyS  = real(energyS) / Const.iters / Const.batchsize
        energyB  = real(energyB) / Const.iters / Const.batchsize
        energyI  = real(energyI) / Const.iters / Const.batchsize
        energy   = real(energy)  / Const.iters / Const.batchsize
        numberB /= Const.iters / Const.batchsize
       
        # Write Data
        filename = dirnameerror * "/physical_quantity" * lpad(it, 3, "0") * ".txt"
        touch(filename)
        open(filename, "a") do io
            write(io, string(n))
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
    end 
end

function sampling(energyS::Complex, energyB::Complex, energyI::Complex, energy::Complex, numberB::Real)
    # Metropolice sampling
    xs = [rand([1f0, -1f0], Const.dimB+Const.dimS) for i in 1:Const.init]
    mu = zeros(Float32, Const.init)
    K  = Func.GPcore.covar(xs)
    ys = rand(MvNormal(mu, K)) .+ im .* rand(MvNormal(mu, K))
    trace   = Func.GPcore.Trace(xs, ys)
    calcxs, calcys = mh(trace)
    
    # Calculate Physical Value
    for n in 1:length(calcxs)
        x = calcxs[n]
        y = calcys[n]
        eS, eB, eI = Func.energy(x, y)
        nB = sum(@views x[1:Const.dimB])
        energyS += eS
        energyB += eB
        energyI += eI
        energy  += eS + eB + eI
        numberB += nB
    end
end

function mh(trace::Func.GPcore.Trace)
    outxs = Vector{Vector{Float32}}(undef, Const.iters)
    outys = Vector{Complex{Float32}}(undef, Const.iters)
    for i in 1:Const.iters
        Func.update(trace)
        xs, ys = trace.xs[end-Const.init:end], trace.ys[end-Const.init:end]
        outxs[i] = xs[end]
        outys[i] = ys[end]
        trace = Func.GPcore.Trace(xs, ys)
    end
    return outxs, outys
end

main()
