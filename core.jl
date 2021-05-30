include("./functions.jl")
include("./model.jl")
using Distributions, Base.Threads, Serialization, LinearAlgebra

const filenames = ["gsdata" * lpad(it, 4, "0") * ".dat" for it in 1:c.iT]
const filename  = "physicalvalue.txt"

function it_evolution(models::Array{GPmodel})
    for it in 1:c.iT
        # Trace Update!
        batchsize = length(models)
        outdata = Vector(undef, batchsize)
        @Threads for n in 1:batchsize
            model = models[n]
            xs, ys = model.xs, model.ys
            ys′ = copy(ys)
            for n in 1:c.init
                x = xs[n]
                y = ys[n]
                e = energy(x, y, trace)
                ys′[n] = log((c.l - e / c.dim) * exp(y))
            end 
            model[n] = makemodel(xs, ys′)
            outdata[n] = (model[n].xs, model[n].ys)
        end
        open(io -> serialize(io, out), "./data/" * filenames[it])
    end
end

function measure()
    touch(filename)
    logvenergy0 = 0f0
    # Imaginary roop
    for it in 1:c.iT
        xydata = open(deserialize, "./data/" * filenames[it])
        models = Array{GPmodel}(undef, c.batchsize)
        for n in 1:c.batchsize
            xs, ys = xydata[n]
            models[n] = makemodel(xs, ys)
        end

        # Initialize Physical Value
        e  = zeros(Complex{Float32}, c.batchsize)
        ve = zeros(Complex{Float32}, c.batchsize)
        h  = zeros(Float32, c.batchsize)
        @threads for n in 1:c.batchsize
            e[n], ve[n], h[n] = sampling(models[n])
        end
        energy  = real(sum(e))  / c.iters / c.batchsize
        venergy = real(sum(ve)) / c.iters / c.batchsize
        magnet  = sum(h) / c.iters / c.batchsize
        entropy = logvenergy0 / c.N - 2f0 * it / c.N * log(c.l - energy)

        # Write Data
        open(filename, "a") do io
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
    end 
end

function sampling(model::GPmodel)
    energy  = 0f0im
    venergy = 0f0im
    magnet = 0f0
    # Metropolice sampling
    xs, ys = mh(model)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        e = energy(x, y, trace) / c.N
        h = sum(@views x[1:c.dim]) / c.N
        energy  += e
        venergy += (c.l - e) * conj(c.l - e)
        magnet  += h
    end
    return energy, venergy, magnet
end

function mh(model::GPmodel)
    outxs = Vector{Vector{Float32}}(undef, c.iters)
    outys = Vector{Complex{Float32}}(undef, c.iters)
    for i in 1:c.burnintime
        x, y = update(model)
    end
    for i in 1:c.iters
        x, y = update(model)
        outxs[i] = x
        outys[i] = y
    end
    return outxs, outys
end
