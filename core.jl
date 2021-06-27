include("./setup.jl")
include("./functions.jl")
include("./model.jl")
using Base.Threads, Serialization, LinearAlgebra, Random

const filenames = ["gpdata" * lpad(it, 4, "0") * ".dat" for it in 0:c.iT]
const filename  = "physicalvalue.txt"

function imaginarytime(model::GPmodel)
    for it in 1:c.iT
        # Model Update!
        xs, ys = model.xs, model.ys
        ys′ = copy(ys)
        @simd for i in 1:c.ndata
            x = xs[i]
            y = ys[i]
            e = funcenergy(x, y, model)
            ys′[i] = log((c.l - e / c.nspin) * exp(y))
        end
        v = sum(exp.(ys′)) / c.ndata
        ys′ .-= log(v)
        model = GPmodel(xs, ys′)
        x_spins = [x.spin for x in xs]
        outdata = (x_spins, ys′)
        open(io -> serialize(io, outdata), "./data/" * filenames[it+1], "w")
    end
end

function measure()
    touch("./data/" * filename)
    # Imaginary roop
    for it in 1:c.iT
        # Make model
        xydata = open(deserialize, "./data/" * filenames[it+1])
        x_spins, ys = xydata
        xs = Vector{State}(undef, c.ndata)
        for i in 1:c.ndata
            xs[i] = State(x_spins[i])
        end
        model = GPmodel(xs, ys)

        # Metropolice sampling
        x_mc = mh(model)

        # Calculate Physical Value
        ene  = 0.0im
        vene = 0.0im
        mag  = 0.0im
        @simd for x in x_mc
            y = predict(model, x)
            e = funcenergy(x, y, model) / c.nspin
            h = sum(x.spin) / c.nspin
            ene  += e
            vene += abs2(c.l - e)
            mag  += h
        end
        energy  = real(ene) / c.nmc
        venergy = vene / c.nmc
        magnet  = mag / c.nmc
        β = 2.0 * it / c.nspin / (c.l - energy)

        # Write Data
        open("./data/" * filename, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(β))
            write(io, "\t")
            write(io, string(energy))
            write(io, "\t")
            write(io, string(magnet))
            write(io, "\n")
        end
    end 
end

function mh(model::GPmodel)
    outxs = Vector{State}(undef, c.nmc)
    x = State(rand([1f0, -1f0], c.nspin))
    rng = MersenneTwister(1234)
    prob = rand(rng, c.nmc * c.mcskip)
    @inbounds for i in 1:c.nmc
        for j in 1:c.mcskip
            update!(model, x, prob[i * j])
        end
        outxs[i] = x
    end
    outxs
end
