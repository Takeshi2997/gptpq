module MCMC
include("./setup.jl")
include("./functions.jl")
include("./legendreTF.jl")
using .Const, .Func, .LegendreTF, Distributions, Base.Threads, LinearAlgebra

function imaginary(dirname::String,filename1::String, filename2::String)
    # Initialize Traces
    traces = Vector{Func.GPcore.Trace}(undef, Const.batchsize)
    for n in 1:Const.batchsize
        xs = [rand([1f0, -1f0], Const.dim) for i in 1:Const.init]
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
        eS = zeros(Complex{Float32}, Const.batchsize)
        e  = zeros(Complex{Float32}, Const.batchsize)
        nA = zeros(Float32, Const.batchsize)
        @threads for n in 1:Const.batchsize
            eS[n], e[n], nA[n] = sampling(traces[n])
        end
        energyS = real(sum(eS)) / Const.iters / Const.batchsize
        energy  = real(sum(e))  / Const.iters / Const.batchsize
        number  = sum(nA) / Const.iters / Const.batchsize

        # Write Data
        open(filename1, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(energy / Const.dim))
            write(io, "\t")
            write(io, string(energyS / Const.dimS))
            write(io, "\t")
            write(io, string(number / Const.dim))
            write(io, "\n")
        end

        if energy / Const.dim < -0.025f0
            # Calculate inverse Temperature
            β = LegendreTF.calc_temperature(energy / Const.dim)
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
    energyS = 0f0im
    energy  = 0f0im
    number  = 0f0
    # Metropolice sampling
    xs = Vector{Vector{Float32}}(undef, Const.iters)
    ys = Vector{Complex{Float32}}(undef, Const.iters)
    mh(trace, xs, ys)
    
    # Calculate Physical Value
    for n in 1:length(xs)
        x = xs[n]
        y = ys[n]
        eS, e = Func.energy(x, y, trace)
        n = sum(x)
        energyS += eS
        energy  += e
        number  += n
    end
    return energyS, energy, number
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
