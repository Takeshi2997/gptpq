module GaussianProcessTPQ
include("./setup.jl")
include("./core.jl")
include("./model.jl")
using Distributions, LinearAlgebra

function main()
    mkdir("./data")

    # Initialize Traces
    models = Array{GPmodel}(undef, c.batchsize)
    for n in 1:c.batchsize
        xs = [rand([1f0, -1f0], c.N) for i in 1:c.num]
        bimu = zeros(Float32, 2 * c.num)
        biI  = Array(Diagonal(ones(Float32, 2 * c.num)))
        biys = rand(MvNormal(bimu, biI))
        ψ = biys[1:c.num] .+ im * biys[c.num+1:end]
        ys = log.(ψ)
        models[n] = makemodel(xs, ys)
    end

    # GP imaginary time evolution
    it_evolution(models)

    # MCMC Sampling
    measure()
end
end
