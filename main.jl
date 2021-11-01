include("./setup.jl")
include("./model.jl")
include("./core.jl")
using LinearAlgebra, Random, Distributions

EngArray = Vector{MersenneTwister}(undef, nthreads())
function main(filename::String)
    for i in 1:nthreads()
        EngArray[i] = MersenneTwister(i)
    end
    eng = EngArray[1]
    data_x = [rand(eng, [1.0, -1.0], c.NSpin)  for i in 1:c.NData]
    bimu = zeros(Float64, 2 * c.NData)
    biI  = Array(Diagonal(ones(Float64, 2 * c.NData)))
    biψ  = rand(MvNormal(bimu, biI))
    data_ψ = biψ[1:c.NData] .+ im * biψ[c.NData+1:end]
    model = GPmodel(data_x, data_ψ, I)

    batch_x = [rand(eng, [1.0, -1.0], c.NSpin)  for i in 1:c.NMC]

    ene = 0.0
    for k in 0:c.iT
        model = imaginarytime(model)
        ene = energy(batch_x, model)
        open("./data/" * filename, "a") do io
            write(io, string(k))
            write(io, "\t")
            write(io, string(ene))
            write(io, "\t")
            write(io, string(β))
            write(io, "\t")
            write(io, string(a.t))
            write(io, "\n")
        end
    end
end

dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir("./data")
filename  = "physicalvalue.txt"
touch("./data/" * filename)
main(filename)
