include("./setup.jl")
include("./mcmc.jl")
using .Const, .MCMC

function main()
    # Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    # Imaginary time development
    for iλ in 1:500
        λ = iλ * 0.001f0
        # Make Data File
        filename = dirname * "/data" * lpad(iλ, 3, "0") * ".txt"
        touch(filename)
        MCMC.imaginary(dirname, filename, 0.1f0)
    end
end

main()
