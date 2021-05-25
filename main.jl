include("./setup.jl")
include("./mcmc.jl")
using .Const, .MCMC

function main()
    # Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    # Imaginary time development
    for iλ in 1:10
        λ = iλ * 0.1f0
        # Make Data File
        filename1 = dirname * "/data" * lpad(iλ, 3, "0") * ".txt"
        filename2 = dirname * "/energy_subsystem" * lpad(iλ, 3, "0") * ".txt"
        touch(filename1)
        touch(filename2)
        MCMC.imaginary(dirname, filename1, filename2, λ)
    end
end

main()
