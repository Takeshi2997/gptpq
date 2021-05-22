include("./setup.jl")
include("./mcmc.jl")
using .Const, .MCMC

function main()
    # Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    # Imaginary time development

    # Make Data File
    filename1 = dirname * "/data.txt"
    filename2 = dirname * "/energy_subsystem.txt"
    touch(filename1)
    touch(filename2)
    MCMC.imaginary(dirname, filename1, filename2)
end

main()
