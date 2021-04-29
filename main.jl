include("./setup.jl")
include("./mcmc.jl")
using .Const, .MCMC

function main()
    # Data File make
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    # Make Data File
    filename1 = dirname * "/physical_quantity.txt"
    touch(filename1)
    filename2 = dirname * "/energy_temperature.txt"
    touch(filename2)
 
    # Imaginary time development
    MCMC.imaginary(dirnam, filename1, filename2)
end

main()
