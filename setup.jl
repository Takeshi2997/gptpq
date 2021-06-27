struct GP_Data{T<:Real, S<:Integer}
    # System Size
    nspin::S
    
    # System Param
    h::T
    J::T
    t::T

    # Update Param
    l::T
    
    # Repeat Number
    ndata::S
    nmc::S
    mcskip::S
    iT::S
    
    # Hyper Param
    θ₁::T
    θ₂::T
end

function GP_Data()
    nspin = 80
    h = 1.0
    J = 1.0
    t = 1.0
    l = 0.8
    ndata = 64
    nmc = 256
    mcskip = 16
    iT = 100
    θ₁ = 1.0
    θ₂ = 0.05
    GP_Data(nspin, h, J, t, l, ndata, nmc, mcskip, iT, θ₁, θ₂)
end

c = GP_Data()
