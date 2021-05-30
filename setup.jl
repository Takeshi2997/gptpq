struct GP_Data{T<:Real, S<:Integer}
    # System Size
    N::S
    
    # System Param
    h::T
    J::T
    t::T
    l::T
    
    # Repeat Number
    num::S
    auxn::S
    burnintime::S
    iters::S
    iT::S
    batchsize::S
    
    # Hyper Param
    θ₁::T
    θ₂::T
end

function GP_Data()
    N = 80
    h = 1f0
    J = 1f0
    t = 1f0
    l = 0.6f0
    num = 64# 4096
    auxn = 32
    burnintime = 10
    iters = 10 # 1000
    iT    = 10 # 200
    batchsize = 32
    θ₁ = 1f0
    θ₂ = 0.1f0
    GP_Data(N, h, J, t, l, num, auxn, burnintime, iters, iT, batchsize, θ₁, θ₂)
end

const c = GP_Data()
