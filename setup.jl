struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    iT::S
    H::T
    t::T
    J::T
    l::T
    A::T
    B::T
    ξ::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    iT = 200
    H = 2.0
    t = 1.0
    J = 1.0
    l = 1.6
    A = 0.1
    B = 0.1
    ξ = 0.1
    GP_Data(NSpin, NData, NMC, MCSkip, iT, H, t, J, l, A, B, ξ)
end
c = GP_Data()

mutable struct Count{T<:Integer}
    t::T
end
a = Count(0)
