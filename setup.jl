struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    iT::S
    H::T
    t::T
    J::T
    Δτ::T
    A::T
    B::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    iT = 100
    H = 2.0
    t = 1.0
    J = 1.0
    Δτ = 0.05
    A = 0.2
    B = 0.1
    GP_Data(NSpin, NData, NMC, MCSkip, iT, H, t, J, Δτ, A, B)
end
c = GP_Data()

mutable struct Count
    t::Real
end
a = Count(1.0)
