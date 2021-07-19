struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    H::T
    t::T
    J::T
    l::T
    A::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 5120
    MCSkip = 16
    H = 2.0
    t = 1.0
    J = 1.0
    l = 1.6
    A = 0.21
    GP_Data(NSpin, NData, NMC, MCSkip, H, t, J, l, A)
end
c = GP_Data()


