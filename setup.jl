struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    Naux::S
    NMC::S
    MCSkip::S
    H::T
    t::T
    J::T
    l::T
    η::T
    A::T
end
function GP_Data()
    NSpin = 80
    NData = 1024
    Naux = 64
    NMC = 1024
    MCSkip = 16
    H = 2.0
    t = 1.0
    J = 1.0
    l = 1.6
    η = 1e-3
    A = 0.21
    GP_Data(NSpin, NData, Naux, NMC, MCSkip, H, t, J, l, η, A)
end
c = GP_Data()


