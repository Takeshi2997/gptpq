include("./setup.jl")

mutable struct State{T<:Real}
    spin::Vector{T}
#    shift::Vector{Vector{T}}
end
# function State(x::Vector{T}) where {T<:Real}
#    shift = [circshift(x, s) for s in 1:c.NSpin]
#    State(x, shift)
# end

mutable struct GPmodel{T<:Complex}
    data_x::Vector{State}
    data_y::Vector{T}
    data_z::Vector{State}
    pvec::Vector{T}
    KI::Array{T}
end
function GPmodel(data_x::Vector{State}, data_y::Vector{T}) where {T<:Complex}
    # Step 1
    data_z = [State(rand([1f0, -1f0], c.NSpin)) for i in 1:c.Naux]
    KMM = Array{T}(undef, c.Naux, c.Naux)
    makematrix(KMM, data_z)
    KMN = [kernel(data_z[i], data_x[j]) for i in 1:length(data_z), j in 1:length(data_x)]

    # Step 2
    Λ = Diagonal([kernel(data_x[i], data_x[i]) + KMN[:, i]' * (KMM \ KMN[:, i]) + 1f-6 for i in 1:length(data_x)])

    # Step3
    QMM = KMM + KMN * (Λ \ KMN')
    u₀ = KMM * (QMM \ (KMN * (Λ \ ys)))
    Σ₀ = KMM * (QMM \ KMM)
    pvec = KMM \ u₀
    KI = makeinverse(Σ₀)

    # Output
    GPmodel(xs, ys, zs, pvec, KI)
end

function kernel(x1::State, x2::State)
#    v = [norm(x1.shift[n] - x2.spin)^2 for n in 1:length(x1.spin)]
#    v ./= c.NSpin
#    sum(exp.(-v ./ c.A))
    v = norm(x1.spin - x2.spin)^2
    v /= c.NSpin
    exp(-v / c.A)
end

function makematrix(K::Array{T}, data_x::Vector{State}) where{T<:Complex}
    for i in 1:length(data_x)
        for j in i:length(data_x)
            K[i, j] = kernel(data_x[i], data_x[j])
            K[j, i] = K[i, j]
        end
    end
end 

function makeinverse(KI::Array{T}) where {T<:Complex}
    # KI[:, :] = inv(KI)
    U, Δ, V = svd(KI)
    invΔ = Diagonal(1.0 ./ Δ .* (Δ .> 1e-6))
    KI[:, :] = V * invΔ * U'
end

function predict(x::State, model::GPmodel)
    data_z, data_y, pvec, KI = model.data_z, model.data_y, model.pvec, model.KI

    # Compute mu var
    kv = map(z -> kernel(z, x), data_z)
    k0 = kernel(x, x)
    mu = kv' * pvec
    var = k0 - kv' * KI * kv

    # sample from gaussian
    log((exp(c.η * (sqrt(var) * randn(Complex{T}) + mu)) - 1.0) / c.η)
end


