include("./setup.jl")

mutable struct State{T<:Real}
    spin::Vector{T}
    shift::Vector{Vector{T}}
end
function State(x::Vector{T}) where {T<:Real}
    shift = [circshift(x, s) for s in 1:c.NSpin]
    State(x, shift)
end

mutable struct GPmodel{T<:Complex, S<:Real}
    data_x::Vector{State}
    data_ψ::Vector{T}
    pvec_r::Vector{S}
    pvec_θ::Vector{S}
    KI::Array{S}
end
function GPmodel(data_x::Vector{State}, data_ψ::Vector{T}) where {T<:Complex}
    l = length(data_x)
    KI = Array{Real{Float64}}(undef, l, l)
    makematrix(KI, data_x)
    makeinverse(KI)
    pvec_r = KI * abs.(data_ψ)
    pvec_θ = KI * angle.(data_ψ)
    GPmodel(data_x, data_ψ, pvec_r, pvec_θ, KI)
end

function kernel(x1::State, x2::State)
    v = [norm(x1.shift[n] - x2.spin)^2 for n in 1:length(x1.spin)]
    v ./= c.NSpin
    c.B * sum(exp.(-v ./ c.A))
end

function makematrix(K::Array{T}, data_x::Vector{State}) where{T<:Complex}
    for i in 1:length(data_x)
        for j in i:length(data_x)
            K[i, j] = kernel(data_x[i], data_x[j])
            K[j, i] = K[i, j]
        end
    end
end 

function makeinverse(KI::Array{T}) where {T<:Real}
    KI[:, :] = inv(KI)
    # U, Δ, V = svd(KI)
    # invΔ = Diagonal(1.0 ./ Δ .* (Δ .> 1e-6))
    # KI[:, :] = V * invΔ * U'
end

function predict(x::State, model::GPmodel)
    data_x, data_ψ, pvec_r, pvec_θ, KI = model.data_x, model.data_ψ, model.pvec_r, model.pvec_θ, model.KI

    # Compute mu var
    kv = map(x1 -> kernel(x1, x), data_x)
    k0 = kernel(x, x)
    mu_r = kv' * pvec_r
    mu_θ = kv' * pvec_θ
    var = k0 - kv' * KI * kv

    # sample from gaussian
    r = sqrt(var) * randn(typeof(mu)) + mu_r
    θ = sqrt(var) * randn(typeof(mu)) + mu_θ
    log(r) + im * θ
end
