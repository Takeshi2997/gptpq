using NLsolve

function fτ(t::Integer)
    x = t/c.iT
    x / (1.0 - x)
end

function f(y::Complex{Float64}, α::Float64)
    α * log(y) + (1.0 - α) * y
end

function g(y::Complex{Float64}, α::Float64, ψ::Complex{Float64})
    exp(f(y, α)) - ψ
end

function nls(func, params...; ini = [0.0])
    if typeof(ini) <: Number
        return nlsolve((vout,vin)->vout[1]=func(vin[1],params...), [ini]).zero[1]
    else
        return nlsolve((vout,vin)->vout .= func(vin,params...), ini).zero
    end
end
