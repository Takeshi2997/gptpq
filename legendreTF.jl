module LegendreTF
using QuadGK

πf0 = Float32(π)

function f(t)
    fm = x -> -log(cosh((abs(cos(x)) / t))) / πf0 * t
    return quadgk(fm, 0f0, πf0)[1]
end

function df(t)
    dfm = x -> -log(cosh((abs(cos(x)) / t))) / πf0 + 
    tanh((abs(cos(x)) / t)) * abs(cos(x)) / t / πf0
    return quadgk(dfm, 0f0, πf0)[1]
end

function s(u, t)
    return (u - f(t)) / t
end

function ds(u, t)
    return -(u - f(t)) / t^2 - df(t) / t
end

function translate(u)
    sout = 0f0
    t = 5.0f0
    tm = 0.0f0
    tv = 0.0f0
    for n in 1:1000
        dt = ds(u, t)
        lr_t = 0.5f0 * sqrt(1.0f0 - 0.999f0^n) / (1.0f0 - 0.9f0^n)
        tm += (1.0f0 - 0.9f0) * (dt - tm)
        tv += (1.0f0 - 0.999f0) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 10.0f0^(-7))
        sout = s(u, t)
    end
    return sout
end

function calc_temperature(u)
    sout = 0f0
    t = 5.0f0
    tm = 0.0f0
    tv = 0.0f0
    for n in 1:1000
        dt = ds(u, t)
        lr_t = 0.5f0 * sqrt(1.0f0 - 0.999f0^n) / (1.0f0 - 0.9f0^n)
        tm += (1.0f0 - 0.9f0) * (dt - tm)
        tv += (1.0f0 - 0.999f0) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 10.0f0^(-7))
        sout = s(u, t)
    end
    return 1f0 / t
end

end
