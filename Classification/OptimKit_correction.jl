using OptimKit
@eval OptimKit function bisect(iter::HagerZhangLineSearchIterator, a::LineSearchPoint, b::LineSearchPoint)
    # applied when (a.dϕ < 0, a.ϕ <= f₀+ϵ), (b.dϕ < 0, b.ϕ > f₀+ϵ)
    θ = iter.parameters.θ
    p₀ = iter.p₀
    c₁ = iter.parameters.c₁
    c₂ = iter.parameters.c₂
    ϵ = iter.parameters.ϵ
    fmax = p₀.f + ϵ
    numfg = 0
    while true
        if (b.α - a.α) <= eps(max(b.α, one(b.α)))
            if iter.parameters.verbosity > 0
                @warn @sprintf("Linesearch bisection failure: [a, b] = [%.2e, %.2e], b-a = %.2e, dϕᵃ = %.2e, dϕᵇ = %.2e, (ϕᵇ - ϕᵃ)/(b-a) = %.2e", a.α, b.α, b.α - a.α, a.dϕ, b.dϕ, (b.ϕ - a.ϕ)/(b.α - a.α))
            end
            return a, b, numfg
        end
        αc = (1 - θ) * a.α + θ * b.α
        c = takestep(iter, αc)
        numfg += 1
        if iter.parameters.verbosity > 2
            @info @sprintf(
            """Linesearch bisect: [a, b] = [%.2e, %.2e], b-a = %.2e, dϕᵃ = %.2e, dϕᵇ = %.2e, (ϕᵇ - ϕᵃ) = %.2e
            ↪︎ c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕᵃ = %.2e, wolfe = %d, approxwolfe = %d""",
            a.α, b.α, b.α-a.α, a.dϕ, b.dϕ, (b.ϕ - a.ϕ), c.α, c.dϕ, c.ϕ - a.ϕ, checkexactwolfe(c, p₀, c₁, c₂), checkapproxwolfe(c, p₀, c₁, c₂, ϵ))
        end
        if checkexactwolfe(c, p₀, c₁, c₂) || checkapproxwolfe(c, p₀, c₁, c₂, ϵ)
            return c, c, numfg
        end
        if c.dϕ >= 0 || numfg>10 # U3.a
            return a, c, numfg
        elseif c.ϕ <= fmax # U3.b
            a = c
        else # U3.c
            # a′ = takestep(iter, a.α)
            # @info @sprintf("""ϕᵃ = %.2e, ϕᵃ′ = %.2e""", a.ϕ, a′.ϕ)
            b = c
        end
    end
end