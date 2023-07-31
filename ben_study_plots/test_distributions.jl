

# every distribution has to have a dimension input
# maybe use multiple inputs for mean etc and use likelihood and prior in performance_state ?

#---- Multivariate Gaussian --------------------------
function mvnormal(dimension)
    D = dimension
    μ = fill(0.0, D)
    σ = fill(1.0, D) #collect(range(1, 10, D))

    likelihood = let D = D, μ = μ, σ = σ
        logfuncdensity(params -> begin

            return logpdf(MvNormal(μ, σ), params.a)
        end)
    end 

    prior = BAT.NamedTupleDist(
        a = Uniform.(-5*σ, 5*σ)
    )
    return likelihood, prior
end

#---- Funnel  --------------------------
function funnel(dimension)
    D = dimension

    likelihood = let D = D
        logfuncdensity(params -> begin

        return logpdf(BAT.FunnelDistribution(a=0.5, b=1., n=D), params.a)
        end)
    end 

    σ = 10*ones(D)
    prior = BAT.NamedTupleDist(
        a = Uniform.(-σ, σ)
    )
    return likelihood, prior
end

#-------------------------------------------
function mixture(dimension)
    D = dimension # not needed tho
    likelihood = logfuncdensity(params -> begin

        r1 = logpdf.(
        MixtureModel(Normal[
        Normal(-10.0, 1.2),
        Normal(0.0, 1.8),
        Normal(10.0, 2.5)], [0.1, 0.3, 0.6]), params.a[1])

        r2 = logpdf.(
        MixtureModel(Normal[
        Normal(-5.0, 2.2),
        Normal(5.0, 1.5)], [0.3, 0.7]), params.a[2])

        r3 = logpdf.(Normal(2.0, 1.5), params.a[3])

        return r1+r2+r3
    end)

    prior = BAT.NamedTupleDist(
        a = [-20..20, -20.0..20.0, -10..10]
    )

    return likelihood, prior
end


function multimodal(dimension)
    D = dimension # not needed tho
    likelihood = logfuncdensity(params -> begin

        r1 = logpdf.(
        MixtureModel(Normal[
        Normal(-1.0, 0.05),
        #Normal(0.0, 1.),
        Normal(1.0, 0.05)], [0.8, 0.2]
        ), params.a[1])

        r2 = logpdf.(
        MixtureModel(Normal[
        Normal(-1, 0.05),
        Normal(1., 0.05)], [0.2, 0.8]
        ), params.a[2])

        #r3 = logpdf.(Normal(2.0, 1.5), params.a[3])

        return r1+r2#+r3
    end)

    prior = BAT.NamedTupleDist(
        #a = [-40..40, -40.0..40.0]
        a = [-10..10, -10.0..10.0]
    )

    return likelihood, prior
end



function bimodal(dimension)
    D = dimension # not needed tho
    likelihood = logfuncdensity(params -> begin

        r1 = logpdf.(MixtureModel([Normal(-50, 1.),Normal(20, 0.5)], [0.8, 0.2]), params.a[1])

        #r2 = logpdf.(Normal(0, 1.), params.a[2])

        return r1 #+ r2
    end)

    prior = BAT.NamedTupleDist(
        #a = [-40..40, -40.0..40.0]
        a = [-40..40]
    )

    return likelihood, prior
end