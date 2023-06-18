using Test
using BAT
using DensityInterface
using InverseFunctions
using Distributions
using Random
using ForwardDiff
using Plots

include("../ecmc.jl")

D = 16
μ = fill(0.0, D)
σ = collect(range(1, 10, D))

truth = rand(MvNormal(μ, σ), Int(1e6))

likelihood = let D = D, μ = μ, σ = σ
    logfuncdensity(params -> begin
       return logpdf(MvNormal(μ, σ), params.a)
    end)
end 


prior = BAT.NamedTupleDist(
    a = Uniform.(-5*σ, 5*σ)
)

posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))


algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^4,
    nburnin = 10^2,
    nchains = 2,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=0.04,
    factorized = false,
    #step_var=1.5*0.04,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(),
)


sampling_result = bat_sample(posterior, algorithm)
samples = sampling_result.result


plot(samples, leftmargin=3.5Plots.mm)

# comparison: ECMC sampes vs IID samples
gr(display_type=:inline)
p = plot(layout=(4,4), size=(1600, 1000))
for i in 1:D
    p = plot!(samples, i, subplot=i, legend=false)
    p = plot!(truth[i, :], subplot=i, lw=2, lc=:black, st=:stephist, normed=true)
end 
p













