using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using BenchmarkTools

gr(size=(1.3*850, 1.3*600), thickness_scaling = 1.5)


#---- Multivariate Gaussian --------------------------
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


#---- Funnel  --------------------------
D = 16
truth = rand(BAT.FunnelDistribution(a=0.5, b=1., n=D), Int(1e6))

likelihood = let D = D,
    logfuncdensity(params -> begin
       return logpdf(BAT.FunnelDistribution(a=0.5, b=1., n=D), params.a)
    end)
end 

σ = 10*ones(D)
prior = BAT.NamedTupleDist(
    a = Uniform.(-σ, σ)
)


#-------------------------------------------
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


rand(prior)

posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))

#algorithm = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5)

include("ecmc_sampler.jl")

δs = 10 .^(range(-2, 1., length=10))
ωs = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]

nδs = length(δs)
nωs = length(ωs)

mfp = Array{Float64}(undef, (nδs, nωs))
ess = Array{Float64}(undef, (nδs, nωs))
acc_C = Array{Float64}(undef, (nδs, nωs))
acc_lift = Array{Float64}(undef, (nδs, nωs))


for i in 1:length(δs), j in 1:length(ωs)
    δ = δs[i]
    ω = ωs[j]

    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples=10^4,
        nburnin = 0,
        chain_length=5, 
        remaining_jumps_before_refresh=50,
        step_amplitude=δ,
        step_var=ω,
        direction_change = RefreshDirection()
    )

    sample = bat_sample(posterior, algorithm);
    samples = sample.result

    D = totalndof(posterior)
    mfp[i, j] = sample.mfp
    acc_C[i, j] = sample.acc_C
    acc_lift[i, j] = sample.acc_lift
    
    ess[i, j] = sum(bat_eff_sample_size(samples).result.a)/(D*length(samples))
end


criterion_string = "acc_C"
criterion = eval(Meta.parse(criterion_string))

plot(criterion,
    title = criterion_string,
    xlabel="step var", ylabel="step amplitude", 
    yticks = (1:nδs, round.(δs, digits=3)),
    xticks = (1:nωs, round.(ωs, digits=3)), xrotation=30,
    st=:heatmap, color=:Blues
)
ann = [(j,i, text(round(criterion[i,j], digits=2), 8, :grey54, :center))
            for i in 1:nδs for j in 1:nωs]
annotate!(ann, linecolor=:white)


ω = ωs[findmax(criterion)[2][2]]
δ = δs[findmax(criterion)[2][1]]


#---- Tuning -----------------------------------------------------------
include("ecmc_sampler_tuning.jl")
algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^5,
    nburnin = 0,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    target_mfps = 5,
    step_amplitude=3,
    step_var=0.2,
    direction_change = RefreshDirection(),
    do_tuning = true,
)


#algorithm = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5)
sample = bat_sample(posterior, algorithm);
samples = sample.result

plot(sample.ecmc_state.acc_C, lw=2)
plot!([algorithm.target_mfps/(algorithm.target_mfps+1)], st=:hline, legend=false)

N = minimum([length(sample.ecmc_state.acc_C), Int(floor(0.3*sample.ecmc_state.n_steps))])
std(sample.ecmc_state.acc_C[end-N+1:end])

sample.ecmc_state.delta_arr[1]
plot(sample.ecmc_state.delta_arr[100:end], legend=false)
plot!([mean(sample.ecmc_state.delta_arr)], st=:hline)
plot!([mean(sample.ecmc_state.delta_arr[end-N+1:end])], st=:hline)

plot(sample.mfps_arr, st=:histogram)
mean(sample.mfps_arr)
best_delta = mean(sample.ecmc_state.delta_arr[end-N+1:end])

#---- Sampling -----------------------------------------------------------
include("ecmc_sampler.jl")
#best_delta = 0.0215
algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^5,
    nburnin = 0,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    target_mfps = 20,
    step_amplitude=best_delta,
    step_var=0.1*best_delta,
    direction_change = RefreshDirection(),
    do_tuning = false,
)


#algorithm = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5)
sample_s = bat_sample(posterior, algorithm);
samples = sample_s.result
mean(sample_s.mfps_arr)

exp(logdensityof(posterior, (a=[0.,0.,0.],)))
#----- Plotting ------------------------------
p = plot(layout=(4,4), size=(1600, 1000))
for i in 1:D
    p = plot!(samples, i, subplot=i, legend=false)
    p = plot!(truth[i, :], subplot=i, lw=2, lc=:black, st=:stephist, normed=true)
end 
p

#converged = bat_convergence(samples, BrooksGelmanConvergence())

plot(samples, thickness_scaling=0.5)