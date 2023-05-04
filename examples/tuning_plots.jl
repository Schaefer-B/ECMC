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


include("../ecmc.jl")


algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^5,
    nburnin = 0,
    nchains = 4,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=0.04,
    factorized = false,
    step_var=1.5*0.04,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(),
)


sample = bat_sample(posterior, algorithm);
samples = sample.result

tuning_state = sample.ecmc_tuning_state[1] # tuning state for chain 1

#----- Plot Acc_C -------------------------------------
plot(tuning_state.acc_C, lw=2, label="Acc_C")
target_acc = algorithm.tuning.target_mfps/(algorithm.tuning.target_mfps+1)
plot!([target_acc], st=:hline, lw=2, label="Target Acc_C")


#----- Plot Delta ------------------------------------------
N = minimum([length(tuning_state.acc_C), Int(floor(0.3*tuning_state.n_steps))])

plot(tuning_state.delta_arr[100:end], label = "delta")
plot!([mean(tuning_state.delta_arr)], st=:hline, label="mean")
plot!([mean(tuning_state.delta_arr[end-N+1:end])], st=:hline, label="mean[N:end]")

#----- Plot MFPS ------------------------------------------
plot(tuning_state.mfps_arr, st=:histogram)


mean(tuning_state.mfps_arr)
best_delta = mean(tuning_state.delta_arr[end-N+1:end])



#----- Plot samples vs. truth ------------------------------
p = plot(layout=(4,4), size=(1600, 1000))
for i in 1:D
    p = plot!(samples, i, subplot=i, legend=false)
    p = plot!(truth[i, :], subplot=i, lw=2, lc=:black, st=:stephist, normed=true)
end 
p
