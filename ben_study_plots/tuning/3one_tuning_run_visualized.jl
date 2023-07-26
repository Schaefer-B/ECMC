using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using BenchmarkTools
using HypothesisTests




include("../test_distributions.jl")

#-----------------
dims = 16
likelihood, prior = mvnormal(dims)
likelihood, prior = funnel(dims)
posterior = PosteriorMeasure(likelihood, prior);



#--------------------------

#for plotting
function idontwannacalliteverytime(title = "Test", everyaxis=false)

    gr(size=(800, 800), thickness_scaling = 1.5)

    acc_C = tuning_state.acc_C
    n_steps = tuning_state.n_steps
    Npercent = 0.3
    n_abs = 180
    standard_deviation = 0.003
    rel_dif_mean = 0.01 
    
    new_acc_C = []
    acc_C_std = []
    mean_delta = []
    current_std_arr = []

    for step in 1:n_steps

        N = Int(floor(Npercent * step))
        mean_acc_C = (acc_C[step] - acc_C[step-N])/N
        push!(new_acc_C, mean_acc_C)
        #push!(mean_delta, mean(tuning_state.delta_arr[step-N:step]))

        n_steps_eval = n_abs+Int(floor(N/2))
        current_acc_arr = []
        if step > N+n_steps_eval
            for i in 1:n_steps_eval
                c_mean = (acc_C[step-i+1] - acc_C[step-N-i+1])/N
                push!(current_acc_arr, c_mean)
            end
            current_std = std(current_acc_arr)
        else
            current_std = 0#std(current_acc_arr)
        end
        push!(current_std_arr, current_std)

    end


    plot_acceptance = plot(title=title)
    plot!(new_acc_C, lw=2, label="Current acceptance rate", ylabel="Acc. rate")
    target_acc = algorithm.tuning.tuning_convergence_check.target_acc
    plot!([target_acc], st=:hline, lw=2, label="Target acceptance rate")
    #plot!(ylims=(0.6, 1.))
    if everyaxis == true
        plot!(xlabel="Steps")
    end


    plot_delta = plot(tuning_state.delta_arr, lw=2, label = "Current delta", ylabel="Delta")
    plot!([tuning_state.tuned_delta], st=:hline, lw=2, label="Chosen delta")
    #plot!(mean_delta, lw=2, lc=:black, label="Current mean delta")
    if everyaxis == true
        plot!(xlabel="Steps")
    end
    

    start_step = 0
    for s in current_std_arr
        start_step += 1
        if s != 0
            break
        end
    end
    start_step += 2 # temporary so it looks better
    x = (start_step):(n_steps)
    y = current_std_arr[start_step:end]
    plot_std = plot((x, y), lw=2, label = "Current std of acc. rate", ylabel="Standard deviation")
    plot!(ylims=(0, Inf))
    plot!(xlims=(0, Inf))
    plot!([0.003], st=:hline, lw=2, label="Convergence boundary")

    plot!(xlabel="Steps")


    p = plot(plot_acceptance, plot_delta, plot_std, layout=(3,1) ,legend=true)
    plot!(xlims=(0, n_steps))
    return p
end

function save_plot(title)
    idontwannacalliteverytime(title, false)
    png(string("plots/", title, " wo steps"))
    idontwannacalliteverytime(title, true)
    png(string("plots/", title, " w steps"))
end


include("../../ecmc.jl")

using Optim
include("../../examples/tuning_with_optim.jl")

ENV["JULIA_DEBUG"] = "BAT"

algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples= 1*10^5,
    nburnin = 0,
    nchains = 2,
    chain_length=5, 
    remaining_jumps_before_refresh=100,
    step_amplitude = 0.5,
    factorized = false,
    step_var=0.05,
    variation_type = NoVariation(),
    direction_change = ReflectDirection(),
    #tuning = ECMCNoTuner(),
    tuning = MFPSTuner(target_acc=0.8, adaption_scheme=GoogleAdaption(automatic_adjusting=true), max_n_steps = 2*10^4, starting_alpha=0.1),
);

#state = sample.ecmc_state[1].n_acc/sample.ecmc_state[1].n_steps

run_time = @elapsed(ecmcsample = bat_sample(posterior, algorithm))
ecmc_samples = ecmcsample.result

tuning_state = ecmcsample.ecmc_tuning_state[1] # tuning state for chain 1
state = ecmcsample.ecmc_state[1]

tuner_plot = idontwannacalliteverytime("", true)

savefig(tuner_plot, "04_google_one_alpha.png")

state.n_acc/state.n_steps
#--------------------------

plot(samples, nbins=100)
using LaTeXStrings
plot(
	ecmc_samples;
    vsel=collect(1:3),
	bins = 200,
    mean=true,
    std=false,
    globalmode=false,
    marginalmode=false,
    #diagonal = Dict(),
    #upper = Dict(),
    #lower = Dict(),
    vsel_label = [L"Θ_1", L"Θ_2", L"Θ_3"]
)


# TEST START
using StatsBase
nsamples = 2*10^5
nchains = 4
algo = IIDSampling(nsamples=nchains*nsamples)
μ = fill(0.0, dims)
σ = fill(1.0, dims)
iid_sample = bat_sample(MvNormal(μ,σ), algo)
iid_samples = iid_sample.result.v
t_samples = ecmc_samples.v.a

isamples = iid_sample.result

mean(iid_samples)
std(iid_samples)

ks = []
ad = []
pchi = []
for dim in 1:dims
    marginal_samples = [t_samples[i][dim] for i=eachindex(t_samples)]
    iid_marginal_samples = [iid_samples[i][dim] for i=eachindex(iid_samples)]

    #bin_start = min(minimum(marginal_samples), minimum(iid_marginal_samples))
    #bin_end = max(maximum(marginal_samples), maximum(iid_marginal_samples))
    #nbins = 200
    #bin_width = (bin_end - bin_start)/nbins

    #samples_hist = fit(Histogram, marginal_samples,bin_start:bin_width:bin_end).weights
    #iid_samples_hist = fit(Histogram, iid_marginal_samples,bin_start:bin_width:bin_end).weights

    #samples_cdf = [sum(samples_hist[1:i]) for i=eachindex(samples_hist)]/length(samples_hist)
    #iid_samples_cdf = [sum(iid_samples_hist[1:i]) for i=eachindex(iid_samples_hist)]/length(iid_samples_hist)

    #p1 = pvalue(ApproximateTwoSampleKSTest(marginal_samples, iid_marginal_samples))
    p1 = pvalue(ApproximateTwoSampleKSTest(marginal_samples, iid_marginal_samples))
    push!(ks, p1)
    p2 = pvalue(OneSampleADTest(marginal_samples, Normal(0.,1.)))
    #p2 = pvalue(OneSampleADTest(samples_cdf, Normal(0.,1.)))
    push!(ad, p2)
    #p = pvalue(KSampleADTest(marginal_samples, iid_marginal_samples))


    #p3 = pvalue(ChisqTest(samples_hist, iid_samples_hist))
    #push!(pchi, p3)
end
bcom = BAT.bat_compare(ecmc_samples, isamples).result
ks_com = bcom.ks_p_values
s = iid_samples.v
iid_marginal_samples = [s[i][1] for i=eachindex(s)]
p = pvalue(KSampleADTest(iid_marginal_samples, iid_marginal_samples))

ks
ad
plot(fit(Histogram, ks, 0:0.05:1))
plot(fit(Histogram, ks_com, 0:0.05:1))
plot(fit(Histogram, ad, 0:0.05:1))
pchi
#c = confint(ChisqTest(h1.weights, h2.weights))
# TEST END
SampledDensity(posterior, samples)

mean(abs.(mean(samples).a))
mean(bat_eff_sample_size(samples).result)

#ESS test
ESS = 0
eff_ss_result = bat_eff_sample_size(samples).result
eff_mean = mean(eff_ss_result.a)
eff_mean = mean(eff_ss_result.a)
for ess_per_dim in bat_eff_sample_size(samples).result.a
    ESS += ess_per_dim 
end
ESS


#---------Ben Plot-----------
tuner_plot = idontwannacalliteverytime("Google tuning with adjusting, Funnel 2048D", true)

png("google_tuning_with_adjusting_high_dimensions")

tuning_state.tuned_delta





















algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples= 1*10^6,
    nburnin = 0,
    nchains = 4,
    chain_length=5, 
    remaining_jumps_before_refresh=100,
    step_amplitude = 0.1,
    factorized = false,
    step_var=0.05,
    variation_type = NormalVariation(),
    direction_change = ReflectDirection(),
    tuning = MFPSTuner(target_acc=0.5, adaption_scheme=GoogleAdaption(automatic_adjusting=true), max_n_steps = 2*10^4, starting_alpha=0.1),
);


ecmc_time = @elapsed(ecmc_sample = bat_sample(posterior, algorithm))
#mcmc_sample = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 2*10^5, nchains = 4))
ecmc_samples = ecmc_sample.result

ecmc_ess = bat_eff_sample_size(ecmc_samples).result.a
ecmc_ess_time = mean(ecmc_ess)/ecmc_time






function multimodalmixture(dimension)
    D = dimension

    likelihood = let D = D
        logfuncdensity(params -> begin

        return logpdf(BAT.MultimodalCauchy(μ = 1., σ = 0.2, n = D), params.a)
        end)
    end 

    σ = 10*ones(D)
    prior = BAT.NamedTupleDist(
        a = Uniform.(-σ, σ)
    )
    return likelihood, prior
end

likelihood, prior = funnel(2048)
posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))



algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples= 10^7,
    nburnin = 0,
    nchains = 2,
    chain_length=8, 
    remaining_jumps_before_refresh=50,
    step_amplitude = 0.1,
    factorized = false,
    step_var=0.1,
    variation_type = NormalVariation(),
    direction_change = StochasticReflectDirection(),
    tuning = MFPSTuner(target_mfps=4, adaption_scheme=GoogleAdaption(automatic_adjusting=true), max_n_steps = 4*10^4, starting_alpha=0.1),
);

#state = sample.ecmc_state[1].n_acc/sample.ecmc_state[1].n_steps

sample = bat_sample(posterior, algorithm);
samples = sample.result;






mcmc_nsamples = 1*10^5
nburninsteps_per_cycle = 0.5*10^5
nburnin_max_cycles = 100
mcmc_nchains = 4
mcmc_sampler = MCMCSampling(
        mcalg = MetropolisHastings(),
        nsteps = mcmc_nsamples, 
        nchains = mcmc_nchains,
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = nburninsteps_per_cycle, max_ncycles = nburnin_max_cycles),
        convergence = BrooksGelmanConvergence(),
)
#comparison to mcmc
mcmc_time = @elapsed(mcmc_sample = bat_sample(posterior, mcmc_sampler))
#mcmc_sample = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 2*10^5, nchains = 4))
mcmc_samples = mcmc_sample.result
v = length(mcmc_samples.v)
w = sum(mcmc_samples.weight)
wp = sum(mcmc_samples.weight)/(mcmc_nsamples*mcmc_nchains)
#plot(mcmc_samples)
mcmc_ess = bat_eff_sample_size(mcmc_samples).result.a
mcmc_ess_time = mean(mcmc_ess)/mcmc_time

mcmc_ESS = 0
for ess_per_dim in bat_eff_sample_size(mcmc_samples).result.a
    mcmc_ESS += ess_per_dim 
end
mcmc_ESS

#funnel 2048D
#samples/steps = 10^7
#nchains = 2
#mcmc ess = 
#mcmc mean = 
#mcmc std = 
#mcmc time = 
#
#ecmc ess = 
#ecmc mean = 
#ecmc std = 
#ecmc time = 

#------------------------------------------
#------------------------------------------
#------------------------------------------
#BAT-plot distribution:
plot(
    samples, #(:(mu[1]), :sigma),
    mean = false, std = false, globalmode = false, marginalmode = false,
    nbins = 200
)
png("Mixture Model")
#------------------------------------------
#delta at start
    plot(tuning_state.delta_arr[1:100], label = "delta")

plot(tuning_state.reject_step_arr[1:10])


#----- Plot Acc_C -------------------------------------
plot(tuning_state.acc_C, lw=2, label="Acc_C")
target_acc = algorithm.tuning.target_mfps/(algorithm.tuning.target_mfps+1)
plot!([target_acc], st=:hline, lw=2, label="Target Acc_C")







#----- Plot Delta ------------------------------------------
N = minimum([length(tuning_state.acc_C), Int(floor(0.3*tuning_state.n_steps))])

plot(tuning_state.delta_arr[250:30000], label = "delta")
plot!([mean(tuning_state.delta_arr)], st=:hline, label="mean")
plot!([mean(tuning_state.delta_arr[end-N+1:end])], st=:hline, label="mean[N:end]")

tuning_state.delta_arr[end]

#----- Plot MFPS ------------------------------------------
plot(tuning_state.mfps_arr, st=:histogram)


mean(tuning_state.mfps_arr)
best_delta = mean(tuning_state.delta_arr[end-N+1:end])
tuning_state.tuned_delta


#----- Plot samples vs. truth ------------------------------
D = totalndof(posterior)
p = plot(layout=(4,4), size=(1600, 1000))
for i in 1:D
    p = plot!(samples, i, subplot=i, legend=false)
    #p = plot!(truth[i, :], subplot=i, lw=2, lc=:black, st=:stephist, normed=true)
end 
p

