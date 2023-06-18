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


include("test_distributions.jl")

#-----------------

likelihood, prior = mvnormal(16)
posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))


#--------------------------

#for plotting
function idontwannacalliteverytime(title = "Test", everyaxis=false)


    acc_C = tuning_state.acc_C
    n_steps = tuning_state.n_steps
    Npercent = 0.3
    n_abs = 180
    standard_deviation = 0.001
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
    plot!(new_acc_C, lw=2, label="Current ratio", ylabel="Acceptance ratio")
    target_acc = algorithm.tuning.target_mfps/(algorithm.tuning.target_mfps+1)
    plot!([target_acc], st=:hline, lw=2, label="Target ratio")
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
    plot_std = plot((x, y), lw=2, label = "Current std of acc. ratio", ylabel="Standard deviation")
    plot!(ylims=(0, Inf))
    plot!(xlims=(0, Inf))
    plot!([0.003], st=:hline, lw=2, label="Convergence boundary")

    plot!(xlabel="Steps")


    p = plot(plot_acceptance, plot_delta, plot_std, layout=(3,1) ,legend=true)
    plot!(xlims=(0, Inf))
    return p
end

function save_plot(title)
    idontwannacalliteverytime(title, false)
    png(string("plots/", title, " wo steps"))
    idontwannacalliteverytime(title, true)
    png(string("plots/", title, " w steps"))
end


include("../ecmc.jl")

using Optim
include("../examples/tuning_with_optim.jl")


algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples= 10*10^4,
    nburnin = 0,
    nchains = 2,
    chain_length=8, 
    remaining_jumps_before_refresh=50,
    step_amplitude=0.008,
    factorized = false,
    step_var=0.1,
    variation_type = NoVariation(),
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(target_mfps=5, adaption_scheme=GoogleAdaption(automatic_adjusting=true), max_n_steps = 2*10^4),
);

state = sample.ecmc_state[1].n_acc/sample.ecmc_state[1].n_steps

sample = bat_sample(posterior, algorithm);
samples = sample.result;

tuning_state = sample.ecmc_tuning_state[1] # tuning state for chain 1
state = sample.ecmc_state[1]

plot(samples)
mean(abs.(mean(samples).a))
mean(abs.(mean(samples).a))

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
tuner_plot = idontwannacalliteverytime("New Google adaption", true)

tuning_state.tuned_delta






































#save 2 plots with title and legends false/true
save_plot("Google Tuner")


#comparison to mcmc
mcmc_samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5, nchains = 2)).result
plot(mcmc_samples)

mcmc_ESS = 0
for ess_per_dim in bat_eff_sample_size(mcmc_samples).result.a
    mcmc_ESS += ess_per_dim 
end
mcmc_ESS
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

