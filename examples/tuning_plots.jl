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
function MVGauss(dimension=16)
    D = dimension
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
    return likelihood, prior
end

#---- Funnel  --------------------------
function Funnel(dimension = 16)
    D = dimension
    truth = rand(BAT.FunnelDistribution(a=0.5, b=1., n=D), Int(1e6))

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
function Mixture()
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


#-----------------

likelihood, prior = Funnel()
posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))


#--------------------------

#for plotting
function idontwannacalliteverytime(title = "Test", everyaxis=false)

    #bucket_width = bucket
    #k_max = Int(floor(length(tuning_state.acc_C)/bucket_width))
    #for k in 1:k_max
    #    for i in (1+(k-1)*bucket_width):(k*bucket_width)
    #        acc = (tuning_state.acc_C[k*bucket_width]*k*bucket_width - tuning_state.acc_C[(1+(k-1)*bucket_width)]*(1+(k-1)*bucket_width))/(bucket_width-1)
    #        push!(new_acc_C, acc)
    #    end
    #end

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
    png(string(title, "1"))
    idontwannacalliteverytime(title, true)
    png(string(title, "2"))
end


include("../ecmc.jl")

using Optim
include("../examples/tuning_with_optim.jl")


algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=0.1*10^5,
    nburnin = 0,
    nchains = 2,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=10^-4,
    factorized = false,
    #step_var=1.5*0.04,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(adaption_scheme=GoogleAdaption(), max_n_steps = 2*10^4),
)


sample = bat_sample(posterior, algorithm);
samples = sample.result

tuning_state = sample.ecmc_tuning_state[1] # tuning state for chain 1


#---------Ben Plot-----------
tuner_plot = idontwannacalliteverytime("Naive Tuner with suppression", false)
#display(tuner_plot)

tuning_state.tuned_delta
length(tuning_state.delta_arr)
std(tuning_state.delta_arr[end-Int(floor(tuning_state.n_steps*0.3)):end])/tuning_state.tuned_delta



#save 2 plots with title and legends false/true
save_plot("Naive Tuner with suppression")

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

