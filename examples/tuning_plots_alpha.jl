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


include("../ecmc.jl")
include("test_distributions.jl")



#----------------functions for tuning------------------
function create_alpha_array(α_start, α_end, steps)
    a = log(α_start)
    b = log(α_end)
    log_arr = collect(range(a, b, steps))
    arr = exp.(log_arr)
    return arr
end

function create_algorithm(start_delta, Direction, automatic_alpha=true, jb_refresh=50)
    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= 1,
        nburnin = 0,
        nchains = 1,
        chain_length=5, 
        remaining_jumps_before_refresh = jb_refresh,
        step_amplitude = start_delta,
        factorized = false,
        step_var = 0.2,
        variation_type = UniformVariation(),
        direction_change = Direction,
        tuning = MFPSTuner(adaption_scheme=GoogleAdaption(automatic_adjusting=automatic_alpha), max_n_steps = 2*10^4),
    )
    return algorithm
end

function create_tuner_states(density, algorithm, nchains, α)
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:nchains]
    

    delta = []
    for i in 1:nchains
        push!(delta, algorithm.step_amplitude)
    end
    #delta = algorithm.step_amplitude

    
    params = [0.98,
    0.207210542888343,
    0.0732514723260891,
    0.4934509024569294,
    17.587673168668637,
    2.069505973296211,
    0.6136869940758715,
    163.53188824455017]

    ecmc_tuner_states = [ECMCTunerState(
        C = initial_samples[i], 
        lift_vector = lift_vectors[i], 
        delta = delta[i], 
        tuned_delta = delta[i], 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh, 
        delta_arr = [delta[i], ],
        step_var = algorithm.step_var,
        α = α,
        params = params
        )  for i in 1:nchains]

    return ecmc_tuner_states
end



function convergence_run(density, algorithm, nchains, α)

    t_states = create_tuner_states(density, algorithm, nchains, α)

    steps_per_chain = Vector{Float64}(undef, nchains)
    deltas_per_chain = Vector{Float64}(undef, nchains)
    std_delta_per_chain = Vector{Float64}(undef, nchains)
    notconverged = zeros(nchains)

    for c in 1:nchains
        samples_per_chain, tuned_state = _ecmc_tuning(algorithm.tuning, density, algorithm, ecmc_tuner_state=t_states[c], chainid=c) 
        steps_per_chain[c] = tuned_state.n_steps
        deltas_per_chain[c] = tuned_state.tuned_delta
        N = Int(floor(algorithm.tuning.tuning_convergence_check.Npercent * tuned_state.n_steps))
        std_delta_per_chain[c] = std(tuned_state.delta_arr[end-N+1:end])/tuned_state.tuned_delta
        if tuned_state.n_steps == algorithm.tuning.max_n_steps
            notconverged[c] = 1
        end
    end

    mean_steps = mean(steps_per_chain)
    std_steps = std(steps_per_chain)
    std_delta = mean(std_delta_per_chain)
    tuned_delta = mean(deltas_per_chain)
    n_notconverged = sum(notconverged)

    return mean_steps, std_steps, std_delta, tuned_delta, n_notconverged
end



function run_all(start_delta, α_arr, runs, distribution, dimension, dir_algo, automatic_alpha)

    likelihood, prior = distribution(dimension);
    posterior = PosteriorMeasure(likelihood, prior);

    algorithm = create_algorithm(start_delta, dir_algo, automatic_alpha)

    posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
    posterior_transformed, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
    shape = varshape(posterior_transformed)

    density = posterior_transformed

    k = length(α_arr)
    mean_steps = Vector{Float64}(undef, k)
    std_steps = Vector{Float64}(undef, k)
    std_delta = Vector{Float64}(undef, k)
    tuned_delta = Vector{Float64}(undef, k)
    n_notconverged = Vector{Int64}(undef, k)

    Threads.@threads for a in 1:k
        mean_steps[a], std_steps[a], std_delta[a], tuned_delta[a], n_notconverged[a] = convergence_run(density, algorithm, runs, α_arr[a]) 
    end

    return mean_steps, std_steps, std_delta, tuned_delta, n_notconverged
end


#----------------functions for plotting------------------

function plot_one_algo(α_arr, mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged, runs)

    title = "Google Tuner without automatic adjusting"

    #---------first plot---------
    plot_alpha = plot(title=title)
    #plot!(α_arr, mean_steps, lw=2, label="Mean steps with standard deviation", xlabel="α" , ylabel="Steps", ribbon=(mean_steps-std_steps, mean_steps+std_steps))
    plot!(α_arr, mean_steps, lw=2, label="Mean steps till convergence", xlabel="α" , ylabel="Steps")
    #plot!(yaxis=:log)

    #---------second plot---------
    lower = tuned_deltas .* (1 .-std_delta_percentage)
    upper = tuned_deltas .* (1 .+std_delta_percentage)
    #plot_delta_mean = plot(α_arr, tuned_deltas, lw=2, label="Mean of tuned deltas", xlabel="α" , ylabel="Delta", ribbon=(lower, upper))
    plot_delta_mean = plot(α_arr, tuned_deltas, lw=2, label="Mean of tuned deltas", xlabel="α" , ylabel="Delta")


    #---------third plot---------
    plot_delta_std = plot(α_arr, std_delta_percentage, lw=2, label="Mean of standard deviation of delta during tuning", xlabel="α" , ylabel="%")
    if maximum(n_notconverged) > 0
        plot!(α_arr, n_notconverged/runs, lw=2, label="% not converged")
    end

    #---------combine all plots---------
    p = plot(plot_alpha, plot_delta_mean, plot_delta_std, layout=(3,1), legend=true)
    plot!(xaxis=:log)
    plot!(xlims=(α_arr[1],Inf))
    return p
end



function bigplot(runs = 50, steps=30, automatic_alpha=false)
    start_delta = 10^0
    distribution = mvnormal
    dimension = 32
    runs = runs
    automatic_alpha = automatic_alpha
    
    direction_algo = StochasticReflectDirection()
    
    α_arr = create_alpha_array(0.0001, 10, steps)
    
    mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged = run_all(start_delta, α_arr, runs, distribution, dimension, direction_algo, automatic_alpha)
    
    return plot_one_algo(α_arr, mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged, runs)
end

function smallplot(runs=50, steps=30, automatic_alpha=false)
    start_delta = 10^0
    distribution = mvnormal
    dimension = 32
    runs = runs
    automatic_alpha = automatic_alpha
    
    direction_algo = StochasticReflectDirection()
    
    α_arr = create_alpha_array(0.001, 1, steps)
    
    mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged = run_all(start_delta, α_arr, runs, distribution, dimension, direction_algo, automatic_alpha)
    
    return plot_one_algo(α_arr, mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged, runs)
end


#------------------initializing, running and plotting----------------
start_delta = 10^0
distribution = mvnormal
dimension = 32
runs = 50
automatic_alpha = false

direction_algo = StochasticReflectDirection()

α_arr = create_alpha_array(0.0001, 10, 30)

mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged = run_all(start_delta, α_arr, runs, distribution, dimension, direction_algo, automatic_alpha)

plot_one_algo(α_arr, mean_steps, std_steps, std_delta_percentage, tuned_deltas, n_notconverged, runs)




bplot = bigplot(4, 10, false)

splot = smallplot(4, 10, false)

