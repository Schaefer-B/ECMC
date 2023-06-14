using Test
using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using BenchmarkTools
using StatsBase
using Serialization

include("../ecmc.jl")
include("../examples/test_distributions.jl")


@with_kw mutable struct performance_state
    target_distribution
    dimension::Int64
    n_samples::Int64
    n_burnin::Int64
    n_real_chains::Int64
    tuning_max_n_steps

    adaption_scheme
    direction_algo
    start_delta
    tuned_deltas
    n_not_converged
    step_variance
    tuned_variances
    MFPS_value
    jumps_before_sample::Int64
    jumps_before_refresh::Int64

    samples
    ESS_average_per_dimension
    ESS_total_average
    ESS_max
    ESS_min

end



#-------------needed functions----------
function create_tuner_states(density, algorithm, nchains)
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
        params = params
        )  for i in 1:nchains]

    return ecmc_tuner_states
end





#calls create_tuner_states and _ecmc_tuning
function run_test!(p_state, dimension, runs=1, variance_test=false)
    
    
    nchains = p_state.n_real_chains
    nsamples = (p_state.n_samples/nchains) + p_state.n_burnin # calculation of nsamples so that chains and burnin get taken into account

    tuner = MFPSTuner(adaption_scheme=p_state.adaption_scheme, max_n_steps = p_state.tuning_max_n_steps, target_mfps = p_state.MFPS_value)

    likelihood, prior = p_state.target_distribution(dimension)
    posterior = PosteriorMeasure(likelihood, prior);

    p_state.dimension = totalndof(posterior)



    algorithm = ECMCSampler(
        #trafo = NoDensityTransform(), 
        trafo = PriorToUniform(),
        nsamples = nsamples,
        nburnin = p_state.n_burnin,
        nchains = p_state.n_real_chains,
        chain_length = p_state.jumps_before_sample, 
        remaining_jumps_before_refresh = p_state.jumps_before_refresh,
        step_amplitude = p_state.start_delta,
        factorized = false,
        #step_var=1.5*0.04,
        direction_change = p_state.direction_algo,
        tuning = tuner,
    )

    posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
    posterior_transformed, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
    shape = varshape(posterior_transformed)

    ess_array = []
    #
    # runs start
    #
    for i in 1:runs #CALCULATE MEANS OVER RUNS ETC
        #
        # tuning starts
        #
        tuner_states = create_tuner_states(posterior_transformed, algorithm, nchains)

        T = typeof(tuner_states[1])
        tuned_states = Array{T}(undef, nchains)
        #steps_per_chain = Vector{Float64}(undef, nchains)
        #deltas_per_chain = Vector{Float64}(undef, nchains)

        
        for c in 1:nchains
            tuner_samples_per_chain, tuned_states[c] = _ecmc_tuning(algorithm.tuning, posterior_transformed, algorithm, ecmc_tuner_state=tuner_states[c], chainid=c) 
            push!(p_state.tuned_deltas, tuned_states[c].tuned_delta)
            #steps_per_chain[c] = tuned_state.n_steps
            #deltas_per_chain[c] = tuned_state.tuned_delta
            if tuned_states[c].n_steps >= p_state.tuning_max_n_steps
                p_state.n_not_converged += 1
            end
        end



        #
        # ecmc states and p_state changes
        #
        ecmc_states = ECMCState(tuned_states, algorithm)

        if variance_test == true
            for state in ecmc_states
                state.step_var = p_state.step_variance
            end
        else
            for c in 1:nchains
                push!(p_state.tuned_variances, ecmc_states[c].step_var)
            end
        end


        #
        # sampling starts
        #
        samples = []
        for c in 1:nchains
            samples_per_chain = _ecmc_sample(posterior_transformed, algorithm, ecmc_state=ecmc_states[c], chainid=c) 
            push!(samples, samples_per_chain)
        end
        if variance_test == true
            for c in 1:nchains
                push!(p_state.tuned_variances, ecmc_states[c].step_var)
            end
        end

        #
        # inverse trafo and results
        #
        samples_trafo = shape.(convert_to_BAT_samples(samples, posterior_transformed))
        samples_notrafo = inverse(trafo).(samples_trafo)

        ess = bat_eff_sample_size(samples_notrafo).result.a

        #push!(p_state.samples, samples_notrafo)
        p_state.samples = samples_notrafo  # this means only the samples of the last run gets saved in p_state
        #p_state.ESS = ess
        #for i in eachindex(ess)
        #    p_state.ESS_total += ess[i]
        #end
        push!(ess_array, ess)

    end

    #
    # mean calculating starts
    #
    if runs > 1
        dim_av = mean(ess_array)
    else
        dim_av = ess_array[1]
    end
    p_state.ESS_average_per_dimension = dim_av
    
    p_state.ESS_total_average = sum(x -> dim_av[x], eachindex(dim_av))
    p_state.ESS_max = maximum(x->maximum(x), ess_array)
    p_state.ESS_min = minimum(x->minimum(x), ess_array)

    return p_state
end



#function create_performance_states()

#    return performance_states
#end

function run_all_tests(p_states, runs=1, variance_test=false)

    result_states = Vector{Any}(undef, length(p_states))

    Threads.@threads for i in eachindex(p_states)
        result_states[i] = run_test!(p_states[i], p_states[i].dimension, runs, variance_test)
    end

    return result_states
end




#----------------------everything that should be looped over--------------
n_samples = 1*10^5 # samples in total not by chain! i changed it here for test purposes and it takes burnin into account too
n_burnin = 0 # burned by chain but n_samples is calculated so that burnin doesnt change the total samples
n_real_chains = [4]
tuning_max_n_steps = 2*10^4

distributions = [funnel]
dimensions = [64]
adaption_schemes = [GoogleAdaption()]
#direction_algos = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]
direction_algos = [ReflectDirection(), TestDirection(), StochasticReflectDirection()]

start_deltas = [10^0]
step_variances = [0.01]
MFPS_values = [5]
jumps_before_sample = [5]
jumps_before_refresh = [50]


#---------------initializing p_states----------
p_start_states = [performance_state(
    target_distribution = distributions[dist],
    dimension = dimensions[dim],
    n_samples = n_samples,
    n_burnin = n_burnin,
    n_real_chains = n_real_chains[chain],
    tuning_max_n_steps = tuning_max_n_steps,

    adaption_scheme = adaption_schemes[ad_s],
    direction_algo = direction_algos[dir],
    start_delta = start_deltas[sdelta],
    tuned_deltas = [],
    n_not_converged = 0,
    step_variance = step_variances[s_var],
    tuned_variances = [],
    MFPS_value = MFPS_values[mfp],
    jumps_before_sample = jumps_before_sample[j_sam], 
    jumps_before_refresh = jumps_before_refresh[j_ref],

    samples = [],
    ESS_average_per_dimension = [],
    ESS_total_average = 0,
    ESS_max = 0,
    ESS_min = 0
    ) for dist=eachindex(distributions), 
        dim=eachindex(dimensions), 
        chain=eachindex(n_real_chains),
        ad_s=eachindex(adaption_schemes),
        dir=eachindex(direction_algos), 
        sdelta=eachindex(start_deltas), 
        s_var=eachindex(step_variances), 
        mfp=eachindex(MFPS_values), 
        j_sam=eachindex(jumps_before_sample), 
        j_ref=eachindex(jumps_before_refresh)
]

#rastaban

#--------running stuff------------
variance_test = true # false means step variance is determined by tuning instead of the array in the initialize stuff
runs = 1000 # over how many runs should the mean for ess etc be calculated?
p_end_states = run_all_tests(p_start_states, runs, variance_test);


#-----------results--------------
#sort for different direction algos
k = findall(x->x.direction_algo == ReflectDirection() ? true : false, p_end_states);
p_states_for_one_direction = p_end_states[k];

# get ess_total_average results sorted after variances
ess_total_dict = Dict(zip(
    [step_variances[i] for i=eachindex(step_variances)],
    [[p_states_for_one_direction[s_i].ESS_total_average for s_i=findall(x->x.step_variance == step_variances[p_i] ? true : false, p_states_for_one_direction)] for p_i=eachindex(step_variances)]
));

s_var_sorted_ess_dict = sort(collect(ess_total_dict)) # or sort(collect(dict), by = x->x[1])

# extract details for one step variance of interest
step_var_interest = step_variances[1]
step_var_index = findall(x->x.step_variance == step_var_interest ? true : false, p_end_states)
p_states_for_one_step_variance = p_end_states[step_var_index];
MFPS_ess_dict = Dict(zip(
    [MFPS_values[i] for i=eachindex(MFPS_values)],
    [[p_states_for_one_step_variance[s_i].ESS_total_average for s_i=findall(x->x.MFPS_value == MFPS_values[p_i] ? true : false, p_states_for_one_step_variance)] for p_i=eachindex(MFPS_values)]
));

mpfs_sorted_ess_dict = sort(collect(MFPS_ess_dict))

#-----
p_s = p_end_states[2]
plot(p_s.samples)
save_string = string("plots/", p_s.direction_algo, " funnel 64D ESS min max = ", Int(round(p_s.ESS_min, digits=0)), ", ", Int(round(p_s.ESS_max, digits=0)))
png(save_string)
round.(mean(mean(p_s.samples).a - fill(0, p_s.dimension)), digits=4)

round.(mean(p_s.samples).a, digits=4)
round.(std(p_s.samples).a, digits=4)

#-----------------------------------------------
#-----Direction algo test----------
for algo in direction_algos
    println(algo)
    k = findall(x->x.direction_algo == algo ? true : false, p_end_states);
    k_states = p_end_states[k]
    for state in k_states
        println("    ", "Average of total ESS = ", round(state.ESS_total_average, digits=2))
        println("    ", "Minimal ESS in one dimension = ", round(state.ESS_min, digits=2))
        println("    ", "Maximal ESS in one dimension = ", round(state.ESS_max, digits=2))
    end
    println()
end

p_end_states[1].ESS_average_per_dimension

#-----MFPS & jumps_before_sample test---
for jbs in jumps_before_sample
    println("Jumps before sample = ", jbs)
    k = findall(x->x.jumps_before_sample == jbs ? true : false, p_end_states);
    k_states = p_end_states[k]
    for state in k_states
        println("    ", "MFPS value = ", state.MFPS_value, ", ESS = ", round(state.ESS_total_average, digits=2), "    ", "(t_acc)^jumps_before_sample = ", round((state.MFPS_value/(state.MFPS_value+1))^(jbs), digits=2))
    end
    println()
end

#----------------
p_end_states[1].ESS_average_per_dimension
p_end_states[1].ESS_total_average

serialize("examples/saved_p_states.jls", p_end_states)

saved_states = deserialize("examples/saved_p_states.jls");
saved_states[1]

plot(test[3].samples)