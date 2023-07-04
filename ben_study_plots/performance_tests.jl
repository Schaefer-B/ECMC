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
using FileIO

include("../ecmc.jl")
include("test_distributions.jl")

ENV["JULIA_DEBUG"] = "BAT"


# IMPORTANT:
# no hmc states whatsover rn
# create_result_state for hmc states is incomplete
# create_algorithm for hmc states missing as well


#-------------structs-----------------
@with_kw mutable struct ecmc_performance_state
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburnin::Int64
    nchains::Int64
    tuning_max_n_steps

    adaption_scheme
    direction_change_algorithm
    start_delta
    tuned_deltas
    tuning_steps
    ntunings_not_converged
    step_variance
    variance_algorithm
    MFPS_value
    jumps_before_sample::Int64
    jumps_before_refresh::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct ecmc_result_state
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburnin::Int64
    nchains::Int64
    tuning_max_n_steps

    adaption_scheme
    direction_change_algorithm
    start_delta
    tuned_deltas
    tuning_steps
    ntunings_not_converged
    step_variance
    variance_algorithm
    MFPS_value
    jumps_before_sample::Int64
    jumps_before_refresh::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct mcmc_performance_state
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburninsteps_per_cycle::Int64
    nburnin_max_cycles::Int64
    nchains::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct mcmc_result_state
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburninsteps_per_cycle::Int64
    nburnin_max_cycles::Int64
    nchains::Int64

    samples
    effective_sample_size
end



#-------------needed functions----------
function get_posterior(distribution, dimension)
    likelihood, prior = distribution(dimension)
    posterior = PosteriorMeasure(likelihood, prior)
    return posterior
end



#---------------------------------

function create_algorithm(p_state::ecmc_performance_state)
    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= p_state.nsamples,
        nburnin = p_state.nburnin,
        nchains = p_state.nchains,
        chain_length = p_state.jumps_before_sample, 
        remaining_jumps_before_refresh = p_state.jumps_before_refresh,
        step_amplitude = p_state.start_delta,
        factorized = false,
        step_var = p_state.step_variance,
        variation_type = p_state.variance_algorithm,
        direction_change = p_state.direction_change_algorithm,
        tuning = MFPSTuner(target_mfps=p_state.MFPS_value, adaption_scheme=p_state.adaption_scheme, max_n_steps = p_state.tuning_max_n_steps, starting_alpha=0.1),
    )
    return algorithm
end

function create_algorithm(p_state::mcmc_performance_state)
    algorithm = MCMCSampling(
        mcalg = MetropolisHastings(),
        nsteps = p_state.nsamples, 
        nchains = p_state.nchains,
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = p_state.nburninsteps_per_cycle, max_ncycles = p_state.nburnin_max_cycles),
        convergence = BrooksGelmanConvergence(),
        #init = MCMCChainPoolInit(nsteps_init = 1000),
        )
    return algorithm
end




#---------------------------------


function run_sampling!(posterior, algorithm, p_state::ecmc_performance_state)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    for chain_id in 1:p_state.nchains
        t_state = sample.ecmc_tuning_state[chain_id]
        p_state.tuning_steps = t_state.n_steps
        if t_state.n_steps == p_state.tuning_max_n_steps
            p_state.ntunings_not_converged += 1
        end
        e_state = sample.ecmc_state[chain_id]
        push!(p_state.tuned_deltas, e_state.step_amplitude)
    end
    p_state.samples = samples
    p_state.effective_sample_size = ess

    return p_state
end


function run_sampling!(posterior, algorithm, p_state::mcmc_performance_state)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    p_state.samples = samples
    p_state.effective_sample_size = ess

    return p_state
end


function run_sampling!(posterior, algorithm, p_state::hmc_performance_state)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    p_state.samples = samples
    p_state.effective_sample_size = ess

    return p_state
end



#---------------------------------

function create_result_state(p_state::ecmc_performance_state)
    println("create result state in ecmc")
    r_state = ecmc_result_state(
    target_distribution = string(p_state.target_distribution),
    dimension = p_state.dimension,
    nsamples = p_state.nsamples,
    nburnin = p_state.nburnin,
    nchains = p_state.nchains,
    tuning_max_n_steps = p_state.tuning_max_n_steps,

    adaption_scheme = string(p_state.adaption_scheme),
    direction_change_algorithm = string(p_state.direction_change_algorithm),
    start_delta = p_state.start_delta,
    tuned_deltas = p_state.tuned_deltas,
    tuning_steps = p_state.tuning_steps,
    ntunings_not_converged = p_state.ntunings_not_converged,
    step_variance = p_state.step_variance,
    variance_algorithm = string(p_state.variance_algorithm),
    MFPS_value = p_state.MFPS_value,
    jumps_before_sample = p_state.jumps_before_sample,
    jumps_before_refresh = p_state.jumps_before_refresh,

    samples = p_state.samples,
    effective_sample_size = p_state.effective_sample_size,
    )

    return r_state
end



function create_result_state(p_state::mcmc_performance_state)
    println("create result state in mcmc")
    r_state = mcmc_result_state(
    target_distribution = string(p_state.target_distribution),
    dimension = p_state.dimension,
    nsamples = p_state.nsamples,
    nburninsteps_per_cycle = p_state.nburninsteps_per_cycle,
    nburnin_max_cycles = p_state.nburnin_max_cycles,
    nchains = p_state.nchains,

    samples = p_state.samples,
    effective_sample_size = p_state.effective_sample_size,
    )
    
    return r_state
end


function create_result_state(p_state::hmc_performance_state)

    return r_state
end

#---------------------------------

function one_run(p_state)

    posterior = get_posterior(p_state.target_distribution, p_state.dimension)

    algorithm = create_algorithm(p_state)

    run_sampling!(posterior, algorithm, p_state)
    
    result_state = create_result_state(p_state)

    return result_state
end


function multi_run(p_states)

    if p_states == []
        return []
    end
    nstates = length(p_states)
    result_states = Vector{Any}(undef, nstates)
    Threads.@threads for p_index in 1:nstates
        result_states[p_index] = one_run(p_states[p_index])
    end

    return result_states
end


function ecmc_mcmc_hmc_results(ecmc_states, mcmc_states, hmc_states)

    ecmc_results = multi_run(ecmc_states)

    mcmc_results = multi_run(mcmc_states)

    hmc_results = multi_run(hmc_states)

    return ecmc_results, mcmc_results, hmc_results
end


function save_state(p_state::ecmc_result_state)
    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")
    extension = ".jld2"
    full_name = string(location,sampler,name,extension)
    save(full_name, Dict("state" => p_state))
end

function multi_save_states(state_arr)
    for state in state_arr
        save_state(state)
    end
end


function load_state(p_state::ecmc_performance_state)
    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")
    extension = ".jld2"
    full_name = string(location,sampler,name,extension)
    saved_state = load(full_name, "state")
    return saved_state
end



#----------------------plotting functions-----------------------------------

function plot_mfps_tests(state_arr)

    


end









#----------------------everything that should be looped over--------------
#ecmc state stuff:
nsamples = 2*10^5
nburnin = 0
nchains = [4]
tuning_max_n_steps = 3*10^4

distributions = [mvnormal]
dimensions = [3]
adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]

start_deltas = [10^-1]
step_variances = [0.1]
variance_algorithms = [NormalVariation()]
MFPS_values = [1,2,3,4,5]
jumps_before_sample = [10]
jumps_before_refresh = [50]


#mcmc state stuff:
mcmc_distributions = [funnel]
mcmc_dimensions = [64]
mcmc_nsamples = 2*10^5
nburninsteps_per_cycle = 10^5
nburnin_max_cycles = 60
mcmc_nchains = 4



#hmc state stuff:



#other:
#kolgorov smirnoff test ks-Test wie ähnlich verteilungen sind
#anderson darling test
#salvatore oder lars für cluster fragen
#sum of weitghts


#---------------initializing p_states----------
ecmc_p_states = [ecmc_performance_state(
    target_distribution = distributions[dist],
    dimension = dimensions[dim],
    nsamples = nsamples,
    nburnin = nburnin,
    nchains = nchains[chain],
    tuning_max_n_steps = tuning_max_n_steps,

    adaption_scheme = adaption_schemes[ad_s],
    direction_change_algorithm = direction_change_algorithms[dir],
    start_delta = start_deltas[sdelta],
    tuned_deltas = [],
    tuning_steps = [],
    ntunings_not_converged = 0,
    step_variance = step_variances[s_var],
    variance_algorithm = variance_algorithms[v_algo],
    MFPS_value = MFPS_values[mfp],
    jumps_before_sample = jumps_before_sample[j_sam], 
    jumps_before_refresh = jumps_before_refresh[j_ref],

    samples = [],
    effective_sample_size = [],
    ) for dist=eachindex(distributions), 
        dim=eachindex(dimensions), 
        chain=eachindex(nchains),
        ad_s=eachindex(adaption_schemes),
        dir=eachindex(direction_change_algorithms), 
        sdelta=eachindex(start_deltas), 
        s_var=eachindex(step_variances), 
        v_algo=eachindex(variance_algorithms),
        mfp=eachindex(MFPS_values), 
        j_sam=eachindex(jumps_before_sample), 
        j_ref=eachindex(jumps_before_refresh)
]


mcmc_p_states = [mcmc_performance_state(
    target_distribution = mcmc_distributions[dist],
    dimension = mcmc_dimensions[dims],
    nsamples = mcmc_nsamples,
    nburninsteps_per_cycle = nburninsteps_per_cycle,
    nburnin_max_cycles = nburnin_max_cycles,
    nchains = mcmc_nchains,

    samples = [],
    effective_sample_size = [],
) for dist=eachindex(mcmc_distributions),
    dims=eachindex(mcmc_dimensions)
]



#hmc_p_states = [hmc_performance_state(

#) for 
#]

#rastaban

#--------running stuff------------

ecmc_results = multi_run(ecmc_p_states)

save_state(ecmc_results[1])
multi_save_states(ecmc_results)


katze = load_state(ecmc_p_states[1])
ecmc_results[1].target_distribution == katze.target_distribution ? println("yes") : println("no")
ecmc_results[1].samples == katze.samples ? println("yes") : println("no")

a = one_run(ecmc_p_states[1])
b = one_run(ecmc_p_states[4])

mcmc_results = multi_run(mcmc_p_states)

length(mcmc_results[1].samples)
hmc_p_states = [] # TEMP
ecmc_results, mcmc_results, hmc_results = ecmc_mcmc_hmc_results(ecmc_p_states, mcmc_p_states, hmc_p_states)

length(mcmc_results[1].samples)

mean(abs.(mean(ecmc_results[1].samples.v.a)))
mean(abs.(mean(ecmc_results[2].samples.v.a)))
mean(abs.(mean(mcmc_results[1].samples.v.a)))

plot(ecmc_results[1].samples)
plot(ecmc_results[2].samples)
plot(mcmc_results[1].samples)


#-----------results--------------
#ESS mean:
for i in eachindex(ecmc_results)
    println(string(ecmc_results[i].direction_change_algorithm, " on ", ecmc_results[i].target_distribution," ", ecmc_results[i].dimension, "D"))
    println(string("    effective_sample_size(mean) = ",round(mean(ecmc_results[i].effective_sample_size), digits=3)))
    println()
end
for i in eachindex(mcmc_results)
    println(string("MCMC MetropolisHastings()", " on ", mcmc_results[i].target_distribution," ", mcmc_results[i].dimension, "D"))
    println(string("    effective_sample_size(mean) = ",round(mean(mcmc_results[i].effective_sample_size), digits=3)))
    println()
end



mode(a.samples)
mode(b.samples)
mode(mcmc_results[1].samples)

mean(a.effective_sample_size)
mean(b.effective_sample_size)
mean(mcmc_results[1].effective_sample_size)






#--------------------------------
#--------------------------------
#OLD stuff:




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