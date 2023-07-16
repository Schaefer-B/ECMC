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
#include("test_distributions.jl")

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

@with_kw mutable struct hmc_performance_state
    target_distribution
    dimension::Int64
    nsamples::Int64
    nchains::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct hmc_result_state
    target_distribution
    dimension::Int64
    nsamples::Int64
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


function min_max_mean_ess(effective_sample_size_arr)
    a = effective_sample_size_arr

    if typeof(a[1]) == typeof(1.)
        a = [a]
    end

    min_ess = mean(x -> minimum(x), a)
    max_ess = mean(x -> maximum(x), a)
    mean_ess = mean(x -> mean(x), a)

    return min_ess, max_ess, mean_ess
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

function one_state_run(p_state, runs=1, save_all_samples=false)

    posterior = get_posterior(p_state.target_distribution, p_state.dimension)

    algorithm = create_algorithm(p_state)
    
    for run_id in 1:runs
        println("Starting run $run_id for ", string(p_state.target_distribution, p_state.dimension,"D ", p_state.direction_change_algorithm))
        run_sampling!(posterior, algorithm, p_state)
        result_state = create_result_state(p_state)
        println("Finished run $run_id for ", string(p_state.target_distribution, p_state.dimension,"D ", p_state.direction_change_algorithm))

        if run_id == 1
            save_state(result_state, run_id)
        elseif save_all_samples == true
            save_state(result_state, run_id)
        end

        save_effective_sample_size(result_state, run_id)
        println("Saved run $run_id for ", string(p_state.target_distribution, p_state.dimension,"D ", p_state.direction_change_algorithm))

        
        p_state.samples = []
        p_state.effective_sample_size = []
        println("Deleted the performance_state samples and ESS in $run_id for ", string(p_state.target_distribution, p_state.dimension,"D ", p_state.direction_change_algorithm))
    end

end


function multiple_states_run(p_states, runs=1, save_all_samples=false)
    println("Starting all runs over a perfomance_state array")
    if p_states == []
        return []
    end
    nstates = length(p_states)
    #result_states = Vector{Any}(undef, nstates)
    for p_index in 1:nstates
        println("Starting runs for performance_state $p_index")
        one_state_run(p_states[p_index], runs, save_all_samples)
        println("Finished runs for performance_state $p_index")
    end
    println("Finished all runs over a performance_state array")
end


function all_states_run(ecmc_states, mcmc_states, hmc_states, runs)

    multiple_states_run(ecmc_states, runs)
    println()
    println("ecmc runs finished")
    println()

    multiple_states_run(mcmc_states, runs)
    println()
    println("mcmc runs finished")
    println()

    multiple_states_run(hmc_states, runs)
    println()
    println("hmc runs finished")
    println()

    
end




#----------------saving functions-------------

function save_state(p_state::ecmc_result_state, run_id=1)
    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,name,name_add,extension)
    save(full_name, Dict("state" => p_state))
end


function save_effective_sample_size(p_state::ecmc_result_state, run_id=1)
    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    location_add = "effective_sample_sizes_only/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,location_add,name,name_add,extension)
    save(full_name, Dict("effective_sample_size" => p_state.effective_sample_size))
end



function multi_save_states(state_arr)
    for state in state_arr
        save_state(state)
    end
end


function load_state(p_state::ecmc_performance_state, run_id=1)
    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,name,name_add,extension)
    saved_state = load(full_name, "state")
    return saved_state
end



function load_effective_sample_sizes(p_state::ecmc_performance_state, runs=1)
    saved_ess = []

    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    location_add = "effective_sample_sizes_only/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")

    for run_id in 1:runs
        name_add = string("_", run_id)
        extension = ".jld2"
        full_name = string(location,sampler,location_add,name,name_add,extension)
        ess = load(full_name, "effective_sample_size")
        push!(saved_ess, ess)
    end
    return saved_ess
end

#----------------------plotting functions-----------------------------------

function plot_mfps_tests(state_arr, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    direction_algos_temp = [string(state_arr[i].direction_change_algorithm) for i=eachindex(state_arr)]
    direction_algos = unique(direction_algos_temp)
    

    for dir_algo in direction_algos

        dir_plot = plot(layout=(3,1))

        direction_states = state_arr[findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, state_arr)]
        dimensions = unique([direction_states[i].dimension for i=eachindex(direction_states)])

        
        dir_plot = plot!(subplot=1, title=dir_algo, xlabel="MFPS values", ylabel="Minimum of ESS")
        dir_plot = plot!(subplot=2, xlabel="MFPS values", ylabel="Maximum of ESS")
        dir_plot = plot!(subplot=3, xlabel="MFPS values", ylabel="Mean of ESS")

        for dimension in dimensions
            states = direction_states[findall(x -> x.dimension == dimension ? true : false, direction_states)]
            
            x_values = []
            min_ess_arr = []
            max_ess_arr = []
            mean_ess_arr = []
            
            for state in states
                x = state.MFPS_value
                state_ess = load_effective_sample_sizes(state, runs)
                min_ess, max_ess, mean_ess = min_max_mean_ess(state_ess)
                push!(min_ess_arr, min_ess)
                push!(max_ess_arr, max_ess)
                push!(mean_ess_arr, mean_ess)
                push!(x_values, x)
            end

            println(string(dir_algo, " ",dimension,"D MvNormal"))
            println(string("   Maximal ESS value for min(ESS): ", maximum(min_ess_arr)))
            println(string("   Maximal ESS value for max(ESS): ", maximum(max_ess_arr)))
            println(string("   Maximal ESS value for mean(ESS): ", maximum(mean_ess_arr)))


            min_ess_arr = min_ess_arr ./(maximum(min_ess_arr))
            max_ess_arr = max_ess_arr ./(maximum(max_ess_arr))
            mean_ess_arr = mean_ess_arr ./(maximum(mean_ess_arr))
            

            dir_plot = plot!(x_values, min_ess_arr, subplot=1, label=string(dimension, "D", " Multivariate Normal"), lw=2)
            dir_plot = plot!(x_values, max_ess_arr, subplot=2, label=string(dimension, "D", " Multivariate Normal"), lw=2)
            dir_plot = plot!(x_values, mean_ess_arr, subplot=3, label=string(dimension, "D", " Multivariate Normal"), lw=2)

        end
        location = "plots/"
        name = string(dir_algo, "MFPS_plot")
        full_string = string(location,name)
        png(full_string)
    end

end




function plot_jbr_tests(state_arr, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    direction_algos_temp = [string(state_arr[i].direction_change_algorithm) for i=eachindex(state_arr)]
    direction_algos = unique(direction_algos_temp)
    

    for dir_algo in direction_algos

        dir_plot = plot(layout=(3,1))

        direction_states = state_arr[findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, state_arr)]
        dimensions = unique([direction_states[i].dimension for i=eachindex(direction_states)])

        
        dir_plot = plot!(subplot=1, title=dir_algo, xlabel="Jumps before refresh", ylabel="Minimum of ESS")
        dir_plot = plot!(subplot=2, xlabel="Jumps before refresh", ylabel="Maximum of ESS")
        dir_plot = plot!(subplot=3, xlabel="Jumps before refresh", ylabel="Mean of ESS")

        for dimension in dimensions
            states = direction_states[findall(x -> x.dimension == dimension ? true : false, direction_states)]
            
            x_values = []
            min_ess_arr = []
            max_ess_arr = []
            mean_ess_arr = []
            
            for state in states
                x = state.jumps_before_refresh
                state_ess = load_effective_sample_sizes(state, runs)
                min_ess, max_ess, mean_ess = min_max_mean_ess(state_ess)
                push!(min_ess_arr, min_ess)
                push!(max_ess_arr, max_ess)
                push!(mean_ess_arr, mean_ess)
                push!(x_values, x)
            end

            println(string(dir_algo, " ",dimension,"D MvNormal"))
            println(string("   Maximal ESS value for min(ESS): ", maximum(min_ess_arr)))
            println(string("   Maximal ESS value for max(ESS): ", maximum(max_ess_arr)))
            println(string("   Maximal ESS value for mean(ESS): ", maximum(mean_ess_arr)))


            min_ess_arr = min_ess_arr ./(maximum(min_ess_arr))
            max_ess_arr = max_ess_arr ./(maximum(max_ess_arr))
            mean_ess_arr = mean_ess_arr ./(maximum(mean_ess_arr))
            

            dir_plot = plot!(x_values, min_ess_arr, subplot=1, label=string(dimension, "D", " Multivariate Normal"), lw=2)
            dir_plot = plot!(x_values, max_ess_arr, subplot=2, label=string(dimension, "D", " Multivariate Normal"), lw=2)
            dir_plot = plot!(x_values, mean_ess_arr, subplot=3, label=string(dimension, "D", " Multivariate Normal"), lw=2)

        end
        dir_plot = plot!(xaxis=:log)
        location = "plots/"
        name = string(dir_algo, "jbr_plot")
        full_string = string(location,name)
        png(full_string)
    end

end





#----------------------everything that should be looped over--------------
runs = 1
#ecmc state stuff:
nsamples = 2*10^5
nburnin = 0
nchains = [4]
tuning_max_n_steps = 3*10^4

distributions = []
dimensions = [1, 10, 50, 100]
adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
direction_change_algorithms = [ReflectDirection()]
direction_change_algorithms = [StochasticReflectDirection()]

start_deltas = [10^-1]
step_variances = [0.1]
variance_algorithms = [NormalVariation()]# evtl checken
MFPS_values = [1]
jumps_before_sample = [5] # checken
jumps_before_refresh = [100]


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

multiple_states_run(ecmc_p_states, runs)

plot_mfps_tests(ecmc_p_states, runs)

plot_jbr_tests(ecmc_p_states, runs)



results = [load_state(ecmc_p_states[i]) for i=eachindex(ecmc_p_states)]

for i in eachindex(results)
    ess_min, ess_max, ess_mean = min_max_mean_ess(results[i].effective_sample_size)
    println()
    println(results[i].dimension, "D ", results[i].target_distribution)
    println("   ESS min = ", ess_min)
    println("   ESS max = ", ess_max)
    println("   ESS mean = ", ess_mean)
    println("   samples mean diff to true mean = ", mean(abs.(mean(results[i].samples).a)))
    println("   samples mean diff to true std = ", mean(abs.(std(results[i].samples).a - fill(1.,results[i].dimension))))
    println()
end

a
#-----------results--------------

# REFLECT AND STOCHASTICREFLECT PERFOM WAY BETTER AT HIGHER DIMENSIONS
# SO REFLECT WITH MFPS 1 AND STOCHASTICREFLECT WITH MFPS 3 ARE CHOSEN
#RefreshDirection() 2D MvNormal
#   Maximal ESS value for min(ESS): 616144.9651099487
#   Maximal ESS value for max(ESS): 621101.1974870306
#   Maximal ESS value for mean(ESS): 618623.0812984895
#RefreshDirection() 32D MvNormal
#   Maximal ESS value for min(ESS): 37553.56759328657
#   Maximal ESS value for max(ESS): 40321.33789954038
#   Maximal ESS value for mean(ESS): 39084.218706463114
#ReverseDirection() 2D MvNormal
#   Maximal ESS value for min(ESS): 49346.51397217456
#   Maximal ESS value for max(ESS): 50152.39102985521
#   Maximal ESS value for mean(ESS): 49749.45250101489
#ReverseDirection() 32D MvNormal
#   Maximal ESS value for min(ESS): 1111.5841529055256
#   Maximal ESS value for max(ESS): 1528.8567862263642
#   Maximal ESS value for mean(ESS): 1304.8157368744608
#GradientRefreshDirection() 2D MvNormal
#   Maximal ESS value for min(ESS): 791446.7329209563
#   Maximal ESS value for max(ESS): 795028.1770723681
#   Maximal ESS value for mean(ESS): 793237.4549966622
#GradientRefreshDirection() 32D MvNormal
#   Maximal ESS value for min(ESS): 91092.13702099383
#   Maximal ESS value for max(ESS): 95391.91464716366
#   Maximal ESS value for mean(ESS): 93365.68368066303
#ReflectDirection() 2D MvNormal
#   Maximal ESS value for min(ESS): 800000.0
#   Maximal ESS value for max(ESS): 800000.0
#   Maximal ESS value for mean(ESS): 800000.0
#ReflectDirection() 32D MvNormal
#   Maximal ESS value for min(ESS): 459724.94200107054
#   Maximal ESS value for max(ESS): 462721.40500328236
#   Maximal ESS value for mean(ESS): 461357.7276877036
#StochasticReflectDirection() 2D MvNormal
#   Maximal ESS value for min(ESS): 800000.0
#   Maximal ESS value for max(ESS): 800000.0
#   Maximal ESS value for mean(ESS): 800000.0
#StochasticReflectDirection() 32D MvNormal
#   Maximal ESS value for min(ESS): 214384.26684142477
#   Maximal ESS value for max(ESS): 219267.91798365355
#   Maximal ESS value for mean(ESS): 217183.60694348012

#USED STUFF FOR MFPS RUNS
runs = 10
#ecmc state stuff:
nsamples = 2*10^5
nburnin = 0
nchains = [4]
tuning_max_n_steps = 3*10^4

distributions = [mvnormal]
dimensions = [2, 32]
adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]

start_deltas = [10^-1]
step_variances = [0.1]
variance_algorithms = [NoVariation()]
MFPS_values = [1,2,3,4,5]
jumps_before_sample = [10]
jumps_before_refresh = [100]







# PERFORMANCE TESTS SETTINGS:
# REFLECT:
runs = 1
#ecmc state stuff:
nsamples = 2*10^5
nburnin = 0
nchains = [4]
tuning_max_n_steps = 3*10^4

distributions = [mvnormal]
dimensions = [1, 10, 50, 100]
adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
direction_change_algorithms = [ReflectDirection()]

start_deltas = [10^-1]
step_variances = [0.1]
variance_algorithms = [NoVariation()]
MFPS_values = [1]
jumps_before_sample = [10]
jumps_before_refresh = [100]


