
include("performance_tests_cluster.jl")

function mvnormal(dimension)
    D = dimension
    μ = fill(0.0, D)
    σ = fill(1.0, D) #collect(range(1, 10, D))

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


#----------------------everything that should be looped over--------------
runs = 1
#ecmc state stuff:
nsamples = 1*10^6
nburnin = 0
nchains = [4]
tuning_max_n_steps = 3*10^4

distributions = [mvnormal]
dimensions = [10]
adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
#direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
direction_change_algorithms = [RefreshDirection(), ReflectDirection(), StochasticReflectDirection()]

start_deltas = [10^-1]
step_variances = [0.1]
variance_algorithms = [NormalVariation()]# evtl checken
MFPS_values = [1]
jumps_before_sample = [5] # checken
jumps_before_refresh = [100]


#mcmc state stuff:
mcmc_distributions = [mvnormal]
mcmc_dimensions = [2]
mcmc_nsamples = 1*10^6
nburninsteps_per_cycle = 10^5
nburnin_max_cycles = 60
mcmc_nchains = 4



#hmc state stuff:






#---------------initializing p_states----------
ecmc_p_states = [ECMCPerformanceState(
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


mcmc_p_states = [MCMCPerformanceState(
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



#hmc_p_states = [HMCPerformanceState(

#) for 
#]

#rastaban

#--------running stuff------------

multiple_states_run(ecmc_p_states, runs)


