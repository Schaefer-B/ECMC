
include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/performance_tests_cluster.jl")
#include("performance_tests_cluster.jl")


#Input           =   /net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/performance_tests_run.jl

#----------------------everything that should be looped over--------------
runs = 10
#ecmc state stuff:
nsamples = 1*10^6
nburnin = 0
nchains = [4]
tuning_max_n_steps = 3*10^4

distributions = [MvNormal]
dimensions = [1, 2, 4, 8, 16, 32, 64, 128]
adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
#direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
direction_change_algorithms = [StochasticReflectDirection()]
#direction_change_algorithms = [RefreshDirection()]

start_deltas = [10^-1]
step_variances = [0.05]
variance_algorithms = [NormalVariation()]# evtl checken
target_acc_values = [0.8] # reflect:0.5 stochasticreflect:0.8
jumps_before_sample = [5] # checken?
jumps_before_refresh = [200] # reflect:100 stochasticreflect:200



#mcmc state stuff:
mcmc_distributions = [MvNormal]
mcmc_dimensions = [1, 2, 4, 8, 16, 32, 64, 128]
mcmc_nsamples = 1*10^6
nburninsteps_per_cycle = 10^5
nburnin_max_cycles = 100
mcmc_nchains = 4



#hmc state stuff:
hmc_distributions = [MvNormal]
hmc_dimensions = [1, 2, 4, 8, 16, 32, 64, 128]
hmc_nsamples = 1*10^6
hmc_nchains = 4



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
    target_acc_value = target_acc_values[tacc],
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
        tacc=eachindex(target_acc_values), 
        j_sam=eachindex(jumps_before_sample), 
        j_ref=eachindex(jumps_before_refresh)
]



#---------------------------------------------------------


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




hmc_p_states = [HMCPerformanceState(
    target_distribution = hmc_distributions[dist],
    dimension = hmc_dimensions[dims],
    nsamples = hmc_nsamples,
    nchains = hmc_nchains,

    samples = [],
    effective_sample_size = [],
) for dist=eachindex(hmc_distributions),
    dims=eachindex(hmc_dimensions)
]


#--------running stuff------------

#multiple_states_run(ecmc_p_states, runs)
#multiple_states_run(mcmc_p_states, runs)
multiple_states_run(hmc_p_states, runs)


