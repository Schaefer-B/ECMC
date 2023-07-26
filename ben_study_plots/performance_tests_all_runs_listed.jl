
function run_001() # Finding the direction algorithms i want to use in further tests
    runs = 10
    #ecmc state stuff:
    nsamples = 1*10^6
    nburnin = 0
    nchains = [4]
    tuning_max_n_steps = 3*10^4

    distributions = [MvNormal]
    dimensions = [32]
    adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
    direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
    #direction_change_algorithms = [RefreshDirection()]

    start_deltas = [10^-1]
    step_variances = [0.05]
    variance_algorithms = [NormalVariation()]# evtl checken
    target_acc_values = [0.23, 0.4, 0.6, 0.8]
    #target_acc_values = [0.5]
    jumps_before_sample = [5] # checken
    jumps_before_refresh = [100]



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

    return ecmc_p_states, runs
end



function run_002() # finding the best target acceptance rate for reflect and stochasticreflect

    runs = 10
    #ecmc state stuff:
    nsamples = 1*10^6
    nburnin = 0
    nchains = [4]
    tuning_max_n_steps = 3*10^4
    
    distributions = [MvNormal]
    dimensions = [32]
    adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
    #direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
    direction_change_algorithms = [ReflectDirection(), StochasticReflectDirection()]
    #direction_change_algorithms = [RefreshDirection()]
    
    start_deltas = [10^-1]
    step_variances = [0.05]
    variance_algorithms = [NormalVariation()]# evtl checken
    target_acc_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    jumps_before_sample = [5] # checken?
    jumps_before_refresh = [100]

    
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


    return ecmc_p_states, runs
end









function run_003() # finding best jbr for ReflectDirection and StochasticReflectDirection
        
    runs = 10
    #ecmc state stuff:
    nsamples = 1*10^6
    nburnin = 0
    nchains = [4]
    tuning_max_n_steps = 3*10^4

    distributions = [MvNormal]
    dimensions = [32]
    adaption_schemes = [GoogleAdaption(automatic_adjusting=true)]
    #direction_change_algorithms = [RefreshDirection(), ReverseDirection(), GradientRefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
    direction_change_algorithms = [ReflectDirection()]
    #direction_change_algorithms = [RefreshDirection()]

    start_deltas = [10^-1]
    step_variances = [0.05]
    variance_algorithms = [NormalVariation()]# evtl checken
    target_acc_values = [0.5]
    jumps_before_sample = [5] # checken?
    jumps_before_refresh = [25, 50, 75, 100, 125, 150, 175, 200]


    #---------------initializing p_states----------
    ecmc_p_states_1 = [ECMCPerformanceState(
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

    ecmc_p_states_1 = [ecmc_p_states_1[i] for i=eachindex(ecmc_p_states_1)]

    direction_change_algorithms = [StochasticReflectDirection()]
    #direction_change_algorithms = [RefreshDirection()]

    start_deltas = [10^-1]
    step_variances = [0.05]
    variance_algorithms = [NormalVariation()]# evtl checken
    target_acc_values = [0.8]
    jumps_before_sample = [5] # checken?
    jumps_before_refresh = [25, 50, 75, 100, 125, 150, 175, 200]


    #---------------initializing p_states----------
    ecmc_p_states_2 = [ECMCPerformanceState(
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

    ecmc_p_states_2 = [ecmc_p_states_2[i] for i=eachindex(ecmc_p_states_2)]

    ecmc_p_states = vcat(ecmc_p_states_1, ecmc_p_states_2)
    return ecmc_p_states, runs
end



function run_004() # performance test reflect runs

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
    direction_change_algorithms = [ReflectDirection()]
    #direction_change_algorithms = [RefreshDirection()]

    start_deltas = [10^-1]
    step_variances = [0.05]
    variance_algorithms = [NormalVariation()]# evtl checken
    target_acc_values = [0.5] # reflect:0.5 stochasticreflect:0.8
    jumps_before_sample = [5] # checken?
    jumps_before_refresh = [100] # reflect:100 stochasticreflect:200

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


    return ecmc_p_states, runs
end


function run_005() # performance test stochasticreflect runs

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


    return ecmc_p_states, runs
end




function run_006() # mcmc runs
    runs = 10
    mcmc_distributions = [MvNormal]
    mcmc_dimensions = [1, 2, 4, 8, 16, 32, 64, 128]
    mcmc_nsamples = 1*10^6
    nburninsteps_per_cycle = 10^5
    nburnin_max_cycles = 100
    mcmc_nchains = 4

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


    return mcmc_p_states, runs
end