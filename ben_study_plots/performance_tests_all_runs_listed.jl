
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




function run_002()
        
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
    target_acc_values = [0.6]
    jumps_before_sample = [5] # checken?
    jumps_before_refresh = [50, 100, 150, 200, 250, 300]


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