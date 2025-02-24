
#TODO: bat_sample(ecmc_tuner_states)
# This is the top-level function called internally when running "bat_sample" with the ECMC sampler
function bat_sample_impl(
    #rng::AbstractRNG,
    target,
    algorithm::ECMCSampler,
    context::BATContext
)
    # transforming target distribution
    density_notrafo = convert(AbstractMeasureOrDensity, target)
    density, trafo = transform_and_unshape(algorithm.trafo, density_notrafo, context)
    shape = varshape(density)
   
    # tuning the ECMC sampler
    tuning_samples, ecmc_tuning_states = _ecmc_multichain_tuning(algorithm.tuning, density, algorithm)
    ecmc_states = ECMCState(ecmc_tuning_states, algorithm)

    # actually running the ECMC sampling 
    samples, ecmc_state = _ecmc_multichain_sample(density, algorithm, ecmc_states = ecmc_states)

    # TODO
    #samples = vec(samples) #TODO: concetenate according to chain id
    #samples = reduce(vcat, samples)
    #logvals = map(logdensityof(density), samples)
    #weights = ones(length(samples))
    #samples_trafo = shape.(DensitySampleVector(samples, logvals, weight = weights))

    # transforming samples back to the original space
    samples_trafo = shape.(convert_to_BAT_samples(samples, density))
    samples_notrafo = inverse(trafo).(samples_trafo)

    # checking convergence of ECMC chains
    vt = BAT.bat_convergence(samples_trafo, GelmanRubinConvergence()).result
    converged = convert(Bool, vt)
    success_str = converged ? "have" : "have *not*"
    @info "Chains $success_str converged, max(R^2) = $(vt.value), threshold = $(vt.threshold)"

    return (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, ecmc_state=ecmc_states, ecmc_tuning_state=ecmc_tuning_states)
end


#----- Run Tuning ----------------------------------------
function _ecmc_multichain_tuning(
    tuner::ECMCTuner,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_tuner_states::Vector{ECMCTunerState} = ECMCTunerState(density, algorithm) # TODO
)
    tuning_samples = [] #TODO: type, don't push
    
    #p = Progress(algorithm.nchains)
    Threads.@threads for c in 1:algorithm.nchains
        tuning_samples_per_chain, ecmc_tuner_state = _ecmc_tuning(tuner, density, algorithm, ecmc_tuner_state=ecmc_tuner_states[c], chainid=c) 
        push!(tuning_samples, tuning_samples_per_chain)
        #next!(p)
    end

    tuning_samples, ecmc_tuner_states
end


function _ecmc_tuning(
    tuner::MFPSTuner,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_tuner_state::ECMCTunerState = ECMCTunerState(density, algorithm), #TODO allow to pass ECMCTunerState for multiple cycles
    chainid::Int64 = 0,
)
    tuning_converged = false
 
    tuning_samples = [] #TODO
    @showprogress 1 "MFPS tuning for chainid $chainid" for i in 1:tuner.max_n_steps
        push!(tuning_samples, _run_ecmc!(ecmc_tuner_state, density, algorithm))
        tuning_converged = check_tuning_convergence!(ecmc_tuner_state, algorithm.tuning.tuning_convergence_check)
        
        tuning_converged ? break : nothing
        ecmc_tuner_state.n_steps >= tuner.max_n_steps ? break : nothing #neccessary?
    end
  
    has_converged_str = tuning_converged ? " " : " has *not* "
    @info "MFPS tuning for chainid $chainid" * has_converged_str * "converged after $(ecmc_tuner_state.n_steps) steps. New step_amplitude = $(ecmc_tuner_state.tuned_delta)"
    
    #mfp = ecmc_tuner_state.mfp/(algorithm.chain_length*algorithm.nsamples)
    #mfps = ecmc_tuner_state.mfps/(algorithm.chain_length*algorithm.nsamples)
    #n_acc = ecmc_tuner_state.n_acc / (algorithm.chain_length*algorithm.nsamples)
    #n_acc_lifts = ecmc_tuner_state.n_acc_lifts / ecmc_tuner_state.n_lifts

    return tuning_samples, ecmc_tuner_state
end


function _ecmc_tuning(
    tuner::ECMCNoTuner,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_tuner_state::ECMCTunerState = ECMCTunerState(density, algorithm), #TODO allow to pass ECMCTunerState for multiple cycles
    chainid::Int64 = 0,
)
    T = typeof(ecmc_tuner_state.C)
    tuning_samples =Array{T}(undef,  (algorithm.nchains, algorithm.nsamples)) #TODO: unknown size != nsamples
    @info "No ECMC tuning for chain $chainid"
    
    return tuning_samples, ecmc_tuner_state
end


#------ Run Sampling ------------------------------------------
function _ecmc_multichain_sample(
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_states::Vector{ECMCState} = [ECMCState(density, algorithm) for i in 1:algorithm.nchains] # TODO
)
    samples = [] #TODO: type, don't push
    Threads.@threads for c in 1:algorithm.nchains
        samples_per_chain = _ecmc_sample(density, algorithm, ecmc_state=ecmc_states[c], chainid=c) 
        push!(samples, samples_per_chain)
    end

    samples, ecmc_states
end

function _ecmc_sample(
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_state::ECMCState = ECMCState(density, algorithm),
    chainid::Int64 = 0, 
) 
    T = typeof(ecmc_state.C)
    samples = Array{T}(undef, algorithm.nsamples)
   
    @showprogress 1 "Run ECMC sampling for chain $chainid" for i in 1:algorithm.nsamples
        samples[i] = _run_ecmc!(ecmc_state, density, algorithm)
    end

    #burnin
    if algorithm.nburnin < algorithm.nsamples
        samples = samples[algorithm.nburnin+1:end]
    else
        println("burnin > samples in a chain! So nothing got burned.") # TODO give error before or have BAT do it
    end

    return samples
end


# This is the actual ECMC algorithm
function _run_ecmc!(
    ecmc_state::AbstractECMCState,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler,
)
    D = totalndof(density)
    @unpack C, current_energy, lift_vector, delta, remaining_jumps_before_refresh = ecmc_state
    remaining_jumps_before_sample = algorithm.chain_length

    while remaining_jumps_before_sample > 0
        ecmc_state.n_steps += 1
        remaining_jumps_before_sample -= 1
        remaining_jumps_before_refresh -= 1

        proposed_C = C + lift_vector * delta
        proposed_C_energy = _energy_function(density, proposed_C) 

        p_accept = exp.(-(proposed_C_energy - current_energy))
    
        u = rand(Uniform(0.0, 1.0))
        if u <= p_accept
            C = proposed_C
            current_energy = proposed_C_energy
            update_ecmc_state_accept!(ecmc_state, delta)
            delta = tune_delta(algorithm.tuning, delta, ecmc_state)

        else
            old_lift_vector = lift_vector
            lift_vector = _change_direction(algorithm.direction_change, C, current_energy, delta, lift_vector, proposed_C, density)
            
            update_ecmc_state_reject!(ecmc_state, delta)
            
            delta = tune_delta(algorithm.tuning, delta, ecmc_state)
   
            # TODO: move to direction changes
            if old_lift_vector != -lift_vector
                ecmc_state.n_acc_lifts += 1
            end   
        end

        if remaining_jumps_before_refresh <= 0 
            remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
            lift_vector = refresh_lift_vector(D)
            delta = refresh_delta(ecmc_state.step_amplitude, ecmc_state.step_var, ecmc_state.delta, ecmc_state.variation_type)
        end
    end
    
    @pack! ecmc_state = C, current_energy, lift_vector, delta, remaining_jumps_before_refresh  

    return C
end


# the following updates are different for ECMCState & ECMCTunerStates:

function update_ecmc_state_accept!(ecmc_state::ECMCState, delta)
    ecmc_state.mfp += delta
    ecmc_state.n_acc += 1
    ecmc_state.mfps += 1 
end

function update_ecmc_state_accept!(ecmc_state::ECMCTunerState, delta)
    ecmc_state.mfp += delta
    ecmc_state.n_acc += 1
    ecmc_state.mfps += 1 
    push!(ecmc_state.delta_arr, delta)
    push!(ecmc_state.acc_C, ecmc_state.n_acc)
    ecmc_state.step_acc = 1 # for google tuning
end


function update_ecmc_state_reject!(ecmc_state::ECMCTunerState, delta)
    ecmc_state.n_lifts += 1
    push!(ecmc_state.acc_C, ecmc_state.n_acc)
    push!(ecmc_state.delta_arr, delta)
    push!(ecmc_state.mfps_arr, ecmc_state.mfps)
    ecmc_state.step_acc = 0 # for google tuning
    ecmc_state.mfps = 0
end

function update_ecmc_state_reject!(ecmc_state::ECMCState, delta)
    ecmc_state.n_lifts += 1
    ecmc_state.mfps = 0
end


function tune_delta(tuning::ECMCTuner, delta::Float64, ecmc_state::ECMCState)
    return delta
end 
 
function tune_delta(tuning::MFPSTuner, delta::Float64, ecmc_state::ECMCTunerState)
    return adapt_delta(tuning.adaption_scheme, delta, ecmc_state, tuning)
end 





function refresh_lift_vector(D)
    v = rand(Normal(0., 1.), D)
    return normalize(v)
end


# "Converting" the posterior distribution to an energy and getting the gradient
_energy_function(density::AbstractMeasureOrDensity, x) = -logdensityof(density, x)

function _energy_gradient(density::AbstractMeasureOrDensity, x)
    energy_f = x -> _energy_function(density, x)

    return ForwardDiff.gradient(energy_f, x)
end

function _energy_difference(
    density::AbstractMeasureOrDensity, 
    new_x::Vector{Float64}, 
    old_x::Vector{Float64}
) 
    new_energy = _energy_function(density, new_x) 
    old_energy = _energy_function(density, old_x)

    return (new_energy - old_energy)
end


function _energy_difference(
    density::AbstractMeasureOrDensity, 
    new_x::Vector{Float64}, 
    current_energy::Float64
) 
    new_energy = _energy_function(density, new_x) 

    return (new_energy - current_energy)
end

#----- convert samples ----------------------------------------------
function  convert_to_BAT_samples(samples, density)
    new_samples = []

    nchains = length(samples)
    #println("While converting: this is nchains = ", nchains)
    #println("and this is samples = ", samples)
    for c in 1:nchains
        #println("and this is c = ", c)
        #println("and samples[c] = ", samples[c])
        n_samples = length(samples[c])
        sample_id = fill(BAT.MCMCSampleID(c, 0, 0, 0), n_samples)

        logvals = map(logdensityof(density), samples[c])
        weights = ones(n_samples)

        dsv = DensitySampleVector(samples[c], logvals,  weight = weights, info = sample_id)
        push!(new_samples, dsv)
    end

    return reduce(vcat, new_samples)
end 






#----- Check Tuning Convergence ----------------------------------------------------------
function check_tuning_convergence!(
    ecmc_tuner_state::ECMCTunerState,
    tuning_convergence_check::AcceptanceRatioConvergence,
)
    target_acc = tuning_convergence_check.target_acc
    acc_C = ecmc_tuner_state.acc_C # TODO: make acc_C & delta_arr fixed-size (N) arrays that are updated in rolling fahsion
    
    # NOTE: N is increasing with n_steps. Mean of growing array gets slow.
    N = Int(floor(tuning_convergence_check.Npercent * ecmc_tuner_state.n_steps)) #user input
    #mean_acc_C = mean(acc_C[end-N+1:end])  
    mean_acc_C = (acc_C[end] - acc_C[end-N])/N

    #mean_has_converged = abs(mean_acc_C - target_acc) < tuning_convergence_check.rel_dif_mean #* target_acc 
    mean_has_converged = abs(mean_acc_C - target_acc) < tuning_convergence_check.abs_dif_mean



    #-----checking the std of the mean accuracy of the last n_steps_eval to see if its low enough for convergence----
    current_acc_arr = []
    n_steps_eval = 180+Int(floor(N/2))
    if ecmc_tuner_state.n_steps > N+n_steps_eval
        for i in 1:n_steps_eval
            c_mean = (acc_C[end-i+1] - acc_C[end-N-i+1])/N
            push!(current_acc_arr, c_mean)
        end
        standard_deviation_is_low_enough = std(current_acc_arr) < tuning_convergence_check.standard_deviation
    else
        standard_deviation_is_low_enough = false
    end
    #standard_deviation_is_low_enough = std(ecmc_tuner_state.delta_arr[end-N:end]) < 0.1#tuning_convergence_check.standard_deviation
    #------

    mean_delta = mean(ecmc_tuner_state.delta_arr[end-N:end])
    #std_delta = std(ecmc_tuner_state.delta_arr[end-N:end]/mean_delta)

    ecmc_tuner_state.tuned_delta = mean_delta
    #ecmc_tuner_state.step_var = std_delta

    enough_steps = ecmc_tuner_state.n_steps > 0.5*10^4
    enough_steps = true # true, if no minimum steps needed

    if mean_has_converged & standard_deviation_is_low_enough & enough_steps
        #@show mean_delta
        #@show abs(mean_acc_C - target_acc) / target_acc
        #@show std(acc_C[end-N+1:end])

        return true#, tuned_algorithm
    end 

    return false#, tuned_algorithm
end