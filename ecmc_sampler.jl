function bat_sample_impl(
    rng::AbstractRNG,
    target::AnyMeasureOrDensity,
    algorithm::ECMCSampler
)
    density_notrafo = convert(AbstractMeasureOrDensity, target)
    density, trafo = transform_and_unshape(algorithm.trafo, density_notrafo)
    shape = varshape(density)

    tuning_samples, ecmc_tuning_state, tuned_algorithm = _ecmc_tuning(algorithm.tuning, density, algorithm)
    ecmc_state = ECMCState(ecmc_tuning_state, tuned_algorithm)
    samples, ecmc_state = _ecmc_sample(density, tuned_algorithm, ecmc_state = ecmc_state)

    logvals = map(logdensityof(density), samples)
    weights = ones(length(samples))

    samples_trafo = shape.(DensitySampleVector(samples, logvals, weight = weights))
    samples_notrafo = inverse(trafo).(samples_trafo)

    return (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, ecmc_state=ecmc_state, ecmc_tuning_state=ecmc_tuning_state)
end



function _ecmc_tuning(
    tuner::MFPSTuner,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_tuner_state::ECMCTunerState = ECMCTunerState(density, algorithm), #TODO allow to pass ECMCTunerState for multiple cycles
)
    T = typeof(ecmc_tuner_state.C)
    tuning_samples = T[] 

    # TODO: needed?
    sizehint!(ecmc_tuner_state.acc_C, algorithm.chain_length * algorithm.nsamples)
    sizehint!(ecmc_tuner_state.delta_arr, algorithm.chain_length * algorithm.nsamples)
    sizehint!(ecmc_tuner_state.mfps_arr, algorithm.chain_length * algorithm.nsamples)

    tuning_converged = false
    tuned_algorithm = algorithm
 
    @showprogress 1 "MFPS tuning" for i in 1:tuner.max_n_steps
        push!(tuning_samples, _run_ecmc(density, algorithm, ecmc_tuner_state))
        tuning_converged, tuned_algorithm = check_tuning_convergence(algorithm.tuning.tuning_convergence_check, algorithm, ecmc_tuner_state)
        
        tuning_converged ? break : nothing
    end 

    has_converged_str = tuning_converged ? " " : " has *not* "
    @info "MFPS tuning" * has_converged_str * "converged after $(ecmc_tuner_state.n_steps) steps. New step_amplitude = $(tuned_algorithm.step_amplitude)"

    mfp = ecmc_tuner_state.mfp/(algorithm.chain_length*algorithm.nsamples)
    mfps = ecmc_tuner_state.mfps/(algorithm.chain_length*algorithm.nsamples)
    n_acc = ecmc_tuner_state.n_acc / (algorithm.chain_length*algorithm.nsamples)
    n_acc_lifts = ecmc_tuner_state.n_acc_lifts / ecmc_tuner_state.n_lifts

    return tuning_samples, ecmc_tuner_state, tuned_algorithm
end


function _ecmc_tuning(
    tuner::ECMCNoTuner,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_tuner_state::ECMCTunerState = ECMCTunerState(density, algorithm), #TODO allow to pass ECMCTunerState for multiple cycles
)
    T = typeof(ecmc_tuner_state.C)
    tuning_samples = T[] 
    @info "No ECMC tuning"

    return tuning_samples, ecmc_tuner_state, algorithm
end


function _ecmc_sample(
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_state::ECMCState = ECMCState(density, algorithm),
)
    initial_sample = ecmc_state.C
    T = typeof(initial_sample)
    samples = Array{T}(undef, algorithm.nsamples)
   
    @showprogress 1 "Run ECMC sampling" for i in 1:algorithm.nsamples
        samples[i] = _run_ecmc(density, algorithm, ecmc_state)
    end
    
    total_steps = algorithm.chain_length*algorithm.nsamples
    mfp = ecmc_state.mfp / total_steps
    mfps = ecmc_state.mfps / total_steps
    n_acc = ecmc_state.n_acc / total_steps
    n_acc_lifts = ecmc_state.n_acc_lifts / ecmc_state.n_lifts

    return samples, ecmc_state
end



function _run_ecmc(
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler,
    ecmc_state::AbstractECMCState,
)
    D = totalndof(density)
    @unpack C, lift_vector, delta, remaining_jumps_before_refresh = ecmc_state
    remaining_jumps_before_sample = algorithm.chain_length

    while remaining_jumps_before_sample > 0
        ecmc_state.n_steps += 1
        remaining_jumps_before_sample -= 1
        remaining_jumps_before_refresh -= 1

        proposed_C = C + lift_vector * delta
        p_accept = exp.(-_energy_difference(density, proposed_C, C))
    
        u = rand(Uniform(0.0, 1.0))
        if u <= p_accept
            C = proposed_C
            update_ecmc_state_accept!(ecmc_state, delta)
        else
            old_lift_vector = lift_vector
            lift_vector = _change_direction(algorithm.direction_change, C, delta, lift_vector, proposed_C, density)
            
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
            delta = refresh_delta(ecmc_state, algorithm.step_amplitude, algorithm.step_var)
        end
    end

    @pack! ecmc_state = C, lift_vector, delta, remaining_jumps_before_refresh  

    return C
end


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
    push!(ecmc_state.acc_C, ecmc_state.n_acc/ecmc_state.n_steps)
end


function update_ecmc_state_reject!(ecmc_state::ECMCTunerState, delta)
    ecmc_state.n_lifts += 1
    length(ecmc_state.acc_C) < 1 ? push!(ecmc_state.acc_C, 0) : push!(ecmc_state.acc_C, ecmc_state.acc_C[end])
    push!(ecmc_state.delta_arr, delta)
    push!(ecmc_state.mfps_arr, ecmc_state.mfps)
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


function refresh_delta(ecmc_state::ECMCTunerState, delta, step_var) 
    return delta
end

function refresh_delta(ecmc_state::ECMCState, step_amplitude, step_var)
    #return rand(Normal(step_amplitude, step_var*step_amplitude))
    u = rand(Uniform(1-step_var, 1+step_var))
    return u * step_amplitude
end



function refresh_lift_vector(D)
    v = rand(Normal(0., 1.), D)
    return normalize(v)
end



_energy_function(density::AbstractMeasureOrDensity, x) = -logdensityof(density, x)

function _energy_gradient(density::AbstractMeasureOrDensity, x)
    energy_f = x -> _energy_function(density, x)

    return ForwardDiff.gradient(energy_f, x)
end

function _energy_difference(
    density::AbstractMeasureOrDensity, 
    new_x, 
    old_x
) 
    new_energy = _energy_function(density, new_x) 
    old_energy = _energy_function(density, old_x)

    return (new_energy - old_energy)
end


#----- Check Tuning Convergence ----------------------------------------------------------
function check_tuning_convergence(
    tuning_convergence_check::AcceptanceRatioConvergence, 
    algorithm::ECMCSampler, 
    ecmc_tuner_state::ECMCTunerState
)
    target_acc = tuning_convergence_check.target_acc
    acc_C = ecmc_tuner_state.acc_C # TODO: make acc_C & delta_arr fixed-size (N) arrays that are updated in rolling fahsion
    
    # NOTE: N is increasing with n_steps. Mean of growing array gets slow.
    N = minimum([length(acc_C), Int(floor(tuning_convergence_check.Npercent * ecmc_tuner_state.n_steps))]) #user input
    mean_acc_C = mean(acc_C[end-N+1:end])  

    mean_has_converged = abs(mean_acc_C - target_acc) < tuning_convergence_check.rel_dif_mean * target_acc 
    variance_is_low_enough = std(acc_C[end-N+1:end]) < tuning_convergence_check.variance

    mean_delta = mean(ecmc_tuner_state.delta_arr[end-N+1:end])
    tuned_algorithm = reconstruct(algorithm, step_amplitude=mean_delta)

    if mean_has_converged & variance_is_low_enough
        @show mean_delta
        @show abs(mean_acc_C - target_acc) / target_acc
        @show std(acc_C[end-N+1:end])

        return true, tuned_algorithm
    end 

    return false, tuned_algorithm
end

