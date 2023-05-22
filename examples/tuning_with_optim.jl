
using Optim

#ecmc_tuning.jl
@with_kw struct OptimTuner{A<:ECMCStepSizeAdaptor, C<:ECMCTuningConvergenceCheck} <: ECMCTuner
    target_mfps::Int64 = 5
    max_n_steps::Int64 = 10^4
    adaption_scheme::A = NoAdaption() 
    tuning_convergence_check::C = AcceptanceRatioConvergence(target_acc = target_mfps / (target_mfps  + 1)) # doesn't matter because _ecmc_tuning won't be called
end
export OptimTuner


#ecmc_tuning.jl
struct NoAdaption <: ECMCStepSizeAdaptor end
export NoAdaption

function adapt_delta(adaption_scheme::NoAdaption, delta, ecmc_state, tuner::OptimTuner)
    return delta
end


#ecmc_sampler.jl
function tune_delta(tuning::OptimTuner, delta::Float64, ecmc_state::ECMCTunerState)
    return adapt_delta(tuning.adaption_scheme, delta, ecmc_state, tuning)
end 



function iterator!(new_delta, iter_steps, target_acc, ecmc_tuner_state::ECMCTunerState, density::AbstractMeasureOrDensity, algorithm::ECMCSampler)
    delta = abs(new_delta[1]) # optim could go for lowering delta into negativ numbers, but we dont want to restrict it
    if delta == 0
        delta = 1e-6
    end
    @pack! ecmc_tuner_state = delta

    for i in 1:iter_steps
        _run_ecmc!(ecmc_tuner_state, density, algorithm)
    end

    N = iter_steps #last N jumps are evaluated
    if ecmc_tuner_state.n_steps > N
        current_acc = (ecmc_tuner_state.acc_C[end]*ecmc_tuner_state.n_steps - ecmc_tuner_state.acc_C[end-N]*(ecmc_tuner_state.n_steps-N))/N
        r_value = abs(target_acc - current_acc)
    else
        current_acc = ecmc_tuner_state.n_acc/ecmc_tuner_state.n_steps #letzte N steps
        r_value = abs(target_acc - current_acc)
    end



    println("at step ", ecmc_tuner_state.n_steps, " with current acc = ", current_acc, " and return value = ", r_value)


    return r_value^2
end


#ecmc_sampler.jl
function _ecmc_tuning(
    tuner::OptimTuner,
    density::AbstractMeasureOrDensity,
    algorithm::ECMCSampler;
    ecmc_tuner_state::ECMCTunerState = ECMCTunerState(density, algorithm),
    chainid::Int64 = 0,
) 

    inital_start_iterations = 1000 # if needed
    for i in 1:inital_start_iterations
        _run_ecmc!(ecmc_tuner_state, density, algorithm)
    end


    iter_steps = 1000
    #optim_chains = 4
    target_acc =  tuner.target_mfps / (tuner.target_mfps  + 1)


    #for chain in 1:optim_chains

        #generate new ecmc_tuner_state
        #optimize each state seperatly and with iterations = old_iteration/optim_chains
    res = optimize(delta -> iterator!(delta, iter_steps, target_acc, ecmc_tuner_state, density, algorithm), # delta is a vector here (with one entry currently)
            [ecmc_tuner_state.delta],
            #[0.1],
            method = NelderMead(),
            g_tol = 1e-8,
            iterations = Int(floor(tuner.max_n_steps/iter_steps)),
            store_trace = true,
            show_trace = false)

    tuned_delta = abs(Optim.minimizer(res)[1]) # since the input delta was a vector (with one entry currently), the minimzer is too
    println("tuned delta = ", tuned_delta)
    println("optimizer res", res)
    @pack! ecmc_tuner_state = tuned_delta
        #get mean of each tuned delta and generate only 1 state with the new mean delta but all acc_C etc in one array
    #end

    
    tuning_samples = [] # to-do?
    return  tuning_samples, ecmc_tuner_state
end




