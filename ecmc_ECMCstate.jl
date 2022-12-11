"""
    struct ECMCSampler <: AbstractSamplingAlgorithm

"""
@with_kw struct ECMCSampler{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = NoDensityTransform()
    nsamples::Int = 10^4
    nburnin::Int = Int(floor(0.1*nsamples)) # TODO ??
    nchains::Int = 4
    chain_length::Int = 50
    step_amplitude::Float64 = 0.5
    step_var::Float64 = 0.5
    remaining_jumps_before_refresh::Int = 5
    direction_change::AbstractECMCDirection = ReverseDirection()
    tuning::ECMCTuner = MFPSTuner(target_mfps=5)
    factorized = false #TODO
end
export ECMCSampler


mutable struct ECMCState <: AbstractECMCState
    C::Vector{Float64}
    lift_vector::Vector{Float64}
    delta::Float64  
    step_amplitude::Float64
    step_var::Float64                
    remaining_jumps_before_refresh::Int64
    n_steps::Int64
    n_acc::Int64
    n_lifts::Int64
    n_acc_lifts::Int64
    mfp::Float64
    mfps::Int64
end
export ECMCState

function ECMCState(density::AbstractMeasureOrDensity, algorithm::ECMCSampler)
    D = totalndof(density)
    initial_samples = [rand(BAT.getprior(density)) for i in 1:algorithm.nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:algorithm.nchains]
    
    delta = algorithm.step_amplitude # for tuning only  #TODO important?
    #delta = refresh_delta(algorithm.step_amplitude, algorithm.step_var)  #for sampling

    jumps_before_refresh = algorithm.remaining_jumps_before_refresh

    ecmc_states = [ECMCState(initial_samples[i], lift_vectors[i], delta, algorithm.step_amplitude, algorithm.step_var, jumps_before_refresh, 
    0, 0, 0, 0, 0., 0,) for i in 1:algorithm.nchains]

    return ecmc_states
end 



# for tuning: includes additional arrays for mfps, delta, accepted C
mutable struct ECMCTunerState <: AbstractECMCState
    C::Vector{Float64}
    lift_vector::Vector{Float64}
    delta::Float64
    tuned_delta::Float64                  
    remaining_jumps_before_refresh::Int64
    n_steps::Int64
    n_acc::Int64
    n_lifts::Int64
    n_acc_lifts::Int64
    mfp::Float64
    mfps::Int64
    mfps_arr::Vector{Int64}
    delta_arr::Vector{Float64}
    acc_C::Vector{Float64}
end
export ECMCTunerState

#TODO: make function calling ECMCTunerState nchains times
function ECMCTunerState(density::AbstractMeasureOrDensity, algorithm::ECMCSampler)
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:algorithm.nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:algorithm.nchains]
    
    delta = algorithm.step_amplitude
    jumps_before_refresh = algorithm.remaining_jumps_before_refresh

    ecmc_tuner_states = [ECMCTunerState(initial_samples[i], lift_vectors[i], delta, delta, jumps_before_refresh, 
    0, 0, 0, 0, 0., 0, [], [delta, ], [])  for i in 1:algorithm.nchains]

    return ecmc_tuner_states
end 



# create ECMCState after tuning
#TODO: nchains
function ECMCState(ecmc_tuner_state::ECMCTunerState, algorithm::ECMCSampler)
    C = ecmc_tuner_state.C
    lift_vector = ecmc_tuner_state.lift_vector
    delta = ecmc_tuner_state.tuned_delta
    step_amplitude = ecmc_tuner_state.tuned_delta
    jumps_before_refresh = algorithm.remaining_jumps_before_refresh
    ECMCState(C, lift_vector, delta, step_amplitude, algorithm.step_var, jumps_before_refresh, 
    0, 0, 0, 0, 0., 0,)
end 

function ECMCState(ecmc_tuner_states::Vector{ECMCTunerState}, algorithm::ECMCSampler)
    return [ECMCState(e.C, e.lift_vector, e.tuned_delta, e.tuned_delta, algorithm.step_var, algorithm.remaining_jumps_before_refresh, 
        0, 0, 0, 0, 0., 0,) for e in ecmc_tuner_states]
end 


