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
    #step_var::Float64 = 0.5
    remaining_jumps_before_refresh::Int = 5
    direction_change::AbstractECMCDirection = ReverseDirection()
    tuning::ECMCTuner = MFPSTuner(target_mfps=5)
    factorized = false #TODO
end
export ECMCSampler


@with_kw mutable struct ECMCState <: AbstractECMCState
    C::Vector{Float64} = []
    lift_vector::Vector{Float64} = []
    delta::Float64 = 1.
    step_amplitude::Float64 = 1.
    step_var::Float64 = 1.
    remaining_jumps_before_refresh::Int64 = 1
    n_steps::Int64 = 0
    n_acc::Int64 = 0
    n_lifts::Int64 = 0
    n_acc_lifts::Int64 = 0
    mfp::Float64 = 0.
    mfps::Int64 = 0
end
export ECMCState


function ECMCState(density::AbstractMeasureOrDensity, algorithm::ECMCSampler)
    D = totalndof(density)
    initial_samples = [rand(BAT.getprior(density)) for i in 1:algorithm.nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:algorithm.nchains]

    ecmc_states = [ECMCState(
        C = initial_samples[i], 
        lift_vector = lift_vectors[i], 
        delta = algorithm.step_amplitude, 
        step_amplitude = algorithm.step_amplitude, 
        step_var = 0.1*algorithm.step_amplitude, 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
        ) for i in 1:algorithm.nchains]

    return ecmc_states
end 



# for tuning: includes additional arrays for mfps, delta, accepted C
@with_kw mutable struct ECMCTunerState <: AbstractECMCState
    C::Vector{Float64} = []
    lift_vector::Vector{Float64} = []
    delta::Float64 = 1.
    tuned_delta::Float64 = 1. 
    remaining_jumps_before_refresh::Int64 = 1
    n_steps::Int64 = 0
    n_acc::Int64 = 0
    n_lifts::Int64 = 0
    n_acc_lifts::Int64 = 0
    mfp::Float64 = 0.
    mfps::Int64 = 0
    mfps_arr::Vector{Int64} = []
    delta_arr::Vector{Float64} = []
    acc_C::Vector{Float64} = []
end
export ECMCTunerState


#TODO: make function calling ECMCTunerState nchains times
function ECMCTunerState(density::AbstractMeasureOrDensity, algorithm::ECMCSampler)
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:algorithm.nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:algorithm.nchains]
    
    delta = algorithm.step_amplitude

    ecmc_tuner_states = [ECMCTunerState(
        C = initial_samples[i], 
        lift_vector = lift_vectors[i], 
        delta = delta, 
        tuned_delta = delta, 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh, 
        delta_arr = [delta, ]
        )  for i in 1:algorithm.nchains]

    return ecmc_tuner_states
end 



# create ECMCState after tuning
#TODO: nchains
function ECMCState(ecmc_tuner_state::ECMCTunerState, algorithm::ECMCSampler)
    return ECMCState(
        C = ecmc_tuner_state.C, 
        lift_vector = ecmc_tuner_state.lift_vector, 
        delta = ecmc_tuner_state.tuned_delta, 
        step_amplitude = ecmc_tuner_state.tuned_delta, 
        step_var = 0.1*ecmc_tuner_state.tuned_delta, 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
        )
end 


function ECMCState(ecmc_tuner_states::Vector{ECMCTunerState}, algorithm::ECMCSampler)
    return [ECMCState(
        C = e.C, 
        lift_vector = e.lift_vector, 
        delta = e.tuned_delta, 
        step_amplitude = e.tuned_delta, 
        step_var = 0.1*e.tuned_delta, 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
        ) for e in ecmc_tuner_states]
end 


