"""
    struct ECMCSampler <: AbstractSamplingAlgorithm
    
Holds user specified information and default starting parameters regarding the sampling process.
    
# Fields
- `trafo::TR`: specified density transformation function.
- `nsamples::Integer`: the number of desired samples.
- `nburnin::Integer`: the number of burned samples.
- `nchains::Integer`: the number of used chains.
- `chain_length::Integer`: the number of jumps before a sample is returned.
- `step_amplitude::Float64`: the starting value for the length of one jump.
- `step_var::Float64`: the amount of variation in the jump length multiplier.
- `remaining_jumps_before_refresh::Integer`: the remaining jumps before a refresh of the direction vector happens.
- `direciton_change::AbstractECMCDirection`: the specified direction change algorithm.
- `variation_type::AbstractStepAmplitudeVariation`: the specified algorithm to variate delta around the step_amplitude.
- `tuning::ECMCTuner`: the specified tuning algorithm.
- `factorized::Bool`: specifies if the density should be used in a factorized state.
"""
@with_kw struct ECMCSampler{TR<:AbstractTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = NoDensityTransform()
    nsamples::Int = 10^4
    nburnin::Int = 0
    nchains::Int = 4
    chain_length::Int = 5 #remaining_jumps_before_sample
    step_amplitude::Float64 = 1.
    step_var::Float64 = 0.2
    remaining_jumps_before_refresh::Int = 50
    direction_change::AbstractECMCDirection = ReflectDirection()
    variation_type::AbstractStepAmplitudeVariation = UniformVariation()
    tuning::ECMCTuner = MFPSTuner(target_mfps=5)
    factorized = false #TODO
end
export ECMCSampler


"""
    mutable struct ECMCState <: AbstractECMCState

Holds information about the current state of sampling.

# Fields
- `C::Vector{Float64}`: the current sample location (in parameterspace).
- `lift_vector::Vector{Float64}`: the current jump direction vector.
- `delta::Float64`: the current value for the multiplier of the jump length.
- `step_amplitude::Float64`: the default value for the multiplier of the jump length.
- `step_var::Float64`: the amount of variation in the jump length multiplier.
- `variation_type::AbstractStepAmplitudeVariation`: the specified algorithm to variate delta around the step_amplitude.
- `remaining_jumps_before_refresh::Int64`: the remaining jumps before a refresh of the direction vector happens.
- `n_steps::Int64=0`: the current number of steps taken.
- `n_acc::Int64=0`: the current number of accepted new sample proposals.
- `n_lifts::Int64=0`: the current number of performed direction changes.
- `n_acc_lifts::Int64=0`: the current number of performed direction changes which have not reversed the direction.
- `mfp::Float64=0.`: the sum of jump length multipliers after accepted jumps.
- `mfps::Int64=0`: the current number of accepted new sample locations in a row.

See also [`ECMCState(::AbstractMeasureOrDensity, ::ECMCSampler)`](@ref).
"""
@with_kw mutable struct ECMCState <: AbstractECMCState
    C::Vector{Float64} = []
    current_energy::Float64 = 0
    lift_vector::Vector{Float64} = []
    delta::Float64 = 1.
    step_amplitude::Float64 = 1.
    step_var::Float64 = 0.1
    variation_type::AbstractStepAmplitudeVariation = UniformVariation()
    remaining_jumps_before_refresh::Int64 = 1
    n_steps::Int64 = 0
    n_acc::Int64 = 0
    n_lifts::Int64 = 0
    n_acc_lifts::Int64 = 0
    mfp::Float64 = 0.
    mfps::Int64 = 0
end
export ECMCState


"""
    function ECMCState(x::AbstractMeasureOrDensity, y::ECMCSampler)

Create and return a vector of ECMCStates.

The length of the returned vector is equal to the number of chains specified in the `ECMCSampler`.
Each start sample is sampled from the prior and the start direction vector is specified by the `refresh_lift_vector` function.

See also [`ECMCState`](@ref), [`ECMCSampler`](@ref), [`refresh_lift_vector`](@ref).
"""
function ECMCState(density::AbstractMeasureOrDensity, algorithm::ECMCSampler)
    D = totalndof(density)
    initial_samples = [rand(BAT.getprior(density)) for i in 1:algorithm.nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:algorithm.nchains]

    ecmc_states = [ECMCState(
        C = initial_samples[i], 
        current_energy = -logdensityof(density, initial_samples[i]),
        lift_vector = lift_vectors[i], 
        delta = algorithm.step_amplitude, 
        step_amplitude = algorithm.step_amplitude, 
        step_var = 0.1*algorithm.step_amplitude, 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
        ) for i in 1:algorithm.nchains]

    return ecmc_states
end 


"""

    mutable struct ECMCTunerState <: AbstractECMCState

Holds information about the current state of tuning.

Acts similar to `ECMCState`, but with additional information relevant for tuning only.

# Fields
- `C::Vector{Float64}`: the current sample location (in parameterspace).
- `lift_vector::Vector{Float64}`: the current jump direction vector.
- `delta::Float64`: the current value for the multiplier of the jump length.
- `tuned_delta::Float64`: 
- `remaining_jumps_before_refresh::Int64`: the remaining jumps before a refresh of the direction vector happens.
- `n_steps::Int64=0`: the current number of steps taken.
- `n_acc::Int64=0`: the current number of accepted new sample proposals.
- `n_lifts::Int64=0`: the current number of performed direction changes.
- `n_acc_lifts::Int64=0`: the current number of performed direction changes which have not reversed the direction.
- `mfp::Float64=0.`: the sum of jump length multipliers after accepted jumps.
- `mfps::Int64=0`: the current number of accepted new sample locations in a row.
- `mfps_arr::Vector{Int64}=[]`: the maximal reached mfps values.
- `delta_arr::Vector{Float64}=[]`: the values for delta after an accepted jump.
- `acc_C::Vector{Float64}=[]`: n_acc at each step.

See also [`ECMCState`](@ref).
"""
@with_kw mutable struct ECMCTunerState <: AbstractECMCState # for tuning: includes additional arrays for mfps, delta, accepted C
    C::Vector{Float64} = []
    current_energy::Float64 = 0
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

    step_amplitude::Float64 = 1. # so refresh_delta can be in a seperate file (although you could change the tuning to only use step_amplitude instead of delta and then not have one dead field)
    step_var::Float64 = 0.1 # convergence can write to here and then ecmc_states can get the tuned version of step_var
    variation_type::AbstractStepAmplitudeVariation = NoVariation()

    γ::Float64 = 0 # for google tuning
    step_acc::Bool = 0 # for google tuning # was this step accepted? y : n
    α = 0.1 # for new google tuning
    α_acc_err = 1 # for new google tuning

    params::Vector{Float64} = [] # for optimizing the naive adaption (and could be used for optimizing α in google tuning etc)
    
end
export ECMCTunerState


"""
    function ECMCTunerState(x::AbstractMeasureOrDensity, y::ECMCSampler)

Create and return a vector of ECMCTunerStates.

The length of the returned vector is equal to the number of chains specified in the `ECMCSampler`.

See also [`ECMCTunerState`](@ref), [`ECMCSampler`](@ref).
"""
function ECMCTunerState(density::AbstractMeasureOrDensity, algorithm::ECMCSampler) # TODO: make function calling ECMCTunerState nchains times
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:algorithm.nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:algorithm.nchains]
    
    delta = algorithm.step_amplitude

    ecmc_tuner_states = [ECMCTunerState(
        C = initial_samples[i], 
        current_energy = -logdensityof(density, initial_samples[i]),
        lift_vector = lift_vectors[i], 
        delta = delta, 
        tuned_delta = delta, 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh, 
        delta_arr = [delta, ],
        step_var = algorithm.step_var,
        α = algorithm.tuning == ECMCNoTuner() ? 0 : algorithm.tuning.starting_alpha,
        params = [0.98,
        0.207210542888343,
        0.0732514723260891,
        0.4934509024569294,
        17.587673168668637,
        2.069505973296211,
        0.6136869940758715,
        163.53188824455017] # TO BE REMOVED
        )  for i in 1:algorithm.nchains]

    return ecmc_tuner_states
end 


"""
    function ECMCState(x::ECMCTunerState, y::ECMCSampler)

Return an `ECMCState` created from an `ECMCTunerState` after the tuning finished.
"""
function ECMCState(ecmc_tuner_state::ECMCTunerState, algorithm::ECMCSampler)
# create ECMCState after tuning
#TODO: nchains
    return ECMCState(
        C = ecmc_tuner_state.C, 
        current_energy = ecmc_tuner_state.current_energy,
        lift_vector = ecmc_tuner_state.lift_vector, 
        delta = ecmc_tuner_state.tuned_delta, 
        step_amplitude = ecmc_tuner_state.tuned_delta, 
        step_var = ecmc_tuner_state.step_var,
        variation_type = algorithm.variation_type,
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
        )
end 


"""
    function ECMCState(x::Vector{ECMCTunerState}, y::ECMCSampler)

Return a vector of ECMCStates created from a vector of ECMCTunerStates after the tuning finished.

See also [`ECMCState(::ECMCTunerState, ::ECMCSampler)`](@ref), [`ECMCState`](@ref), [`ECMCTunerState`](@ref).
"""
function ECMCState(ecmc_tuner_states::Vector{ECMCTunerState}, algorithm::ECMCSampler)
    return [ECMCState(
        C = e.C, 
        current_energy = e.current_energy,
        lift_vector = e.lift_vector, 
        delta = e.tuned_delta, 
        step_amplitude = e.tuned_delta, 
        step_var = e.step_var, 
        variation_type = algorithm.variation_type,
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh
        ) for e in ecmc_tuner_states]
end 


